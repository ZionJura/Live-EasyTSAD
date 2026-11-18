"""Run trusted EasyTSAD experiments with canonical scientific outputs.

Scores, native evaluation JSON and runtime files live under ROOT/Results.
Attempt provenance, logs and append-only progress live under
EasyTSAD/Methods/<method>/reproduction/runs/<attempt>/<schema>/<dataset>.
No scientific result copies are created in LiveLeaderBoard/results/runs.
Compatible with Python 3.9 method environments.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import tempfile
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import toml

# EasyTSAD.Controller must be imported before EasyTSAD.Methods (circular import).
import EasyTSAD.Controller  # noqa: F401
from EasyTSAD.Controller.PathManager import PathManager
from EasyTSAD.DataFactory.LoadData import load_data
from EasyTSAD.Evaluations import Performance
from EasyTSAD.Evaluations.Protocols import (EventF1PA, PointAuprcPA, PointF1,
                                            PointF1PA, PointPrc, PointRoc)
from EasyTSAD.TrainingSchema import MTS, AllInOne, Naive, ZeroShot, ZeroShotCrossDS
from EasyTSAD.utils import update_nested_dict
from gpu_monitor import GPUMonitor
from progress import append_event
from export import export_lock, _atomic

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "LiveLeaderBoard"
RESULTS_DIR = LB_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
INDEX = RESULTS_DIR / "index.jsonl"
METHODS_DIR = ROOT / "EasyTSAD" / "Methods"
GLOBAL_CFG = ROOT / "EasyTSAD" / "Controller" / "GlobalCfg.toml"

SCHEMAS = {"naive": Naive, "all_in_one": AllInOne, "zero_shot": ZeroShot, "mts": MTS,
           "zero_shot_cross_ds": ZeroShotCrossDS}

# Stable metric keys -> protocol factories. The keys are what metrics.json,
# experiment specs and the leaderboard export refer to.
EVALS = {
    "point_f1": lambda: PointF1PA(),
    "event_f1": lambda: EventF1PA(mode="log", base=3),
    "event_f1_squeeze": lambda: EventF1PA(mode="squeeze"),
    "event_f1_raw": lambda: EventF1PA(mode="raw"),
    "auprc": lambda: PointPrc(),
    "auroc": lambda: PointRoc(),
    "auprc_pa": lambda: PointAuprcPA(),
    "point_f1_nopa": lambda: PointF1(),
}
DEFAULT_EVALS = ["point_f1", "event_f1", "auprc"]
TRACKED_PACKAGES = ["easytsad", "numpy", "scipy", "scikit-learn", "pandas", "torch",
                    "pytorch-lightning", "transformers", "tslearn", "matrixprofile", "stumpy"]

log = logging.getLogger("logger")  # EasyTSAD's global logger name


# ----------------------------------------------------------------- helpers
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def package_versions() -> dict:
    from importlib import metadata
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def environment_lock_path(env: str) -> Path | None:
    if not env:
        return None
    registry_path = ROOT / "envs/registry.toml"
    registry = toml.load(registry_path) if registry_path.exists() else {}
    spec = registry.get("envs", {}).get(env, {})
    project = ROOT / spec["project"] if spec.get("project") else ROOT / "envs" / env
    return project / "uv.lock"


def source_files(method_dir: Path) -> list[Path]:
    files = []
    for directory, children, names in os.walk(method_dir):
        children[:] = [name for name in children if name not in
                       {"__pycache__", ".venv", "venv", ".git", "reproduction"}]
        files.extend(Path(directory) / name for name in names if name.endswith(".py") or name == "config.toml")
    return sorted(files)


def env_info() -> dict:
    env = os.environ.get("TSAD_ENV", "")
    lock = environment_lock_path(env)
    info = {"name": env or None, "python": platform.python_version(),
            "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock and lock.exists() else None,
            "packages": package_versions(), "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, obj) -> None:
    """Durably replace one complete JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, indent=2, default=float, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def to_builtin(x):
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def resolve_method_config(method: str, dataset: str, spec: dict) -> tuple[dict, dict]:
    """Method config.toml (Default + per-dataset section) + spec overrides."""
    cfg_path = METHODS_DIR / method / "config.toml"
    cfg = toml.load(cfg_path) if cfg_path.exists() else {}
    data = {"preprocess": "raw", "diff_order": 0}
    data.update(cfg.get("Data_Params", {}))
    mp = cfg.get("Model_Params", {})
    hparams = copy.deepcopy(mp.get("Default", {}))
    hparams.update(copy.deepcopy(mp.get(dataset, {})))
    ov = spec.get("overrides", {}).get(method, {})
    data.update(ov.get("data", {}))
    hparams.update(ov.get("hparams", {}))
    hparams.update(ov.get("hparams_by_dataset", {}).get(dataset, {}))
    return data, hparams


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def existing_success(method: str, schema: str, dataset: str, chash: str) -> Path | None:
    candidates = list((METHODS_DIR / method / "reproduction/runs").glob("*/" + schema + "/" + dataset + "/manifest.json"))
    # Previously recorded leaderboard runs remain valid for skip compatibility.
    candidates += list((RUNS_DIR / method / schema / dataset).glob("*/manifest.json"))
    for path in sorted(candidates, reverse=True):
        man = json.loads(path.read_text())
        if man.get("status") == "success" and man.get("config_hash") == chash:
            if man.get("result_format") == "easytsad-v1":
                expected = man.get("result_sha256", {})
                if not expected or any(not (ROOT / man.get(k, k)).exists() or hashlib.sha256((ROOT / man.get(k, k)).read_bytes()).hexdigest() != v
                                       for k, v in expected.items()):
                    continue
            return path.parent
    return None


def attach_log(path: Path) -> logging.Handler:
    log.setLevel(logging.INFO)
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("(%(asctime)s) [%(levelname)s]: %(message)s"))
    log.addHandler(h)
    if not any(isinstance(x, logging.StreamHandler) and not isinstance(x, logging.FileHandler)
               for x in log.handlers):
        log.addHandler(logging.StreamHandler())
    return h


def mean_metrics(per_curve: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [per_curve[curve][k] for curve in sorted(per_curve) if k in per_curve[curve]]
        if not vals:
            continue
        fields = [f for f in vals[0] if f != "threshold"]
        out[k] = {f: float(np.mean([v[f] for v in vals])) for f in fields}
    return out


def diagnostic_flags(per_curve: dict, curve_flags: dict) -> dict:
    """Flag primitive score failures; low positive F1 never triggers diagnosis."""
    core = [record[key]["f1"] for record in per_curve.values() for key in ("point_f1", "event_f1")
            if key in record and "f1" in record[key]]
    return {"constant": any(flags.get("constant", False) for flags in curve_flags.values()),
            "all_zero": any(flags.get("all_zero", False) for flags in curve_flags.values()),
            "core_f1_zero": bool(core) and all(value == 0 for value in core)}


# ----------------------------------------------------------------- one run
def _cross_dataset_split(spec: dict) -> tuple[list[str], list[str]]:
    sources = spec.get("src_datasets")
    targets = spec.get("dst_datasets", spec.get("datasets"))
    for name, datasets in (("src_datasets", sources), ("dst_datasets or datasets", targets)):
        if (not isinstance(datasets, list) or not datasets
                or any(not isinstance(dataset, str) or not dataset.strip() for dataset in datasets)):
            raise ValueError("zero_shot_cross_ds requires an explicit nonempty list of dataset names for " + name)
        if len(set(datasets)) != len(datasets):
            raise ValueError("zero_shot_cross_ds requires unique dataset names in " + name)
    overlap = set(sources) & set(targets)
    if overlap:
        raise ValueError("zero_shot_cross_ds source and target datasets must be disjoint: " + repr(sorted(overlap)))
    if spec.get("curves"):
        raise ValueError("zero_shot_cross_ds requires full datasets; curve subsets are not supported")
    return list(sources), list(targets)


def publication_recovery_curves(previous: dict, run_dir: Path, identity: dict,
                                attempt_id: str, chash: str, score_dir: Path) -> list[str]:
    if (previous.get("status") != "failed" or not previous.get("run_id")
            or previous.get("attempt_id") != attempt_id
            or any(previous.get(key) != identity[key] for key in ("method", "schema", "dataset"))):
        raise ValueError("publication recovery requires a failed manifest from the same attempt and dataset")
    if previous.get("config_hash") != chash:
        raise ValueError("publication recovery config_hash mismatch")
    error = previous.get("error") or ""
    phase = previous.get("execution_phase")
    publication_failure = phase == "publication" and bool(error)
    if phase is None:
        publication_failure = "with export_lock():" in error and "in export_lock\n" in error
    if not publication_failure:
        raise ValueError("publication recovery requires an explicit publication failure")
    if previous.get("score_dir") != str(score_dir.relative_to(ROOT)):
        raise ValueError("publication recovery score directory mismatch")
    total = previous.get("total_curves")
    if not isinstance(total, int) or total <= 0 or previous.get("n_curves_scored") != total:
        raise ValueError("publication recovery requires complete scored curve coverage")
    recorded = previous.get("scored_curves")
    if recorded is None:
        recorded = []
        with (run_dir / "progress.jsonl").open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get("type") == "curve_evaluated"
                        and row.get("run_id") == previous["run_id"]
                        and all(row.get(key) == identity[key] for key in ("method", "schema", "dataset"))):
                    recorded.append(row.get("curve"))
    if (not isinstance(recorded, list) or len(recorded) != total
            or any(not isinstance(curve, str) or not curve or Path(curve).name != curve
                   or curve in (".", "..") for curve in recorded)
            or len(set(recorded)) != total):
        raise ValueError("publication recovery requires a complete, unique scored curve list")
    missing = [curve for curve in recorded if not (score_dir / (curve + ".npy")).is_file()]
    if missing:
        raise ValueError("publication recovery missing score files: " + repr(missing))
    return recorded


def run_one(spec: dict, method: str, dataset: str, run_id: str, force: bool) -> str:
    schema = spec["schema"]
    if schema == "zero_shot_cross_ds":
        src_datasets, dst_datasets = _cross_dataset_split(spec)
        if dataset not in dst_datasets:
            raise ValueError("zero_shot_cross_ds run dataset must be a declared target: " + repr(dataset))
    dtype = spec.get("dataset_type", "MTS" if schema == "mts" else "UTS")
    curves = spec.get("curves")
    eval_keys = spec.get("evals", DEFAULT_EVALS)
    seed = int(spec.get("seed", 2026))
    data_params, hparams = resolve_method_config(method, dataset, spec)

    gcfg = toml.load(GLOBAL_CFG)
    gcfg = update_nested_dict(gcfg, spec.get("global_cfg", {}))
    ds_cfg = gcfg["DatasetSetting"]
    ev_cfg = gcfg["EvalSetting"]
    margins = (0, 0)
    if ev_cfg.get("use_margin"):
        margins = tuple(ev_cfg["margin"].get(dataset, ev_cfg["margin"]["default"]))

    source = hashlib.sha256()
    for path in source_files(METHODS_DIR / method):
        source.update(str(path.relative_to(METHODS_DIR / method)).encode() + b"\0" + path.read_bytes())
    lock = environment_lock_path(os.environ.get("TSAD_ENV", ""))
    resolved = {"source_sha256": source.hexdigest(),
                "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock and lock.exists() else None,
                "method": method, "schema": schema, "dataset_type": dtype, "dataset": dataset,
                "curves": curves, "data_params": data_params, "hparams": hparams,
                "evals": eval_keys, "margins": list(margins), "seed": seed,
                "train_proportion": ds_cfg["train_proportion"],
                "valid_proportion": ds_cfg["valid_proportion"],
                "transfer": gcfg["Transfer"] if schema in ("zero_shot", "zero_shot_cross_ds") else None}
    if schema == "zero_shot_cross_ds":
        resolved.update(src_datasets=src_datasets, dst_datasets=dst_datasets)
    chash = config_hash(resolved)
    recovering = spec.get("recover_publication") is True
    if not force and not recovering:
        prev = existing_success(method, schema, dataset, chash)
        if prev is not None:
            print(f"[skip] {method}/{schema}/{dataset}: identical successful run {prev.name}")
            return "skipped"

    attempt_id = os.environ.get("TSAD_ATTEMPT_ID", run_id)
    if Path(attempt_id).name != attempt_id or attempt_id in (".", ".."):
        raise ValueError("invalid attempt ID")
    run_dir = METHODS_DIR / method / "reproduction/runs" / attempt_id / schema / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    # Serialize writes to the same canonical dataset even across independent runners.
    dataset_lock = open(run_dir.parents[3] / ("." + schema + "-" + dataset + ".lock"), "a")
    fcntl.flock(dataset_lock, fcntl.LOCK_EX)
    score_dir = ROOT / "Results/Scores" / method / schema / dataset
    eval_dir = ROOT / "Results/Evals" / method / schema / dataset
    runtime_dir = ROOT / "Results/RunTime" / method / schema / dataset
    metric_keys = {k: EVALS[k]().name for k in eval_keys}
    identity = {"run_id": run_id, "method": method, "schema": schema, "dataset": dataset}
    def emit(event, curve=None, **data):
        return append_event(run_dir, identity, event, curve, **data)
    previous_manifest, recovery_error = None, None
    recovery_backup = run_dir / ("manifest.recovery-" + str(time.time_ns()) + ".json")
    if recovering:
        try:
            previous_manifest = json.loads((run_dir / "manifest.json").read_text())
            if not isinstance(previous_manifest, dict):
                raise ValueError("previous manifest must be a JSON object")
            write_json(recovery_backup, previous_manifest)
        except (OSError, ValueError) as error:
            recovery_error = "publication recovery cannot read or preserve previous manifest: " + str(error)
    manifest = {"schema_version": 1, "run_id": run_id, "experiment": spec.get("name"),
                "method": method, "schema": schema, "dataset_type": dtype, "dataset": dataset,
                "source_sha256": source.hexdigest(), "attempt_id": attempt_id,
                "result_format": "easytsad-v1", "partial": bool(curves),
                "score_dir": str(score_dir.relative_to(ROOT)), "all_path": str((eval_dir / "all.json").relative_to(ROOT)),
                "avg_path": str((eval_dir / "avg.json").relative_to(ROOT)), "runtime_path": str((runtime_dir / "time.json").relative_to(ROOT)),
                "metric_keys": metric_keys, "result_sha256": {},
                "n_curves_scored": 0, "n_curves_evaluated": 0, "total_curves": None,
                "diagnostics": {"constant": False, "all_zero": False, "core_f1_zero": False},
                "status": "running", "execution_phase": "recovery" if recovering else "execution",
                "started_at": now(), "finished_at": None,
                "duration_s": None, "config_hash": chash, "config": resolved,
                "git": git_info(), "env": env_info(), "command": sys.argv, "error": None}
    if recovering and isinstance(previous_manifest, dict):
        manifest.update(recovery_source_run_id=previous_manifest.get("run_id"),
                        recovery_source_manifest=recovery_backup.name)
    write_json(run_dir / "manifest.json", manifest)
    emit("dataset_started")
    handler = attach_log(run_dir / "run.log")
    t0 = time.time()
    gpu_monitor = None
    original_cwd = os.getcwd()
    try:
        if recovering:
            if recovery_error:
                raise ValueError(recovery_error)
            replay_curves = publication_recovery_curves(
                previous_manifest, run_dir, identity, attempt_id, chash, score_dir)
        if spec.get("require_cuda") and not recovering:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("This experiment requires CUDA; refusing to fall back to CPU.")
        if spec.get("monitor_gpu") and not recovering:
            gpu_monitor = GPUMonitor(run_dir).start()
        set_seed(seed)
        # Preserve EasyTSAD's original scientific output paths.
        PathManager.del_instance()
        gcfg["Paths"]["Results"].update(base_dir=str(ROOT / "Results"), score_dir="Scores",
                                          runtime_dir="RunTime", eval_dir="Evals")
        pm = PathManager(gcfg)
        pm.load_dataset_path(str(Path(spec.get("dataset_dir", ROOT / "datasets")).resolve()))
        # Some methods write relative checkpoints; keep concurrent runs independent.
        if spec.get("isolate_workdir"):
            os.chdir(run_dir)

        dc = {"dataset_type": dtype, "datasets": [dataset], "curves": curves,
              "specify_curves": bool(curves), "train_proportion": ds_cfg["train_proportion"],
              "valid_proportion": ds_cfg["valid_proportion"], "Transfer": gcfg["Transfer"]}
        if schema == "zero_shot_cross_ds":
            dc.update(datasets=src_datasets + [dataset], src_datasets=src_datasets,
                      dst_datasets=[dataset])

        if not recovering:
            __import__(f"EasyTSAD.Methods.{method}.{method}")  # registers the method class
            exp = SCHEMAS[schema](dc, method, None, None, None)
            exp.cfg["Data_Params"] = dict(data_params)
        if gpu_monitor is not None:
            gpu_monitor.set_phase("load_train_test")
        # Keep only evaluation labels; retaining duplicate raw series increases training memory.
        raw = {name: curve.test_label.copy() for name, curve in
               load_data(dict(dc, datasets=[dataset]), preprocess="raw", diff_p=0)[dataset].items()}
        per_curve, native_per_curve, skipped = {}, {}, []
        accumulated = None
        scored_curves = set()
        curve_flags = {}
        manifest["total_curves"] = len(raw)
        expected_curves = set(raw)
        if schema == "zero_shot":
            transfer = dc["Transfer"].get(dataset, dc["Transfer"].get("Default", {}))
            keys = list(raw)
            np.random.RandomState(transfer["random_seed"]).shuffle(keys)
            expected_curves = set(keys[int(np.ceil(transfer["proportion"] * len(keys))):])
        if recovering and set(replay_curves) != expected_curves:
            raise ValueError("publication recovery scored curves do not match expected dataset coverage")
        emit("dataset_loaded", total_curves=len(raw), score_dir=str(score_dir),
             all_path=manifest["all_path"], avg_path=manifest["avg_path"])
        def native_average():
            average = {}
            for metric in copy.deepcopy(accumulated or []):
                metric.avg()
                average.update(to_builtin(metric.to_dict()))
            return average
        def current_summary():
            average = native_average()
            return {k: average[name] if isinstance(average[name], dict) else {"value": average[name]}
                    for k, name in metric_keys.items() if name in average}
        def evaluate_curve(curve, score_path):
            nonlocal accumulated
            if curve in scored_curves:
                raise ValueError("duplicate curve callback: " + curve)
            # Evaluation may not alter random streams used by subsequent training.
            np_state, py_state = np.random.get_state(), random.getstate()
            try:
                score = np.load(score_path, allow_pickle=False)
                if score.ndim != 1 or not score.size or not np.issubdtype(score.dtype, np.number) or not np.isfinite(score).all():
                    raise ValueError("invalid score array for " + curve + ": expected nonempty finite numeric 1-D")
                labels = raw[curve].copy()
                if score.size > labels.size:
                    raise ValueError("score exceeds label length for " + curve)
                flags = {"constant": bool(np.all(score == score[0])), "all_zero": bool(np.all(score == 0))}
                protocols = [EVALS[k]() for k in eval_keys]
                perf = Performance(method, dataset, curve, scores=score, labels=labels, margins=margins)
                if perf.all_label_normal:
                    skipped.append(curve)
                else:
                    res, native = perf.perform_eval(protocols)
                    rec = {}
                    for key, metric in zip(eval_keys, res):
                        val = to_builtin(list(metric.to_dict().values())[0])
                        rec[key] = val if isinstance(val, dict) else {"value": val}
                    json.dumps(rec, allow_nan=False)
                    per_curve[curve] = rec
                    native_per_curve[curve] = to_builtin(native)
                    if accumulated is None:
                        accumulated = copy.deepcopy(res)
                    else:
                        for aggregate, metric in zip(accumulated, res):
                            aggregate.add(metric)
                scored_curves.add(curve)
                core = [per_curve[curve][key]["f1"] for key in ("point_f1", "event_f1")
                        if key in per_curve.get(curve, {}) and "f1" in per_curve[curve][key]]
                flags["core_f1_zero"] = bool(core) and all(value == 0 for value in core)
                curve_flags[curve] = flags
                manifest.update(n_curves_scored=len(scored_curves), n_curves_evaluated=len(per_curve),
                                diagnostics=diagnostic_flags(per_curve, curve_flags))
                write_json(run_dir / "manifest.json", manifest)
                emit("curve_evaluated", curve, metrics=per_curve.get(curve), skipped=curve in skipped,
                     summary=current_summary(), n_curves_scored=len(scored_curves),
                     n_curves_evaluated=len(per_curve), total_curves=manifest["total_curves"],
                     diagnostics=flags, provisional=True)
            finally:
                np.random.set_state(np_state)
                random.setstate(py_state)
        def observer(event, dataset=None, curve=None, **data):
            emit(event, curve, **data)
            if event == "curve_scored":
                evaluate_curve(curve, data["score_path"])
        if recovering:
            for curve in replay_curves:
                evaluate_curve(curve, score_dir / (curve + ".npy"))
        else:
            exp.observer = observer
            exp.do_exp(tsDatas=exp.load_data(), hparams=hparams)
        if gpu_monitor is not None:
            gpu_monitor.set_phase("evaluation")
        # Custom schemas without callbacks remain supported; do not evaluate twice.
        for score_file in sorted(score_dir.glob("*.npy")) if not recovering else []:
            if score_file.stem in raw and score_file.stem not in scored_curves:
                # Only callbacks establish fresh scores; legacy custom schemas must
                # write a score after this attempt started.
                if score_file.stat().st_mtime >= t0:
                    evaluate_curve(score_file.stem, score_file)
        actual_curves = scored_curves
        if schema in ("naive", "mts", "all_in_one", "zero_shot", "zero_shot_cross_ds") and actual_curves != expected_curves:
            raise ValueError("incomplete curve coverage: missing=" + repr(sorted(expected_curves - actual_curves)) + "; unexpected=" + repr(sorted(actual_curves - expected_curves)))
        if actual_curves != set(per_curve) | set(skipped):
            raise ValueError("score files changed after evaluation")
        if not per_curve and not skipped:
            raise ValueError("experiment produced no curve scores")
        manifest["total_curves"] = len(expected_curves)
        manifest.update(execution_phase="publication", scored_curves=sorted(scored_curves),
                        skipped_all_normal=sorted(skipped))
        write_json(run_dir / "manifest.json", manifest)
        # Publish native original metrics only after every expected score has been
        # evaluated. A selected-curves run never overwrites a complete pair.
        with export_lock():
            preserve_complete = bool(curves) and (eval_dir / "avg.json").exists()
            previous = {path: path.read_bytes() if path.exists() else None
                        for path in (eval_dir / "all.json", eval_dir / "avg.json")}
            try:
                if not preserve_complete:
                    write_json(eval_dir / "all.json", dict(sorted(native_per_curve.items())))
                    write_json(eval_dir / "avg.json", native_average())
                    manifest["result_sha256"] = {
                        manifest[key]: hashlib.sha256((ROOT / manifest[key]).read_bytes()).hexdigest()
                        for key in ("all_path", "avg_path")}
                    manifest["result_sha256"].update({str((score_dir / (curve + ".npy")).relative_to(ROOT)):
                        hashlib.sha256((score_dir / (curve + ".npy")).read_bytes()).hexdigest()
                        for curve in sorted(scored_curves)})
                manifest["canonical_metrics_committed"] = not preserve_complete
                manifest["scored_curves"] = sorted(scored_curves)
                manifest["skipped_all_normal"] = sorted(skipped)
                manifest["status"] = "success"
                manifest["finished_at"] = now()
                manifest["duration_s"] = round(time.time() - t0, 2)
                write_json(run_dir / "manifest.json", manifest)
            except BaseException:
                # A failed avg/manifest replacement must not invalidate the last
                # complete native pair. The benchmark remains at its last commit.
                if not preserve_complete:
                    for path, payload in previous.items():
                        if payload is None:
                            path.unlink(missing_ok=True)
                        else:
                            _atomic(path, payload)
                manifest["canonical_metrics_committed"] = False
                manifest["result_sha256"] = {}
                raise
    except Exception as e:  # keep going with the next dataset
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        log.error(manifest["error"])
    except KeyboardInterrupt:  # SIGINT / SIGTERM: record and stop
        manifest["status"] = "aborted"
        manifest["error"] = "Interrupted by signal."
        raise
    finally:
        os.chdir(original_cwd)
        if gpu_monitor is not None:
            try:
                gpu_monitor.stop()
                manifest["gpu_memory_file"] = "gpu_memory.json"
            except Exception as error:
                manifest["gpu_monitor_error"] = f"{type(error).__name__}: {error}"
                log.warning("GPU monitor failed: %s", error)
        manifest["finished_at"] = now()
        manifest["duration_s"] = round(time.time() - t0, 2)
        write_json(run_dir / "manifest.json", manifest)
        log.removeHandler(handler)
        handler.close()
        emit("dataset_finished", status=manifest["status"], error=manifest.get("error"),
             diagnostics=manifest["diagnostics"], n_curves_scored=manifest["n_curves_scored"],
             n_curves_evaluated=manifest["n_curves_evaluated"], total_curves=manifest["total_curves"],
             all_path=manifest["all_path"], avg_path=manifest["avg_path"],
             canonical_metrics_committed=manifest.get("canonical_metrics_committed", False),
             provisional=False,
             summary=current_summary() if "current_summary" in locals() else {})
        fcntl.flock(dataset_lock, fcntl.LOCK_UN)
        dataset_lock.close()
    if manifest["status"] == "success" and not curves:
        try:
            from export import export
            _, benchmark = export()
            emit("benchmark_updated", revision=json.loads(benchmark.read_text()).get("revision"))
        except Exception as error:
            # Publication failure must never turn a scientifically successful run into a failure.
            emit("benchmark_publish_failed", error=repr(error))
            log.exception("Benchmark publication failed; retry export to recover")
    print(f"[{manifest['status']}] {method}/{schema}/{dataset} -> {run_dir.relative_to(ROOT)}")
    return manifest["status"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="experiment spec (TOML)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--run-id", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--datasets", nargs="*", help="subset of the spec's datasets")
    ap.add_argument("--force", action="store_true", help="re-run even if an identical successful run exists")
    a = ap.parse_args()

    spec = toml.load(a.spec)
    if spec["schema"] not in SCHEMAS:
        sys.exit(f"unsupported schema '{spec['schema']}'. Supported: {', '.join(SCHEMAS)}")
    unknown = [k for k in spec.get("evals", []) if k not in EVALS]
    if unknown:
        sys.exit(f"unknown eval keys {unknown}. Known: {', '.join(EVALS)}")
    if spec["schema"] == "zero_shot_cross_ds":
        try:
            _, targets = _cross_dataset_split(spec)
            datasets = a.datasets or targets
            if any(dataset not in targets for dataset in datasets):
                raise ValueError("zero_shot_cross_ds --datasets must contain only declared target datasets")
        except ValueError as error:
            sys.exit(str(error))
    else:
        datasets = a.datasets or spec["datasets"]
    if spec.get("curves") and len(datasets) != 1:
        sys.exit("'curves' requires exactly one dataset")
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # SIGTERM -> KeyboardInterrupt
    statuses = [run_one(spec, a.method, d, a.run_id, a.force) for d in datasets]
    sys.exit(1 if "failed" in statuses else 0)


if __name__ == "__main__":
    main()
