"""LiveLeaderBoard experiment runner.

Runs ONE method on the datasets of an experiment spec and stores every
(method, schema, dataset) combination as a self-contained run directory:

    LiveLeaderBoard/results/runs/<method>/<schema>/<dataset>/<run_id>/
        manifest.json        provenance, resolved config, status, timings
        metrics.json         summary + per-curve evaluation results
        runtime.json         train/test wall-clock time (from EasyTSAD)
        model_statistic.txt  parameter statistics written by the method
        run.log              EasyTSAD log of this run
        scores/<curve>.npy   raw anomaly scores

Executed inside the method's uv environment (see `scripts/tsad exp`), so it
must stay compatible with the oldest supported Python (3.9).
"""
from __future__ import annotations

import argparse
import copy
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
from EasyTSAD.TrainingSchema import MTS, AllInOne, Naive, ZeroShot
from EasyTSAD.utils import update_nested_dict
from gpu_monitor import GPUMonitor

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "LiveLeaderBoard"
RESULTS_DIR = LB_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
INDEX = RESULTS_DIR / "index.jsonl"
METHODS_DIR = ROOT / "EasyTSAD" / "Methods"
GLOBAL_CFG = ROOT / "EasyTSAD" / "Controller" / "GlobalCfg.toml"

SCHEMAS = {"naive": Naive, "all_in_one": AllInOne, "zero_shot": ZeroShot, "mts": MTS}

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


def env_info() -> dict:
    env = os.environ.get("TSAD_ENV", "")
    project = os.environ.get("TSAD_ENV_PROJECT")
    if env and not project:
        registry = toml.load(ROOT / "envs" / "registry.toml")
        project = registry.get("envs", {}).get(env, {}).get("path", f"envs/{env}")
    lock = ROOT / project / "uv.lock" if project else None
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


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
    base = RUNS_DIR / method / schema / dataset
    if not base.exists():
        return None
    for d in sorted(base.iterdir(), reverse=True):
        m = d / "manifest.json"
        if m.exists():
            man = json.loads(m.read_text())
            if man.get("status") == "success" and man.get("config_hash") == chash:
                return d
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
        vals = [c[k] for c in per_curve.values() if k in c]
        if not vals:
            continue
        fields = [f for f in vals[0] if f != "threshold"]
        out[k] = {f: float(np.mean([v[f] for v in vals])) for f in fields}
    return out


# ----------------------------------------------------------------- one run
def run_one(spec: dict, method: str, dataset: str, run_id: str, force: bool) -> str:
    schema = spec["schema"]
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

    resolved = {"method": method, "schema": schema, "dataset_type": dtype, "dataset": dataset,
                "curves": curves, "data_params": data_params, "hparams": hparams,
                "evals": eval_keys, "margins": list(margins), "seed": seed,
                "train_proportion": ds_cfg["train_proportion"],
                "valid_proportion": ds_cfg["valid_proportion"],
                "transfer": gcfg["Transfer"] if schema == "zero_shot" else None}
    chash = config_hash(resolved)
    if not force:
        prev = existing_success(method, schema, dataset, chash)
        if prev is not None:
            print(f"[skip] {method}/{schema}/{dataset}: identical successful run {prev.name}")
            return "skipped"

    run_dir = RUNS_DIR / method / schema / dataset / run_id
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "run_id": run_id, "experiment": spec.get("name"),
                "method": method, "schema": schema, "dataset_type": dtype, "dataset": dataset,
                "status": "running", "started_at": now(), "finished_at": None,
                "duration_s": None, "config_hash": chash, "config": resolved,
                "git": git_info(), "env": env_info(), "command": sys.argv, "error": None}
    write_json(run_dir / "manifest.json", manifest)
    handler = attach_log(run_dir / "run.log")
    t0 = time.time()
    gpu_monitor = None
    original_cwd = os.getcwd()
    try:
        if spec.get("require_cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("This experiment requires CUDA; refusing to fall back to CPU.")
        if spec.get("monitor_gpu"):
            gpu_monitor = GPUMonitor(run_dir).start()
        set_seed(seed)
        # Path manager: datasets from ./datasets, all outputs into run_dir.
        PathManager.del_instance()
        gcfg["Paths"]["Results"]["base_dir"] = str(run_dir)
        pm = PathManager(gcfg)
        pm.load_dataset_path(str(Path(spec.get("dataset_dir", ROOT / "datasets")).resolve()))
        pm.get_score_path = lambda m, s, d, c, build=True: str(run_dir / "scores" / f"{c}.npy")
        pm.get_rt_time_path = lambda m, s, d: str(run_dir / "runtime.json")
        pm.get_rt_statistic_path = lambda m, s, d: str(run_dir / "model_statistic.txt")
        # Some methods write relative checkpoints; keep concurrent runs independent.
        if spec.get("isolate_workdir"):
            os.chdir(run_dir)

        dc = {"dataset_type": dtype, "datasets": [dataset], "curves": curves,
              "specify_curves": bool(curves), "train_proportion": ds_cfg["train_proportion"],
              "valid_proportion": ds_cfg["valid_proportion"], "Transfer": gcfg["Transfer"]}

        __import__(f"EasyTSAD.Methods.{method}.{method}")  # registers the method class
        exp = SCHEMAS[schema](dc, method, None, None, None)
        exp.cfg["Data_Params"] = dict(data_params)
        if gpu_monitor is not None:
            gpu_monitor.set_phase("load_train_test")
        exp.do_exp(tsDatas=exp.load_data(), hparams=hparams)

        # ---- evaluation on the saved scores, against raw labels
        if gpu_monitor is not None:
            gpu_monitor.set_phase("evaluation")
        raw = load_data(dc, preprocess="raw", diff_p=0)[dataset]
        protocols = [EVALS[k]() for k in eval_keys]
        per_curve, skipped = {}, []
        for score_file in sorted((run_dir / "scores").glob("*.npy")):
            curve = score_file.stem
            perf = Performance(method, dataset, curve, scores=np.load(score_file),
                               labels=raw[curve].test_label.copy(), margins=margins)
            if perf.all_label_normal:
                skipped.append(curve)
                continue
            res, _ = perf.perform_eval(protocols)
            rec = {}
            for key, metric in zip(eval_keys, res):
                val = to_builtin(list(metric.to_dict().values())[0])
                rec[key] = val if isinstance(val, dict) else {"value": val}
            per_curve[curve] = rec
        metrics = {"summary": mean_metrics(per_curve, eval_keys),
                   "n_curves_scored": len(per_curve) + len(skipped),
                   "n_curves_evaluated": len(per_curve), "skipped_all_normal": skipped,
                   "protocols": {k: p.name for k, p in zip(eval_keys, protocols)},
                   "per_curve": per_curve}
        write_json(run_dir / "metrics.json", metrics)
        manifest["status"] = "success"
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
        line = {k: manifest[k] for k in ("run_id", "experiment", "method", "schema", "dataset",
                                         "status", "finished_at", "duration_s", "config_hash")}
        line["path"] = str(run_dir.relative_to(RESULTS_DIR))
        if manifest["status"] == "success":
            line["summary"] = {k: v.get("f1", v.get("value")) for k, v in metrics["summary"].items()}
        with open(INDEX, "a") as f:
            f.write(json.dumps(line) + "\n")
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
    datasets = a.datasets or spec["datasets"]
    if spec.get("curves") and len(datasets) != 1:
        sys.exit("'curves' requires exactly one dataset")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # SIGTERM -> KeyboardInterrupt
    statuses = [run_one(spec, a.method, d, a.run_id, a.force) for d in datasets]
    sys.exit(1 if "failed" in statuses else 0)


if __name__ == "__main__":
    main()
