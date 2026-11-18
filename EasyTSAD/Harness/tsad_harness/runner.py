"""Run the method through the existing `scripts/tsad exp` + LiveLeaderBoard pipeline.

Nothing here evaluates anything itself: the harness only writes experiment specs
into `<task>/runs/`, invokes `scripts/tsad exp` (synchronously for SMOKE,
detached for EVALUATE) and reads back the run directories the pipeline writes
to method-local reproduction/runs metadata and native ROOT/Results scientific files.

Failures are typed by `classify_failure` so the agent knows which files a
repair may touch (`ALLOWED_SCOPE`).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import tomlkit

from .config import HarnessError, ROOT, load
from .guard import load_registry
from .task import Task, now

TSAD = ROOT / "scripts" / "tsad"
RESULTS = ROOT / "LiveLeaderBoard" / "results"
RUNS = RESULTS / "runs"
LOGS = RESULTS / "logs"
RUN_ID_RE = re.compile(r"run_id=(\d{8}T\d{6}Z)")

FAILURE_CLASSES = ("ALGORITHM_IMPLEMENTATION", "ENVIRONMENT", "ADAPTER_INTERFACE",
                   "RESOURCE_EXECUTION", "UNKNOWN")
ALLOWED_SCOPE = {
    "ENVIRONMENT": "the method's own env only (envs/<env>/pyproject.toml via `scripts/tsad add`; "
                   "registry entry); never a shared env",
    "ADAPTER_INTERFACE": "EasyTSAD/Methods/<M>/<M>.py (+ __init__.py, config.toml layout): shapes, "
                         "lifecycle, score alignment",
    "ALGORITHM_IMPLEMENTATION": "EasyTSAD/Methods/<M>/, preserving the official source and "
                                "algorithm settings",
    "RESOURCE_EXECUTION": "runtime settings only (device, inference batch size, num_workers); "
                          "never the algorithm",
    "UNKNOWN": "diagnose first (read the log); ask the human if the cause is outside the method",
}


# ------------------------------------------------------------------ registry
def _registry() -> dict:
    """`[envs]` of envs/registry.toml; {} when it does not parse (see guard.load_registry)."""
    return load_registry()[0]


def method_env(method: str) -> str | None:
    """Name of the uv env `method` is registered to in envs/registry.toml (first match)."""
    return next((e for e, s in _registry().items() if method in s.get("methods", [])), None)


# --------------------------------------------------------------------- specs
def spec_name(task: Task, kind: str) -> str:
    return f"harness_{task.method}_{kind}"


def _method_defaults(method: str) -> dict:
    p = load().methods_dir / method / "config.toml"
    try:
        with open(p, "rb") as f:
            return tomllib.load(f).get("Model_Params", {}).get("Default", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def epochs_claim(task: Task) -> dict | None:
    """Contract claim `train.epochs`, if any."""
    try:
        claims = json.loads((task.dir / "contract.json").read_text()).get("claims", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return next((c for c in claims if isinstance(c, dict) and c.get("id") == "train.epochs"), None)


def non_iterative(claim: dict | None) -> bool:
    """A `train.epochs` claim that declares no epoch loop (value null / 0 / "n/a", or iterative=false)."""
    v = claim.get("value") if isinstance(claim, dict) else 1
    return isinstance(claim, dict) and (claim.get("iterative") is False or v is None
                                        or (v == 0 and not isinstance(v, bool))
                                        or str(v).strip().lower() == "n/a")


def epochs_key(task: Task) -> str | None:
    """config.toml key holding the epoch count (contract claim `train.epochs`.config_key)."""
    key = (epochs_claim(task) or {}).get("config_key")
    if not isinstance(key, str) or not key:
        return None
    key = key.split(".")[-1] if key.startswith("Model_Params.") else key
    return key if key in _method_defaults(task.method) else None


def evaluation_identity(task: Task, plan: dict, dataset: str) -> str:
    path = task.method_dir / "config.toml"
    config = tomllib.loads(path.read_text()) if path.exists() else {}
    parameters = config.get("Model_Params", {})
    identity = {"method": task.method, "dataset": dataset,
        "data": config.get("Data_Params", {}),
        "parameters": {**parameters.get("Default", {}), **parameters.get(dataset, {})},
        "evaluation": {key: value for key, value in plan.items()
            if key not in ("name", "methods", "datasets", "gpus", "compute_device", "recover_publication",
                           "min_gpu_memory_mib", "preferred_gpus")}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]


def evaluation_config(task: Task) -> dict:
    """Return independent evaluation config, merging global, task spec, then state."""
    effective = deepcopy(load()["eval"])
    effective.update(deepcopy(getattr(task, "spec", {}).get("evaluation") or {}))
    effective.update(deepcopy(getattr(task, "state", {}).get("evaluation") or {}))
    return effective


def write_spec(task: Task, kind: str, *, epochs: int | None = None,
               hparams: dict | None = None, evaluation: dict | None = None, **fields) -> Path:
    """Write `<task>/runs/<kind>.toml` for `scripts/tsad exp`.

    kind "smoke" uses `[smoke]` (one dataset, a curve subset), "eval" uses
    evaluation_config(task) with optional `evaluation` overrides.
    `epochs` is mapped through the contract's `train.epochs` config_key (skipped
    when unknown); `hparams` are extra hparam overrides; other keyword arguments
    replace top-level spec fields (e.g. `datasets=[...]`).
    """
    cfg = load()
    if kind == "smoke":
        s = cfg["smoke"]
        spec = {"name": spec_name(task, kind), "schema": s["schema"], "dataset_type": s["dataset_type"],
                "datasets": [s["dataset"]], "curves": list(s["curves"])}
    elif kind == "eval":
        s = evaluation_config(task)
        s.update(evaluation or {})
        spec = {"name": spec_name(task, kind), "schema": s["schema"], "dataset_type": s["dataset_type"],
                "datasets": list(s["datasets"])}
    else:
        raise ValueError(f"unknown spec kind {kind!r}")
    spec.update(methods=[task.method], evals=list(s["evals"]),
                seed=int((s if kind == "eval" else cfg["eval"]).get("seed", 2026)),
                isolate_workdir=True)
    spec.update(fields)
    if kind == "eval":
        for key in ("src_datasets", "dst_datasets", "dataset_dir", "curves", "global_cfg", "overrides", "recover_publication"):
            if key in s:
                spec[key] = deepcopy(s[key])
    hp = dict(hparams or {})
    if epochs is not None and (key := epochs_key(task)):
        hp[key] = int(epochs)
    doc = tomlkit.document()
    doc.add(tomlkit.comment(f"generated by tsad_harness.runner.write_spec ({kind}) at {now()}"))
    for k, v in spec.items():
        doc[k] = v
    if hp:
        doc["overrides"] = {task.method: {"hparams": hp}}
    out = task.dir / "runs" / f"{kind}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tomlkit.dumps(doc))
    return out


# --------------------------------------------------------------- run reading
def find_run(method: str, schema: str, dataset: str, *, run_id: str | None = None,
             experiment: str | None = None, attempt_id: str | None = None) -> Path | None:
    """Find method-local metadata, retaining read compatibility with historical runs."""
    metadata = load().methods_dir / method / "reproduction" / "runs"
    candidates = list(metadata.glob(f"{attempt_id or '*'}/{schema}/{dataset}/manifest.json"))
    legacy = RUNS / method / schema / dataset
    candidates += list(legacy.glob(f"{run_id or '*'}/manifest.json"))
    best, best_at = None, ""
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if run_id and manifest.get("run_id", path.parent.name) != run_id:
            continue
        if attempt_id and manifest.get("attempt_id", path.parents[2].name if path.is_relative_to(metadata) else None) != attempt_id:
            continue
        if any(manifest.get(key, value) != value for key, value in
               (("method", method), ("schema", schema), ("dataset", dataset))):
            continue
        if experiment and manifest.get("experiment") != experiment:
            continue
        at = manifest.get("finished_at") or manifest.get("started_at") or ""
        if at >= best_at:
            best, best_at = path.parent, at
    return best


def check_scores(run_dir: Path | None, expected: int | None = None) -> dict:
    """Load scores/*.npy: each must be a non-empty, 1-D, all-finite array."""
    problems, lengths = [], {}
    score_dir = run_dir / "scores" if run_dir else None
    selected = None
    if run_dir and (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("score_dir"):
            score_dir = ROOT / manifest["score_dir"]
            if isinstance(manifest.get("scored_curves"), list):
                selected = manifest["scored_curves"]
            elif isinstance(manifest.get("config", {}).get("curves"), list):
                selected = manifest["config"]["curves"]
    if score_dir and selected is not None:
        files = []
        for curve in sorted(set(selected)):
            path = score_dir / (curve + ".npy")
            if path.exists():
                files.append(path)
            else:
                problems.append(f"{curve}: score file missing")
    else:
        files = sorted(score_dir.glob("*.npy")) if score_dir else []
    for f in files:
        try:
            a = np.load(f, allow_pickle=False)
        except Exception as e:  # corrupt or object array
            problems.append(f"{f.stem}: cannot load ({type(e).__name__}: {e})")
            continue
        lengths[f.stem] = int(a.size)
        if a.ndim != 1:
            problems.append(f"{f.stem}: score array has shape {a.shape}, expected 1-D")
        elif a.size == 0:
            problems.append(f"{f.stem}: empty score array")
        elif not np.issubdtype(a.dtype, np.number):
            problems.append(f"{f.stem}: non-numeric dtype {a.dtype}")
        elif not np.isfinite(a).all():
            problems.append(f"{f.stem}: {int((~np.isfinite(a)).sum())} non-finite values (NaN/inf)")
    if not files:
        problems.append("no score files written")
    elif expected is not None and len(files) != expected:
        problems.append(f"{len(files)} score files, expected {expected}")
    return {"ok": not problems, "n_files": len(files), "lengths": lengths, "problems": problems}


def read_summary(run_dir: Path | None) -> dict:
    """Read this attempt's trusted progress; canonical averages are the fallback.

    A partial attempt can point at a previous complete native pair, so its live
    event summary takes precedence. No metric copy is written by the harness.
    """
    if run_dir is None:
        return {}
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        manifest = {}
    summary = None
    try:
        with (run_dir / "progress.jsonl").open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # an append can be observed before its final newline
                data = event.get("data", {})
                if event.get("type") in ("curve_evaluated", "dataset_finished") and isinstance(data.get("summary"), dict):
                    summary = data["summary"]
    except OSError:
        pass
    if summary is not None:
        return summary
    if manifest.get("result_format") == "easytsad-v1":
        if not manifest.get("canonical_metrics_committed") or not manifest.get("avg_path"):
            return {}
        path = ROOT / manifest["avg_path"]
        try:
            payload = path.read_bytes()
            expected = manifest.get("result_sha256", {})
            checksum = expected.get(manifest["avg_path"], expected.get("avg_path"))
            if checksum and hashlib.sha256(payload).hexdigest() != checksum:
                return {}
            native = json.loads(payload)
        except (OSError, ValueError):
            return {}
        return {key: native[name] if isinstance(native[name], dict) else {"value": native[name]}
                for key, name in manifest.get("metric_keys", {}).items() if name in native}
    try:
        return json.loads((run_dir / "metrics.json").read_text()).get("summary", {})
    except (OSError, ValueError):
        return {}


def _tail(p: Path | None, n: int = 4000) -> str:
    try:
        return p.read_text(errors="replace")[-n:] if p else ""
    except OSError:
        return ""


def _killpg(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=30)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def source_hash(method_dir: Path) -> str:
    """sha1 over the method's source inputs (*.py and config.toml, paths + contents)."""
    h = hashlib.sha1()
    files = []
    for directory, children, names in os.walk(method_dir):
        children[:] = [name for name in children if name not in
                       {"__pycache__", ".venv", "venv", ".git", "reproduction"}]
        files.extend(Path(directory) / name for name in names if name.endswith(".py") or name == "config.toml")
    for f in sorted(files):
        h.update(str(f.relative_to(method_dir)).encode() + b"\0")
        try:
            h.update(f.read_bytes())
        except OSError:
            pass
        h.update(b"\0")
    return h.hexdigest()


def _gpu(section: str) -> str | None:
    cfg = load()
    g = str(cfg[section].get("gpus", cfg["eval"].get("gpus", "")) or "").strip()
    return g or None


def discover_resources(gpus: str | None = None, *, min_gpu_memory_mib: int | float = 0,
                       preferred_gpus: str | None = None) -> dict:
    """Inspect resources without torch; select idle GPUs or the scheduler's CPU slot.

    None, an empty string and "auto" respect CUDA_VISIBLE_DEVICES. Explicit
    indices/UUIDs override that mask; "cpu" and "-1" force CPU. An empty returned
    `gpus` string means omit --gpus (scripts/tsad.py uses one None slot).
    Busy or unmeasurable visible GPUs raise a retryable HarnessError. This is a
    snapshot, not a reservation; callers must rediscover immediately before launch.
    Automatic selection ranks eligible idle GPUs by preference, then free memory.
    Preferences are indices/UUIDs and never expand visibility or explicit selections.
    """
    if (isinstance(min_gpu_memory_mib, bool) or not isinstance(min_gpu_memory_mib, (int, float))
            or not 0 <= min_gpu_memory_mib < float("inf")):
        raise HarnessError("Evaluation min_gpu_memory_mib must be a finite non-negative number")
    if preferred_gpus is not None and not isinstance(preferred_gpus, str):
        raise HarnessError("Evaluation preferred_gpus must be a comma-separated string of GPU indices or UUIDs")
    preferred = [token.strip() for token in (preferred_gpus or "").split(",") if token.strip()]
    if any(not (token.isdigit() or (token.startswith("GPU-") and len(token) > 4)) for token in preferred):
        raise HarnessError("Evaluation preferred_gpus must contain GPU indices or UUIDs")
    requested = (gpus or "auto").strip() or "auto"
    automatic = requested.lower() == "auto"
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    selection = mask if automatic else requested
    resources = {"gpus": "", "cpu_count": os.cpu_count() or 1, "devices": [],
                 "requested_gpus": requested, "cuda_visible_devices": mask,
                 "source": "nvidia-smi", "discovered_at": now(),
                 "min_gpu_memory_mib": min_gpu_memory_mib, "preferred_gpus": preferred_gpus}
    if selection is not None and selection.strip().lower() in ("", "cpu", "-1"):
        resources["reason"] = "CPU explicitly selected or CUDA devices hidden"
        return resources
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        resources["reason"] = f"GPU discovery unavailable: {type(error).__name__}: {error}"
        return resources

    def number(value: str) -> float | None:
        try:
            parsed = float(value)
            return parsed if 0 <= parsed < float("inf") else None
        except ValueError:
            return None

    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or not fields[0].isdigit() or not fields[1]:
            continue
        total, used, free, utilization = [number(value) for value in fields[2:]]
        idle = (total is not None and total > 0 and free is not None and free > 0
                and used is not None and used <= 256 and utilization is not None and utilization <= 5)
        devices.append({"index": int(fields[0]), "uuid": fields[1],
                        "total_memory_mib": total, "used_memory_mib": used,
                        "free_memory_mib": free, "utilization_gpu_percent": utilization,
                        "idle": idle})
    if not devices:
        resources["reason"] = "No GPUs reported by nvidia-smi"
        return resources
    if selection is not None:
        visible = []
        for token in selection.split(","):
            token = token.strip()
            matches = [device for device in devices
                       if token == str(device["index"])
                       or (token.startswith("GPU-") and device["uuid"].startswith(token))]
            if len(matches) != 1 or matches[0] in visible:
                if automatic:
                    break
                raise HarnessError(f"Invalid or duplicate GPU selector {token!r}; use GPU indices, UUIDs or cpu")
            visible.append(matches[0])
        devices = visible
    resources["devices"] = devices
    resources["idle_policy"] = {"max_used_memory_mib": 256, "max_utilization_gpu_percent": 5}
    if not devices:
        resources["reason"] = "No GPUs visible through CUDA_VISIBLE_DEVICES"
        return resources
    try:
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        active = {line.strip() for line in processes.stdout.splitlines() if line.strip()}
        for device in devices:
            device["compute_active"] = device["uuid"] in active
            device["idle"] = device["idle"] and not device["compute_active"]
    except (OSError, subprocess.SubprocessError) as error:
        raise HarnessError("RESOURCE_EXECUTION (retryable): cannot verify GPU compute processes; "
                           f"retry discover_resources or launch_eval later: {error}") from error
    selected = [device for device in devices if device["idle"]]
    if not selected or (not automatic and len(selected) != len(devices)):
        raise HarnessError("RESOURCE_EXECUTION (retryable): requested/visible GPUs are busy or their "
                           "idle state cannot be verified; retry discover_resources or launch_eval later")
    eligible = [device for device in selected if device["free_memory_mib"] >= min_gpu_memory_mib]
    if not eligible or (not automatic and len(eligible) != len(selected)):
        raise HarnessError(f"RESOURCE_EXECUTION (retryable): requested/visible idle GPUs do not meet "
                           f"min_gpu_memory_mib={min_gpu_memory_mib}; retry on a GPU with more free memory")
    if automatic:
        def preference(device: dict) -> tuple:
            rank = next((index for index, token in enumerate(preferred)
                         if token == str(device["index"])
                         or (token.startswith("GPU-") and device["uuid"].startswith(token))), len(preferred))
            return rank, -device["free_memory_mib"]

        eligible.sort(key=preference)
    resources["gpus"] = ",".join(str(device["index"]) for device in eligible)
    return resources


# --------------------------------------------------------------------- smoke
def run_smoke(task: Task, *, epochs: int | None = None) -> dict:
    """Synchronous smoke run; returns status, run dir, metrics, typed failure and score check.

    The epoch count is capped by `[smoke].max_epochs` and by the method's own default.
    The result is also stored as `runs/smoke-<n>.json` and `runs/smoke_latest.json`.
    """
    cfg = load()
    s = cfg["smoke"]
    n = task.bump("smoke_runs")
    cap = int(s.get("max_epochs", 2))
    ep = min(int(epochs), cap) if epochs else cap
    key = epochs_key(task)
    if key and isinstance(_method_defaults(task.method).get(key), int):
        ep = min(ep, _method_defaults(task.method)[key])
    warning = None
    if not key and not non_iterative(epochs_claim(task)):
        warning = ("no epoch cap applied: contract claim train.epochs has no config_key present in "
                   "config.toml [Model_Params.Default], so the smoke run trains with the method's full "
                   "epoch count (add the config_key, or mark the claim \"iterative\": false)")
    spec = write_spec(task, "smoke", epochs=ep if key else None)
    src_hash = source_hash(task.method_dir)
    cmd = [str(TSAD), "exp", str(spec), "--methods", task.method, "--force"]
    if g := _gpu("smoke"):
        cmd += ["--gpus", g.split(",")[0].strip()]
    out_path = task.dir / "runs" / f"smoke-{n:03d}.out"
    task.log("runner/smoke_start", n=n, cmd=cmd, epochs=ep if key else None, epochs_key=key,
             warning=warning)
    timeout = 60 * float(s.get("timeout_minutes", 30))
    status, rc = None, None
    with open(out_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=cfg.network_env(), stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(proc)
            status = "timeout"
    exp_out = _tail(out_path, 100_000)
    m = RUN_ID_RE.search(exp_out)
    run_id = m.group(1) if m else None
    run_dir = find_run(task.method, s["schema"], s["dataset"], run_id=run_id,
                       experiment=spec_name(task, "smoke")) if run_id else None
    man = {}
    if run_dir:
        try:
            man = json.loads((run_dir / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
    job_log = LOGS / f"{run_id}_{spec_name(task, 'smoke')}" / f"{task.method}_{s['dataset']}.out" \
        if run_id else None
    log_tail = _tail(job_log) or _tail(run_dir / "run.log" if run_dir else None) or exp_out[-4000:]
    status = status or man.get("status") or "error"
    error = (man.get("error") or "") if status != "timeout" else f"smoke run exceeded {timeout / 60:.0f} min"
    if status != "timeout" and not man:   # died before the pipeline wrote this run's manifest
        error = (f"scripts/tsad exp exited with rc={rc} and wrote no manifest for run_id={run_id}:\n"
                 + (_tail(job_log, 2000) + "\n" + exp_out[-2000:]).strip())
    if status not in ("success", "timeout") and not error:
        error = exp_out[-3000:]
    if status == "timeout" and warning:
        error += f" ({warning})"
    score = check_scores(run_dir, expected=len(s["curves"])) if run_dir else \
        {"ok": False, "n_files": 0, "lengths": {}, "problems": ["no run directory"]}
    if status == "success" and not score["ok"]:
        status, error = "failed", "score check failed: " + "; ".join(score["problems"])
    metrics = read_summary(run_dir)
    fclass = None
    if status != "success":
        fclass = "RESOURCE_EXECUTION" if status == "timeout" else \
            "ADAPTER_INTERFACE" if error.startswith("score check failed") else \
            classify_failure(error + "\n" + log_tail, task.method)
    res = {"n": n, "status": status, "run_id": run_id,
           "run_dir": (str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir))
           if run_dir else None,
           "epochs": ep if key else None, "epochs_key": key, "metrics_summary": metrics,
           "error": error[-4000:], "failure_class": fclass, "rc": rc, "source_hash": src_hash,
           **({"warning": warning} if warning else {}),
           "allowed_scope": ALLOWED_SCOPE.get(fclass) if fclass else None,
           "log_tail": log_tail[-4000:], "score_check": score, "finished_at": now(),
           "exp_output": str(out_path.relative_to(task.dir))}
    for name in (f"smoke-{n:03d}.json", "smoke_latest.json"):
        (task.dir / "runs" / name).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    task.log("runner/smoke", **{k: res[k] for k in ("n", "status", "run_dir", "failure_class",
                                                     "metrics_summary", "score_check")},
             error=error[-1500:])
    return res


def latest_smoke(task: Task) -> dict | None:
    p = task.dir / "runs" / "smoke_latest.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------- eval
def evaluation_resources(evaluation: dict, gpus: str | None = None) -> dict:
    device = evaluation.get("compute_device", "auto")
    if device not in ("cpu", "cuda", "auto"):
        raise HarnessError("Evaluation compute_device must be cpu, cuda or auto")
    execution = {key: evaluation[key] for key in ("min_gpu_memory_mib", "preferred_gpus")
                 if key in evaluation and device != "cpu"}
    resources = discover_resources("cpu" if device == "cpu" else
                                   gpus if gpus is not None else evaluation.get("gpus", "auto"), **execution)
    resources["compute_device"] = device
    if device == "cuda" and not resources["gpus"]:
        raise HarnessError("Method requires CUDA but no CUDA device is available for evaluation")
    if device != "cpu" and execution.get("min_gpu_memory_mib", 0) > 0 and not resources["gpus"]:
        raise HarnessError("RESOURCE_EXECUTION (retryable): no GPU available to satisfy min_gpu_memory_mib="
                           f"{execution['min_gpu_memory_mib']}")
    return resources


def launch_eval(task: Task, gpus: str | None = None, *, datasets: list[str] | None = None,
                attempt_id: str | None = None, evaluation: dict | None = None) -> dict:
    """Start the full evaluation detached (`nohup`, new session); returns pid + log path."""
    cfg = load()
    effective = evaluation_config(task)
    effective.update(deepcopy(evaluation or {}))
    if datasets is not None:
        effective["datasets"] = list(datasets)
    resources = evaluation_resources(effective, gpus)
    gpus = resources["gpus"]
    spec = write_spec(task, "eval", evaluation=effective)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = task.dir / "runs" / f"eval-{ts}.out"
    cmd = ["nohup", str(TSAD), "exp", str(spec), "--methods", task.method, "--force",
           *(["--gpus", gpus] if gpus else [])]
    env = cfg.network_env()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if attempt_id:
        env["TSAD_ATTEMPT_ID"] = attempt_id
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    info = {"pid": proc.pid, "log": str(log_path.relative_to(task.dir)), "spec": str(spec.relative_to(task.dir)),
            "argv": cmd, "cwd": ".",
            "gpus": gpus, "resources": resources, "started_at": now(),
            "datasets": list(effective["datasets"]), "evaluation": effective, "attempt_id": attempt_id,
            "commandVerified": True}
    from .scheduler import process_identity
    info["processIdentity"] = process_identity(proc.pid)
    task.set(eval=info)
    task.log("runner/eval_launch", cmd=cmd, **info)
    return info


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:  # reap our own zombie child, if any
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and stat.read_text().split(") ")[-1].startswith("Z"))


def eval_alive(task: Task) -> bool:
    """True while the detached evaluation process recorded in state['eval'] is running."""
    info = task.state.get("eval") or {}
    if info.get("processIdentity"):
        from .scheduler import process_identity
        if process_identity(info.get("pid", 0)) != info["processIdentity"]:
            return False
    return _alive(info.get("pid"))


def eval_status(task: Task) -> dict:
    """Progress of the launched evaluation: per-dataset run status and overall state."""
    cfg = load()
    info = task.state.get("eval")
    if not info:
        return {"launched": False, "running": False, "done": False, "datasets": {}, "failed": []}
    effective = dict(cfg["eval"])
    effective.update(info.get("evaluation") or {})
    log_text = _tail(task.dir / info["log"], 200_000)
    m = RUN_ID_RE.search(log_text)
    run_id = m.group(1) if m else None
    running = eval_alive(task)
    datasets = {}
    for d in (info.get("evaluation") or {}).get("datasets", info.get("datasets", effective["datasets"])):
        rd = find_run(task.method, effective["schema"], d, run_id=run_id,
                      attempt_id=info.get("attempt_id")) if run_id or info.get("attempt_id") else None
        man = rd / "manifest.json" if rd else None
        if man and man.exists():
            mj = json.loads(man.read_text())
            datasets[d] = {"status": mj.get("status"), "run_dir": str(rd.relative_to(ROOT)),
                           "manifest": str(man.relative_to(ROOT)),
                           "error": (mj.get("error") or "")[-4000:],
                           "diagnostics": mj.get("diagnostics", {}), "metrics_summary": read_summary(rd),
                           **{key: mj[key] for key in ("config_hash", "score_dir", "all_path", "avg_path", "runtime_path",
                               "metric_keys", "n_curves_scored", "n_curves_evaluated", "total_curves", "partial",
                               "canonical_metrics_committed", "execution_phase") if key in mj}}
        else:
            datasets[d] = {"status": "pending" if running else "missing"}
    failed = [d for d, v in datasets.items() if v["status"] != "success"]
    return {"launched": True, "running": running, "run_id": run_id, "pid": info.get("pid"),
            "log": info["log"], "datasets": datasets,
            "schema": effective["schema"], "dataset_type": effective.get("dataset_type"),
            "done": not running and bool(datasets) and not failed,
            "failed": [] if running else failed, "log_tail": log_text[-1500:]}


# ------------------------------------------------------------ classification
def publication_failure(result: dict) -> bool:
    error = result.get("error") or ""
    return result.get("status") != "success" and (result.get("execution_phase") == "publication"
        or ("export_lock" in error and "pipeline/export.py" in error))


_RESOURCE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|\bMemoryError\b|CUDA error: out of memory|"
    r"No space left on device|\bKilled\b|signal 9|TimeoutExpired|timed out|exceeded \d+ min|"
    r"CUBLAS_STATUS_ALLOC_FAILED|cuDNN error: CUDNN_STATUS_(NOT_SUPPORTED|ALLOC_FAILED)|"
    r"Too many open files|Interrupted by signal|DataLoader worker .* killed", re.I)
_ENV = re.compile(
    r"No matching distribution|Failed to (build|download|resolve)|uv failed|"
    r"not registered in envs/registry\.toml|undefined symbol|GLIBC|libcu\w+\.so|"
    r"CUDA driver version is insufficient|no kernel image is available|"
    r"Torch not compiled with CUDA|module '(numpy|torch|scipy|pandas|sklearn)' has no attribute|"
    r"requires a different Python", re.I)
_MISSING_MOD = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([\w.]+)'")
_CANNOT_IMPORT = re.compile(r"ImportError: cannot import name '\w+' from '([\w.]+)'")
_FRAME = re.compile(r'File "([^"]+)", line \d+')
_ADAPTER_MSG = re.compile(
    r"NotImplementedError|anomaly_score|test_phase|train_valid_phase|TSData|"
    r"operands could not be broadcast|length of (scores|labels)|"
    r"score.*(shape|length)|(shape|length).*score|unexpected keyword argument|"
    r"missing \d+ required positional argument", re.I)
_ALGO_MSG = re.compile(
    r"shapes cannot be multiplied|size mismatch|Expected .* dimension|"
    r"must match the size|out of bounds|index \d+ is out of|RuntimeError: Given groups|"
    r"element 0 of tensors does not require grad|ZeroDivisionError|LinAlgError", re.I)


def classify_failure(text: str, method: str) -> str:
    """Heuristic failure type from a traceback / run log (see FAILURE_CLASSES)."""
    if not text.strip():
        return "UNKNOWN"
    if _RESOURCE.search(text):
        return "RESOURCE_EXECUTION"
    for rx in (_MISSING_MOD, _CANNOT_IMPORT):
        for mod in rx.findall(text):
            top = mod.split(".")[0]
            if top == "EasyTSAD" or (method and (mod == method or mod.startswith(method + "."))):
                return "ADAPTER_INTERFACE"   # wrong layout / relative import of the method itself
            return "ENVIRONMENT"
    if _ENV.search(text):
        return "ENVIRONMENT"
    frames = _FRAME.findall(text)
    mdir = f"/Methods/{method}/"
    own = [f for f in frames if mdir in f]
    if own:
        relative = own[-1].split(mdir, 1)[1]
        return "ADAPTER_INTERFACE" if relative in (f"{method}.py", "__init__.py") else "ALGORITHM_IMPLEMENTATION"
    if frames and re.search(r"/EasyTSAD/(TrainingSchema|Evaluations|DataFactory|Controller)/", frames[-1]):
        return "ADAPTER_INTERFACE"
    if _ADAPTER_MSG.search(text):
        return "ADAPTER_INTERFACE"
    if _ALGO_MSG.search(text):
        return "ALGORITHM_IMPLEMENTATION"
    return "UNKNOWN"


# ------------------------------------------------------------------- checks
def _run(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Run a command in its own session with a timeout; returns (returncode, output)."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=load().network_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        _killpg(proc)
        return -9, (proc.communicate()[0] or "") + f"\n[timeout after {timeout:.0f}s]"


def import_check(method: str, timeout: float = 900) -> tuple[bool, str]:
    """Import EasyTSAD.Methods.<M>.<M> inside the method's env (via `scripts/tsad run`)."""
    code = ("import EasyTSAD.Controller, importlib; "
            f"importlib.import_module('EasyTSAD.Methods.{method}.{method}')")
    rc, out = _run([str(TSAD), "run", method, "--", "python", "-c", code], timeout)
    return rc == 0, out[-4000:]


def check_method(task: Task, timeout: float = 900) -> dict:
    """Import only this method inside its registered environment."""
    env = method_env(task.method)
    if env is None:
        return {"ok": False, "env": None, "lines": [],
                "error": f"{task.method} is not registered in envs/registry.toml"}
    ok, out = import_check(task.method, timeout)
    lines = [f"ok     {task.method}"] if ok else [f"FAIL   {task.method}"]
    res = {"ok": ok, "env": env, "lines": lines}
    if not ok:
        res["error"] = out[-3000:]
    return res


__all__ = ["method_env", "write_spec", "discover_resources", "run_smoke", "latest_smoke", "launch_eval", "eval_status",
           "classify_failure", "check_scores", "read_summary", "check_method", "import_check", "find_run", "epochs_key", "epochs_claim", "non_iterative",
           "source_hash", "eval_alive", "ALLOWED_SCOPE",
           "FAILURE_CLASSES"]
