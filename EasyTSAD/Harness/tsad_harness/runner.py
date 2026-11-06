"""Run the method through the existing `scripts/tsad exp` + LiveLeaderBoard pipeline.

Nothing here evaluates anything itself: the harness only writes experiment specs
into `<task>/runs/`, invokes `scripts/tsad exp` (synchronously for SMOKE,
detached for EVALUATE) and reads back the run directories the pipeline writes
to `LiveLeaderBoard/results/runs/<M>/<schema>/<dataset>/<run_id>/`.

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
from pathlib import Path

import numpy as np
import tomlkit

from .config import ROOT, load
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
    "ALGORITHM_IMPLEMENTATION": "EasyTSAD/Methods/<M>/src/, faithful to the contract; record any "
                                "changed contract value as a deviation note",
    "RESOURCE_EXECUTION": "runtime settings only (device, inference batch size, num_workers); "
                          "never the algorithm; record it",
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


def write_spec(task: Task, kind: str, *, epochs: int | None = None,
               hparams: dict | None = None, **fields) -> Path:
    """Write `<task>/runs/<kind>.toml` for `scripts/tsad exp`.

    kind "smoke" uses `[smoke]` (one dataset, a curve subset), "eval" uses `[eval]`.
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
        s = cfg["eval"]
        spec = {"name": spec_name(task, kind), "schema": s["schema"], "dataset_type": s["dataset_type"],
                "datasets": list(s["datasets"])}
    else:
        raise ValueError(f"unknown spec kind {kind!r}")
    spec.update(methods=[task.method], evals=list(s["evals"]), seed=int(cfg["eval"].get("seed", 2026)))
    spec.update(fields)
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
             experiment: str | None = None) -> Path | None:
    """Run dir of `run_id` (None if it wrote no manifest), else the newest finished one of `experiment`."""
    base = RUNS / method / schema / dataset
    if run_id:
        return base / run_id if (base / run_id / "manifest.json").exists() else None
    best, best_at = None, ""
    for m in base.glob("*/manifest.json"):
        try:
            man = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if experiment and man.get("experiment") != experiment:
            continue
        at = man.get("finished_at") or ""
        if at >= best_at:
            best, best_at = m.parent, at
    return best


def check_scores(run_dir: Path | None, expected: int | None = None) -> dict:
    """Load scores/*.npy: each must be a non-empty, 1-D, all-finite array."""
    problems, lengths = [], {}
    files = sorted((run_dir / "scores").glob("*.npy")) if run_dir else []
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
    files = [f for f in method_dir.rglob("*") if f.is_file() and "__pycache__" not in f.parts
             and (f.suffix == ".py" or f.name == "config.toml")] if method_dir.is_dir() else []
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
    metrics = {}
    if run_dir and (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text()).get("summary", {})
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
def launch_eval(task: Task, gpus: str | None = None) -> dict:
    """Start the full evaluation detached (`nohup`, new session); returns pid + log path."""
    cfg = load()
    gpus = gpus or _gpu("eval")
    spec = write_spec(task, "eval")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = task.dir / "runs" / f"eval-{ts}.out"
    cmd = ["nohup", str(TSAD), "exp", str(spec), "--methods", task.method, "--force",
           *(["--gpus", gpus] if gpus else [])]
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=cfg.network_env(), stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    info = {"pid": proc.pid, "log": str(log_path.relative_to(task.dir)), "spec": str(spec.relative_to(task.dir)),
            "gpus": gpus, "started_at": now(), "datasets": list(cfg["eval"]["datasets"])}
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
    return _alive((task.state.get("eval") or {}).get("pid"))


def eval_status(task: Task) -> dict:
    """Progress of the launched evaluation: per-dataset run status and overall state."""
    cfg = load()
    info = task.state.get("eval")
    if not info:
        return {"launched": False, "running": False, "done": False, "datasets": {}, "failed": []}
    log_text = _tail(task.dir / info["log"], 200_000)
    m = RUN_ID_RE.search(log_text)
    run_id = m.group(1) if m else None
    running = eval_alive(task)
    datasets = {}
    for d in info.get("datasets", cfg["eval"]["datasets"]):
        rd = RUNS / task.method / cfg["eval"]["schema"] / d / run_id if run_id else None
        man = rd / "manifest.json" if rd else None
        if man and man.exists():
            mj = json.loads(man.read_text())
            datasets[d] = {"status": mj.get("status"), "run_dir": str(rd.relative_to(ROOT)),
                           "error": (mj.get("error") or "")[:500]}
        else:
            datasets[d] = {"status": "pending" if running else "missing"}
    failed = [d for d, v in datasets.items() if v["status"] != "success"]
    return {"launched": True, "running": running, "run_id": run_id, "pid": info.get("pid"),
            "log": info["log"], "datasets": datasets,
            "done": not running and bool(datasets) and not failed,
            "failed": [] if running else failed, "log_tail": log_text[-1500:]}


# ------------------------------------------------------------ classification
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
            if top == "EasyTSAD" or mod.startswith(method):
                return "ADAPTER_INTERFACE"   # wrong layout / relative import of the method itself
            return "ENVIRONMENT"
    if _ENV.search(text):
        return "ENVIRONMENT"
    frames = _FRAME.findall(text)
    mdir = f"/Methods/{method}/"
    own = [f for f in frames if mdir in f]
    if own:  # the innermost frame of the method decides: src/ = algorithm, adapter otherwise
        return "ALGORITHM_IMPLEMENTATION" if f"{mdir}src/" in own[-1] else "ADAPTER_INTERFACE"
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
    """`scripts/tsad check <env>` restricted to the output lines of this method."""
    env = method_env(task.method)
    if env is None:
        return {"ok": False, "env": None, "lines": [],
                "error": f"{task.method} is not registered in envs/registry.toml"}
    rc, out = _run([str(TSAD), "check", env], timeout)
    lines = [l.strip() for l in out.splitlines() if re.search(rf"\b{re.escape(task.method)}\b", l)]
    ok = any(l.startswith("ok") for l in lines) and not any(l.startswith("FAIL") for l in lines)
    res = {"ok": ok, "env": env, "lines": lines}
    if not ok:
        res["error"] = out[-3000:] if not lines else "\n".join(lines)
    return res


__all__ = ["method_env", "write_spec", "run_smoke", "latest_smoke", "launch_eval", "eval_status",
           "classify_failure", "check_scores", "check_method", "import_check", "find_run", "epochs_key", "epochs_claim", "non_iterative",
           "source_hash", "eval_alive", "ALLOWED_SCOPE",
           "FAILURE_CLASSES"]
