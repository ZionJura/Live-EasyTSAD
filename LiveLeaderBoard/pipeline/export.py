"""Aggregate canonical EasyTSAD results and legacy run directories into leaderboard tables.

Uses stdlib on Python >= 3.11, or the runner's existing toml on Python 3.9/3.10.

Outputs (regenerated from Results/Evals and legacy results/runs):
  results/leaderboard.json   latest successful run per (method, schema, dataset)
                             with all summary metrics and a pointer to the run
  results/benchmark.json     the same results in the format consumed by the
                             leaderboard web app (leaderboard/src/data.ts)
  leaderboard/public/benchmark.json  byte-identical published copy
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import math
import hashlib
import os
import tempfile
import time
import fcntl
from contextlib import contextmanager
try:
    import tomllib
except ImportError:  # method environments may use Python 3.9/3.10
    import toml as tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "LiveLeaderBoard"
RESULTS_DIR = LB_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CATALOG = LB_DIR / "catalog.toml"
FAMILIES = {"深度学习", "传统方法", "预训练模型"}
WEB_SCHEMAS = {"mts", "naive", "all_in_one", "zero_shot", "zero_shot_cross_ds"}


PROTOCOL_KEYS = {
    "point_f1": "best f1 under pa",
    "event_f1": "event-based f1 under pa with mode log",
    "event_f1_squeeze": "event-based f1 under pa with mode squeeze",
    "event_f1_raw": "event-based f1 under pa with mode raw",
    "auprc": "point-based auprc", "auroc": "point-based auroc",
    "auprc_pa": "point-based auprc pa", "point_f1_nopa": "point-wise f1",
}


def canonical_runs():
    """Read original protocol JSON, with attempt metadata when available."""
    manifests = {}
    for path in (ROOT / "EasyTSAD/Methods").glob("*/reproduction/runs/*/*/*/manifest.json"):
        man = json.loads(path.read_text())
        if man.get("result_format") != "easytsad-v1" or man.get("status") != "success":
            continue
        if not man.get("canonical_metrics_committed", False):
            continue
        key = (man["method"], man["schema"], man["dataset"])
        manifests.setdefault(key, []).append((man, path.parent))
    for avg_path in sorted((ROOT / "Results/Evals").glob("*/*/*/avg.json")):
        method, schema, dataset = avg_path.parts[-4:-1]
        key = (method, schema, dataset)
        all_path = avg_path.with_name("all.json")
        if not all_path.exists():
            continue
        avg_bytes, all_bytes = avg_path.read_bytes(), all_path.read_bytes()
        hashes = {"avg_path": hashlib.sha256(avg_bytes).hexdigest(),
                  "all_path": hashlib.sha256(all_bytes).hexdigest()}
        def matched_hashes(man):
            recorded = man.get("result_sha256", {})
            normalized = {}
            for field in ("avg_path", "all_path"):
                reference = man.get(field, "")
                normalized[field] = recorded.get(field, recorded.get(reference))
            return normalized == hashes
        matching = [(man, directory) for man, directory in manifests.get(key, []) if matched_hashes(man)]
        if matching:
            man, metadata_dir = max(matching, key=lambda item: item[0].get("finished_at") or "")
            man = dict(man)
        elif manifests.get(key):
            # Never combine half of a new native pair with an older manifest.
            # Canonical writers and this exporter share export_lock.
            continue
        else:
            # Original TSADController output predating runner metadata is valid.
            stamp = dt.datetime.fromtimestamp(avg_path.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            metadata_dir = avg_path.parent
            man = {"method": method, "schema": schema, "dataset": dataset,
                   "status": "success", "run_id": "canonical", "finished_at": stamp,
                   "dataset_type": "MTS" if schema == "mts" else "UTS", "duration_s": None,
                   "config_hash": None, "git": {}, "env": {}, "config": {}}
        average = json.loads(avg_bytes)
        per_curve = json.loads(all_bytes)
        metric_keys = dict(PROTOCOL_KEYS)
        metric_keys.update(man.get("metric_keys", {}))
        summary = {}
        for stable, protocol in metric_keys.items():
            value = average.get(protocol, average.get(stable))
            if value is not None:
                summary[stable] = value if isinstance(value, dict) else {"value": value}
        man["_dir"] = metadata_dir
        man["_metrics"] = {"summary": summary, "n_curves_evaluated": len(per_curve)}
        man["_canonical"] = True
        yield man


def latest_runs(*, full_only=False) -> dict[tuple[str, str, str], dict]:
    """Newest successful candidate, excluding partials BEFORE selection for web."""
    best = {}
    candidates = list(canonical_runs())
    for man_path in RUNS_DIR.glob("*/*/*/*/manifest.json"):
        man = json.loads(man_path.read_text())
        metrics_path = man_path.parent / "metrics.json"
        if man.get("status") != "success" or not metrics_path.exists():
            continue
        man["_dir"] = man_path.parent
        man["_metrics"] = json.loads(metrics_path.read_text())
        candidates.append(man)
    for man in candidates:
        if full_only and (man.get("partial", bool(man.get("config", {}).get("curves"))) or man.get("schema") not in WEB_SCHEMAS):
            continue
        key = (man["method"], man["schema"], man["dataset"])
        previous = best.get(key)
        # Canonical outputs win ties with imported legacy metadata.
        if previous is None or (man.get("finished_at") or "", bool(man.get("_canonical"))) > (previous.get("finished_at") or "", bool(previous.get("_canonical"))):
            best[key] = man
    return best


def _atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


@contextmanager
def export_lock():
    """Reopen the same lock after transient EINVAL, at most three attempts.

    Persistent filesystem locking errors still abort; no alternate lock is used.
    Closing the descriptor releases the lock, including on process death.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        with open(RESULTS_DIR / ".export.lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            except OSError as error:
                if error.errno != errno.EINVAL or attempt == 2:
                    raise
            else:
                yield
                return
        time.sleep(0.1 * (attempt + 1))


def _recover_pending():
    intent = RESULTS_DIR / ".benchmark-pending.json"
    if not intent.exists():
        return False
    pending = json.loads(intent.read_text())
    payload = pending["payload"].encode()
    if hashlib.sha256(payload).hexdigest() != pending["sha256"]:
        raise ValueError("benchmark pending intent checksum mismatch")
    for path in (RESULTS_DIR / "benchmark.json", ROOT / "leaderboard/public/benchmark.json"):
        _atomic(path, payload)
    _atomic(RESULTS_DIR / "benchmark-publication.json", _encode({"revision": pending["revision"], "sha256": pending["sha256"], "status": "committed"}))
    intent.unlink()
    return True


def recover_pending():
    """Complete an interrupted two-target publication; safe at service startup."""
    with export_lock():
        return _recover_pending()


def _publish(benchmark):
    revision = hashlib.sha256(_encode(benchmark)).hexdigest()
    benchmark["revision"] = revision
    payload = _encode(benchmark)
    _atomic(RESULTS_DIR / ".benchmark-pending.json", _encode({"revision": revision, "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload.decode()}))
    _recover_pending()


def export() -> tuple[Path, Path]:
    with export_lock():
        _recover_pending()
        return _export()


def _export() -> tuple[Path, Path]:
    catalog = tomllib.loads(CATALOG.read_text()) if CATALOG.exists() else {}
    cat_methods = catalog.get("methods", {})
    cat_datasets = catalog.get("datasets", {})
    metric_map = catalog.get("export", {}).get("metrics", {})

    rows = []
    full_rows = []
    invalid = []
    candidates = latest_runs()
    full_candidates = latest_runs(full_only=True)
    # Include both winners when the latest run is a partial.
    manifests = {str(m["_dir"]): m for m in list(candidates.values()) + list(full_candidates.values())}
    for man in sorted(manifests.values(), key=lambda m: str(m["_dir"])):
        method, schema, dataset = man["method"], man["schema"], man["dataset"]
        metrics = man["_metrics"]
        row = {
            "method": method, "schema": schema, "dataset": dataset,
            "dataset_type": man["dataset_type"],
            "run_id": man["run_id"], "experiment": man.get("experiment"),
            "run_path": str(man["_dir"].relative_to(ROOT) if man.get("_canonical") else man["_dir"].relative_to(RESULTS_DIR)),
            "finished_at": man["finished_at"], "duration_s": man["duration_s"],
            "config_hash": man.get("config_hash"), "git_commit": man.get("git", {}).get("commit"),
            "git_dirty": man.get("git", {}).get("dirty"), "env": man.get("env", {}).get("name"),
            "partial": man.get("partial", bool(man.get("config", {}).get("curves"))),
            "n_curves_evaluated": metrics["n_curves_evaluated"],
            "metrics": metrics["summary"],
        }
        key = (method, schema, dataset)
        if candidates.get(key, {}).get("_dir") == man["_dir"]:
            rows.append(row)
        if full_candidates.get(key, {}).get("_dir") == man["_dir"]:
            full_rows.append(row)
    def clean(value, path=""):
        if isinstance(value, dict):
            return {k: clean(v, path + "/" + str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, path + "/" + str(i)) for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            invalid.append({"path": path, "value": repr(value), "error": "non-finite metric"})
            return None
        return value
    rows = clean(rows)
    full_rows = clean(full_rows)
    today = dt.date.today().isoformat()
    lb_path = RESULTS_DIR / "leaderboard.json"
    _atomic(lb_path, _encode({"schema_version": 1, "generated_at": today, "results": rows}))

    # ---- web app format (version 1). Partial runs (curve subsets) are excluded.
    full = full_rows
    methods, datasets = [], []
    for m in sorted({r["method"] for r in full}):
        meta = cat_methods.get(m, {})
        fam = meta.get("family", "深度学习")
        entry = {"id": m, "name": meta.get("name", m),
                 "family": fam if fam in FAMILIES else "深度学习",
                 "description": meta.get("description", ""),
                 "color": meta.get("color", "#8a8a8a")}
        # Optional paper metadata (e.g. written by the reproduction harness); omitted when absent.
        paper = {k: meta[k] for k in ("paper_title", "paper_url", "venue", "year", "code_url") if meta.get(k)}
        if paper:
            entry["paper"] = paper
        methods.append(entry)
    for d in sorted({r["dataset"] for r in full}):
        meta = cat_datasets.get(d, {})
        task = meta.get("task") or next(r["dataset_type"].lower() for r in full if r["dataset"] == d)
        datasets.append({"id": d, "task": task, "domain": meta.get("domain", "")})
    results = []
    for r in full:
        vals = {}
        for web_key, (src_key, field) in metric_map.items():
            record = r["metrics"].get(src_key, {})
            if not isinstance(record, dict):
                invalid.append({"run_path": r["run_path"], "metric": web_key, "value": repr(record), "error": "expected metric record object"})
                record = {}
            v = record.get(field)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1):
                invalid.append({"run_path": r["run_path"], "metric": web_key, "value": repr(v), "error": "expected finite number in [0, 1]"})
                v = None
            vals[web_key] = round(float(v), 6) if v is not None else None
        results.append({"method": r["method"], "dataset": r["dataset"], "schema": r["schema"],
                        "metrics": vals})
    bm_path = RESULTS_DIR / "benchmark.json"
    _atomic(RESULTS_DIR / "export-diagnostics.json", _encode({"invalid_values": invalid}))
    _publish({"version": 1, "demo": False, "updatedAt": today,
              "methods": methods, "datasets": datasets, "results": results})
    return lb_path, bm_path


def table(schema: str | None = None, metric: str = "event_f1") -> str:
    lb = json.loads((RESULTS_DIR / "leaderboard.json").read_text())["results"]
    lines = []
    for sch in sorted({r["schema"] for r in lb if schema in (None, r["schema"])}):
        rs = [r for r in lb if r["schema"] == sch]
        ds = sorted({r["dataset"] for r in rs})
        ms = sorted({r["method"] for r in rs})
        get = {(r["method"], r["dataset"]): r for r in rs}
        lines.append(f"\n[{sch}] metric={metric}")
        lines.append(f"{'method':<16}" + "".join(f"{d:>10}" for d in ds))
        for m in ms:
            cells = []
            for d in ds:
                r = get.get((m, d))
                v = r["metrics"].get(metric, {}) if r else {}
                v = v.get("f1", v.get("value")) if v else None
                cells.append(f"{v:>9.4f}{'*' if r and r['partial'] else ' '}" if v is not None else f"{'-':>10}")
            lines.append(f"{m:<16}" + "".join(cells))
    if any(r["partial"] for r in lb):
        lines.append("\n* partial run (curve subset), not exported to benchmark.json")
    return "\n".join(lines)
