"""Aggregate LiveLeaderBoard run directories into leaderboard tables.

Stdlib only (runs under the `scripts/tsad` interpreter, Python >= 3.11).

Outputs (both regenerated from results/runs on every call):
  results/leaderboard.json   latest successful run per (method, schema, dataset)
                             with all summary metrics and a pointer to the run
  results/benchmark.json     the same results in the format consumed by the
                             leaderboard web app (leaderboard/src/data.ts)
"""
from __future__ import annotations

import datetime as dt
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "LiveLeaderBoard"
RESULTS_DIR = LB_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CATALOG = LB_DIR / "catalog.toml"
FAMILIES = {"深度学习", "传统方法", "预训练模型"}
WEB_SCHEMAS = {"mts", "naive", "all_in_one", "zero_shot", "zero_shot_cross_ds"}


def latest_runs() -> dict[tuple[str, str, str], dict]:
    """Newest successful manifest per (method, schema, dataset)."""
    best: dict[tuple[str, str, str], dict] = {}
    for man_path in RUNS_DIR.glob("*/*/*/*/manifest.json"):
        man = json.loads(man_path.read_text())
        if man.get("status") != "success" or not (man_path.parent / "metrics.json").exists():
            continue
        key = (man["method"], man["schema"], man["dataset"])
        if key not in best or man["finished_at"] > best[key]["finished_at"]:
            man["_dir"] = man_path.parent
            best[key] = man
    return best


def export() -> tuple[Path, Path]:
    catalog = tomllib.loads(CATALOG.read_text()) if CATALOG.exists() else {}
    cat_methods = catalog.get("methods", {})
    cat_datasets = catalog.get("datasets", {})
    metric_map = catalog.get("export", {}).get("metrics", {})

    rows = []
    for (method, schema, dataset), man in sorted(latest_runs().items()):
        metrics = json.loads((man["_dir"] / "metrics.json").read_text())
        rows.append({
            "method": method, "schema": schema, "dataset": dataset,
            "dataset_type": man["dataset_type"],
            "run_id": man["run_id"], "experiment": man.get("experiment"),
            "run_path": str(man["_dir"].relative_to(RESULTS_DIR)),
            "finished_at": man["finished_at"], "duration_s": man["duration_s"],
            "config_hash": man["config_hash"], "git_commit": man["git"]["commit"],
            "git_dirty": man["git"]["dirty"], "env": man["env"]["name"],
            "partial": bool(man["config"].get("curves")),
            "n_curves_evaluated": metrics["n_curves_evaluated"],
            "metrics": metrics["summary"],
        })
    today = dt.date.today().isoformat()
    lb_path = RESULTS_DIR / "leaderboard.json"
    lb_path.write_text(json.dumps({"schema_version": 1, "generated_at": today, "results": rows},
                                  indent=2, ensure_ascii=False) + "\n")

    # ---- web app format (version 1). Partial runs (curve subsets) are excluded.
    full = [r for r in rows if not r["partial"] and r["schema"] in WEB_SCHEMAS]
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
            v = r["metrics"].get(src_key, {}).get(field)
            vals[web_key] = round(min(max(float(v), 0.0), 1.0), 6) if v is not None else None
        results.append({"method": r["method"], "dataset": r["dataset"], "schema": r["schema"],
                        "metrics": vals})
    bm_path = RESULTS_DIR / "benchmark.json"
    bm_path.write_text(json.dumps({"version": 1, "demo": False, "updatedAt": today,
                                   "methods": methods, "datasets": datasets, "results": results},
                                  indent=2, ensure_ascii=False) + "\n")
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
