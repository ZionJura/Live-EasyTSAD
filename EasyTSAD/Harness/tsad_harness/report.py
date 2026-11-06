"""Module 3: results, leaderboard catalog, reproduction report and site publishing.

Everything here is deterministic (no LLM):
  * `write_catalog` adds/updates `[methods.<Name>]` in LiveLeaderBoard/catalog.toml
    (tomlkit, comments and ordering preserved, idempotent).
  * `export` runs the unchanged `scripts/tsad export`.
  * `write_report` renders `report.md` (task dir) and `REPRODUCTION.md` (method dir)
    from contract.json, notes.jsonl, corrections.jsonl, runs/*.json and leaderboard.json.
  * `publish` copies benchmark.json into the web app, builds it and optionally deploys.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import tomlkit

from .config import ROOT, HarnessError, load
from .task import Task, now, read_jsonl

CATALOG = ROOT / "LiveLeaderBoard" / "catalog.toml"
RESULTS_DIR = ROOT / "LiveLeaderBoard" / "results"
REGISTRY = ROOT / "envs" / "registry.toml"
FAMILIES = ("深度学习", "传统方法", "预训练模型")
METRICS = (("point_f1", "f1"), ("event_f1", "f1"), ("auprc", "value"))
STATUS_ORDER = ("explicit", "inferred", "unspecified")
KEEP_FIELDS = ("name", "family", "description", "color")   # never overwritten once present


# ------------------------------------------------------------------ helpers
def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _cell(v) -> str:
    """Markdown-table-safe single-line cell."""
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    if not isinstance(v, str):
        v = json.dumps(v, ensure_ascii=False)
    return v.replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _primary(summary: dict | None, key: str, field: str):
    v = (summary or {}).get(key)
    return v.get(field) if isinstance(v, dict) else v


def paper_info(task: Task) -> dict:
    """Merged paper metadata: contract < task.toml < paper/meta.json (most specific wins)."""
    meta = _json(task.dir / "paper" / "meta.json", {}) or {}
    code = meta.get("code") or {}
    info: dict = {}
    for src in ((_json(task.dir / "contract.json", {}) or {}).get("paper") or {},
                task.spec.get("paper") or {}, meta.get("paper") or {},
                {"code_url": code.get("url"), "code_commit": code.get("commit")}):
        info.update({k: v for k, v in src.items() if v not in (None, "", [])})
    if not info.get("code_url"):
        info["code_url"] = task.spec.get("code_url") or ""
    if not info.get("url") and info.get("arxiv_id"):
        info["url"] = f"https://arxiv.org/abs/{info['arxiv_id']}"
    return info


def method_env(method: str) -> str | None:
    """Environment that runs `method` according to envs/registry.toml."""
    try:
        envs = tomllib.loads(REGISTRY.read_text()).get("envs", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return next((n for n, e in envs.items() if isinstance(e, dict) and method in e.get("methods", [])), None)


# ------------------------------------------------------------------ catalog
def _rgb(hexcolor: str) -> tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def pick_color(name: str, existing: list[str]) -> str:
    """Deterministic muted `#rrggbb` from the name, at least a small RGB distance from `existing`."""
    used = [_rgb(c) for c in existing if isinstance(c, str) and len(c) == 7 and c.startswith("#")]
    seed = int(hashlib.sha1(name.encode()).hexdigest(), 16)
    best, best_d = "#8a8a8a", -1.0
    for i in range(72):
        hue = ((seed % 360) + i * 137.5) % 360 / 360
        sat = 0.18 + ((seed >> 9) + i) % 4 * 0.04
        lig = 0.50 + ((seed >> 13) + i) % 3 * 0.04
        rgb = tuple(round(c * 255) for c in colorsys.hls_to_rgb(hue, lig, sat))
        d = min((sum((a - b) ** 2 for a, b in zip(rgb, u)) ** 0.5 for u in used), default=999.0)
        color = "#%02x%02x%02x" % rgb
        if d >= 28:
            return color
        if d > best_d:
            best, best_d = color, d
    return best


def _description(task: Task, paper: dict) -> str:
    summary = ((_json(task.dir / "contract.json", {}) or {}).get("summary") or "").strip()
    text = summary.split(". ")[0].rstrip(".") if summary else paper.get("title", "")
    return text if len(text) <= 90 else text[:87].rstrip() + "..."


def catalog_entry(task: Task) -> dict:
    """Catalog fields for the task's method (empty optional fields omitted, color not yet chosen)."""
    paper = paper_info(task)
    family = task.spec.get("family") or load()["report"]["default_family"]
    entry = {
        "name": task.spec.get("display_name") or task.method,
        "family": family if family in FAMILIES else "深度学习",
        "description": _description(task, paper),
        "paper_title": paper.get("title", ""),
        "paper_url": paper.get("url", ""),
        "venue": paper.get("venue", ""),
        "year": paper.get("year"),
        "code_url": paper.get("code_url", ""),
        "reproduced_by": "tsad-harness",
    }
    if isinstance(entry["year"], str) and entry["year"].isdigit():
        entry["year"] = int(entry["year"])
    return {k: v for k, v in entry.items() if v not in (None, "")}


def write_catalog(task: Task) -> dict:
    """Add or update `[methods.<Name>]` in the catalog; keeps an existing color. Returns the entry."""
    doc = tomlkit.parse(CATALOG.read_text()) if CATALOG.exists() else tomlkit.document()
    if "methods" not in doc:
        doc["methods"] = tomlkit.table(is_super_table=True)
    methods = doc["methods"]
    entry = catalog_entry(task)
    cur = methods.get(task.method)
    color = cur.get("color") if cur is not None else None
    if not color:
        others = [str(m.get("color", "")) for k, m in methods.items() if k != task.method]
        color = pick_color(task.method, others)
    entry = {**entry, "color": str(color)}
    if cur is None:
        tbl = tomlkit.table()
        for k in ("name", "family", "description", "color", "paper_title", "paper_url",
                  "venue", "year", "code_url", "reproduced_by"):
            if k in entry:
                tbl[k] = entry[k]
        methods[task.method] = tbl
    else:  # keep curated display fields, refresh the paper metadata
        for k, v in entry.items():
            if k in KEEP_FIELDS and cur.get(k):
                entry[k] = cur[k]
            elif cur.get(k) != v:
                cur[k] = v
    text = tomlkit.dumps(doc)
    tomllib.loads(text)  # never write an unparsable catalog
    if not CATALOG.exists() or CATALOG.read_text() != text:
        CATALOG.write_text(text)
    task.log("report/catalog", entry=entry, path=str(CATALOG))
    return entry


# ------------------------------------------------------------------- export
def export() -> Path:
    """Run `scripts/tsad export` (regenerates results/*.json); returns the benchmark.json path."""
    p = subprocess.run([str(ROOT / "scripts" / "tsad"), "export"], cwd=ROOT, env=load().network_env(),
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise HarnessError(f"scripts/tsad export failed:\n{(p.stdout + p.stderr)[-3000:]}")
    return RESULTS_DIR / "benchmark.json"


# ------------------------------------------------------------------- report
def _leaderboard_rows(method: str) -> list[dict]:
    lb = _json(RESULTS_DIR / "leaderboard.json", {}) or {}
    return [r for r in lb.get("results", []) if r.get("method") == method]


def _run_outcomes(task: Task) -> list[tuple[str, dict]]:
    out = []
    for p in sorted((task.dir / "runs").glob("*.json")):
        d = _json(p)
        if p.stem != "smoke_latest" and isinstance(d, dict) and "status" in d:
            out.append((p.stem, d))
    return out


def render_report(task: Task) -> str:
    """Markdown fidelity + results report for the task."""
    paper = paper_info(task)
    contract = _json(task.dir / "contract.json", {}) or {}
    notes = read_jsonl(task.dir / "notes.jsonl")
    corrections = read_jsonl(task.dir / "corrections.jsonl")
    env = method_env(task.method) or "(not registered)"
    s = [f"# Reproduction report: {task.method}", "",
         f"Generated by tsad-harness on {now()}. Task `{task.id}`, phase `{task.phase.value}`, "
         f"status `{task.status.value}`.", "", "## Paper", ""]
    for label, key in (("Title", "title"), ("Authors", "authors"), ("Venue", "venue"), ("Year", "year"),
                       ("arXiv", "arxiv_id"), ("URL", "url"), ("Official code", "code_url"),
                       ("Code commit", "code_commit")):
        v = paper.get(key)
        if v:
            s.append(f"- {label}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    if s[-1] == "":
        s.append("Unknown.")
    s += ["", "## Contract", ""]
    if not contract:
        s.append("No contract.json yet.")
    else:
        for k in ("task_type", "summary"):
            if contract.get(k):
                s.append(f"- {k.replace('_', ' ').capitalize()}: {contract[k]}")
        score = contract.get("score") or {}
        for k in ("definition", "alignment"):
            if score.get(k):
                s.append(f"- Score {k}: {score[k]}")
        if contract.get("dependencies"):
            s.append(f"- Dependencies: {', '.join(map(str, contract['dependencies']))}")
        oc = contract.get("official_code") or {}
        if oc.get("url"):
            s.append(f"- Official code used: {oc['url']} @ {oc.get('commit') or '?'}"
                     + (f" ({', '.join(oc['used_files'])})" if oc.get("used_files") else ""))
        claims = contract.get("claims") or []
        counts = {st: sum(c.get("status") == st for c in claims) for st in STATUS_ORDER}
        s += ["", "Claims: " + ", ".join(f"{n} {st}" for st, n in counts.items()), "",
              _table(["id", "value", "status", "source", "mutable"],
                     [[c.get("id"), repr(c["value"]) if isinstance(c.get("value"), float) else c.get("value"), c.get("status"), c.get("source"), c.get("mutable")]
                      for c in claims])]
        if contract.get("open_questions"):
            s += ["", "Open questions:"] + [f"- {q}" for q in contract["open_questions"]]
    s += ["", "## Deviations, assumptions and issues", ""]
    if not notes:
        s.append("None recorded.")
    for kind in ("deviation", "assumption", "issue"):
        items = [n for n in notes if n.get("kind") == kind]
        if items:
            s += ["" if s[-1] else None, f"### {kind.capitalize()}s", ""]
            s += [f"- {n.get('text', '')}" + (f" (claim `{n['contract_id']}`)" if n.get("contract_id") else "")
                  for n in items]
    s = [x for x in s if x is not None]
    s += ["", "## Environment", "", f"- uv env: `{env}`",
          f"- Method folder: `EasyTSAD/Methods/{task.method}/`"]
    runs = _run_outcomes(task)
    s += ["", "## Smoke and harness runs", ""]
    if runs:
        s.append(_table(["run", "status", "failure class"] + [m for m, _ in METRICS],
                        [[name, d.get("status"), d.get("failure_class")]
                         + [_primary(d.get("metrics_summary"), m, f) for m, f in METRICS] for name, d in runs]))
    else:
        s.append("No run outcomes recorded.")
    rows = _leaderboard_rows(task.method)
    s += ["", "## Evaluation (LiveLeaderBoard)", ""]
    if rows:
        s.append(_table(["schema", "dataset"] + [m for m, _ in METRICS] + ["curves", "partial", "run"],
                        [[r.get("schema"), r.get("dataset")] + [_primary(r.get("metrics"), m, f) for m, f in METRICS]
                         + [r.get("n_curves_evaluated"), "yes" if r.get("partial") else "no", r.get("run_path")]
                         for r in rows]))
        if any(r.get("partial") for r in rows):
            s.append("\nPartial runs (curve subsets) are not exported to the web leaderboard.")
    else:
        s.append("No successful runs in LiveLeaderBoard/results/leaderboard.json.")
    s += ["", "## Human corrections", ""]
    if corrections:
        s.append(_table(["at", "author", "phase", "target phase", "note"],
                        [[c.get("at"), c.get("author"), c.get("phase"), c.get("target_phase"), c.get("note")]
                         for c in corrections]))
    else:
        s.append("None.")
    return "\n".join(s).rstrip() + "\n"


def write_report(task: Task) -> Path:
    """Write `<task>/report.md` and, when the method folder exists, its REPRODUCTION.md."""
    text = render_report(task)
    path = task.dir / "report.md"
    path.write_text(text)
    written = [str(path)]
    if task.method_dir.is_dir():
        (task.method_dir / "REPRODUCTION.md").write_text(text)
        written.append(str(task.method_dir / "REPRODUCTION.md"))
    task.log("report/written", paths=written)
    return path


# ------------------------------------------------------------------ publish
def publish(*, build: bool = True, deploy: bool = False) -> dict:
    """Export, copy benchmark.json into the web app, optionally build and (only if asked) deploy."""
    cfg = load()
    web = cfg.path("report", "leaderboard_dir")
    bm = export()
    dst = web / "public" / "benchmark.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bm, dst)
    out: dict = {"benchmark": str(bm), "public": str(dst), "built": False, "deployed": False}

    def sh(cmd: list[str], what: str) -> None:
        p = subprocess.run(cmd, cwd=web, env=cfg.network_env(), capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise HarnessError(f"{what} failed:\n{(p.stdout + p.stderr)[-3000:]}")

    if build:
        sh(["npm", "run", "build"], "npm run build")
        out["built"] = True
    if deploy:
        script = cfg.path("report", "deploy_script")
        sh(["bash", str(script)] + (["--skip-build"] if build else []), "deploy")
        out["deployed"] = True
    return out
