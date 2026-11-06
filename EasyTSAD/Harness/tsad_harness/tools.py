"""TSAD MCP tool server (stdio) used by the DSH agent as `mcp__tsad__<name>`.

Started by DSH as `python -m tsad_harness.tools` with `TSAD_TASK_DIR` pointing at
the task directory. Every tool re-reads the task from disk (the pipeline and
this server are separate processes), logs the call to the task journal as
`tool/<name>` and returns a compact JSON string.

Heavy modules (runner/validate pull in numpy and tomlkit) are imported inside the
tools, so the server answers MCP `initialize` as soon as `mcp` itself is loaded.
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

try:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from .config import ROOT, load
from .task import Task, now, read_jsonl

NOTE_KINDS = ("deviation", "assumption", "issue")
mcp = FastMCP("tsad", log_level="WARNING")
tool = mcp.tool(structured_output=False)   # plain JSON-string results


def _task() -> Task:
    d = Path(os.environ["TSAD_TASK_DIR"]).resolve()
    with open(d / "task.toml", "rb") as f:
        spec = tomllib.load(f)
    return Task(d, spec, json.loads((d / "state.json").read_text()))


def _append(p: Path, rec: dict) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _reply(task: Task, name: str, args: dict, result: dict) -> str:
    out = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    task.log(f"tool/{name}", args=args, result=out[:3000])
    return out


@tool
def task_info() -> str:
    """Task overview: method, phase, paths, env, paper, contract status, remaining smoke runs, notes."""
    from . import runner, validate
    t = _task()
    cfg = load()
    s = cfg["smoke"]
    notes = read_jsonl(t.dir / "notes.jsonl")
    qs = read_jsonl(t.dir / "questions.jsonl")
    info = {"method": t.method, "phase": t.phase.value, "status": t.status.value,
            "task_dir": str(t.dir), "method_dir": str(t.method_dir.relative_to(ROOT)),
            "method_dir_exists": t.method_dir.exists(), "env": runner.method_env(t.method),
            "paper": t.spec.get("paper", {}), "code_url": t.spec.get("code_url", ""),
            "contract": "valid" if (t.dir / "contract.json").exists() and not validate.contract(t)
            else ("invalid" if (t.dir / "contract.json").exists() else "missing"),
            "smoke": {"dataset": s["dataset"], "curves": s["curves"], "schema": s["schema"],
                      "max_epochs": s.get("max_epochs"),
                      "runs_left": int(cfg["budget"]["max_smoke_runs"]) - t.counter("smoke_runs")},
            "last_smoke": {k: (runner.latest_smoke(t) or {}).get(k) for k in ("n", "status", "failure_class")},
            "notes": {k: sum(n["kind"] == k for n in notes) for k in NOTE_KINDS},
            "open_questions": [q for q in qs if not q.get("answered")]}
    return _reply(t, "task_info", {}, info)


@tool
def save_contract(contract: dict) -> str:
    """Validate and save the reproduction contract (contract.json). Returns errors to fix, if any."""
    from . import validate
    t = _task()
    errs = validate.contract_errors(contract, t.method)
    if errs:
        n = len(contract["claims"]) if isinstance(contract.get("claims"), list) else 0
        return _reply(t, "save_contract", {"claims": n},
                      {"ok": False, "errors": errs})
    (t.dir / "contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    return _reply(t, "save_contract", {"claims": len(contract["claims"])},
                  {"ok": True, "path": str(t.dir / "contract.json"), "claims": len(contract["claims"])})


@tool
def record_note(kind: str, text: str, contract_id: str = "") -> str:
    """Record a deviation, assumption or issue (kind = deviation|assumption|issue); contract_id = claim id."""
    t = _task()
    args = {"kind": kind, "text": text, "contract_id": contract_id}
    if kind not in NOTE_KINDS or not text.strip():
        return _reply(t, "record_note", args, {"ok": False, "error": f"kind must be one of {NOTE_KINDS} "
                                                                      f"and text non-empty"})
    p = t.dir / "notes.jsonl"
    rec = {"id": f"n{len(read_jsonl(p)) + 1}", "at": now(), "phase": t.phase.value, **args}
    _append(p, rec)
    return _reply(t, "record_note", args, {"ok": True, "id": rec["id"]})


@tool
def check_method() -> str:
    """Import-check this method in its registered uv env (`scripts/tsad check <env>`, method lines only)."""
    from . import runner
    t = _task()
    return _reply(t, "check_method", {}, runner.check_method(t))


@tool
def run_smoke(epochs: int | None = None) -> str:
    """Run the smoke experiment through scripts/tsad exp (counts against the smoke budget).

    epochs: optional, capped at [smoke].max_epochs. Returns status, failure_class,
    allowed edit scope, error, log tail, score check and metrics (metrics are not a target).
    """
    from . import runner
    t = _task()
    limit = int(load()["budget"]["max_smoke_runs"])
    if t.counter("smoke_runs") >= limit:
        return _reply(t, "run_smoke", {"epochs": epochs},
                      {"status": "refused", "error": f"smoke budget exhausted ({limit} runs); "
                                                     f"call ask_human or end your turn"})
    res = runner.run_smoke(t, epochs=epochs)
    res["runs_left"] = limit - t.counter("smoke_runs")
    res.pop("exp_output", None)
    return _reply(t, "run_smoke", {"epochs": epochs}, res)


@tool
def ask_human(question: str) -> str:
    """Ask the human a question; the task pauses (AWAITING_HUMAN) after this turn. End your turn next."""
    t = _task()
    p = t.dir / "questions.jsonl"
    qid = f"q{len(read_jsonl(p)) + 1}"
    _append(p, {"id": qid, "at": now(), "phase": t.phase.value, "question": question, "answered": False})
    t.state["pending_question"] = True
    t.save()
    return _reply(t, "ask_human", {"question": question},
                  {"ok": True, "id": qid, "note": "question recorded; end your turn now"})


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
