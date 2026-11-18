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

from .config import ROOT
from .task import Task, now

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
    """Source location, environment, execution errors and user corrections."""
    from . import runner
    t = _task()
    handoff = t.dir / "handoff.json"
    info = {"method": t.method, "phase": t.phase.value, "status": t.status.value,
            "task_dir": str(t.dir), "method_dir": str(t.method_dir.relative_to(ROOT)),
            "method_dir_exists": t.method_dir.exists(), "env": runner.method_env(t.method),
            "paper": t.spec.get("paper", {}), "code_url": t.spec.get("code_url", ""),
            "integration": json.loads(handoff.read_text()) if handoff.exists() else None}
    return _reply(t, "task_info", {}, info)


@tool
def check_method() -> str:
    """Import-check only this method in its registered uv environment."""
    from . import runner
    t = _task()
    return _reply(t, "check_method", {}, runner.check_method(t))


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
