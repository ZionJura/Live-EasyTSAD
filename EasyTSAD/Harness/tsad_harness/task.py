"""Task directory layout, phase state machine and the task journal.

A task reproduces one paper as one EasyTSAD method. Everything the harness
knows about it lives in `<tasks_dir>/<task_id>/`:

    task.toml           inputs (method name, paper ids, code url, env hint)
    state.json          current phase, phase history, counters, DSH session id
    journal.jsonl       append-only log of every harness action (for analysis)
    paper/              paper.pdf, paper.md (text), meta.json
    code/               official repository clone (read-only reference)
    contract.json       reproduction contract written in EVIDENCE
    notes.jsonl         deviations/assumptions recorded by the agent
    corrections.jsonl   human corrections with context snapshots
    dsh/                rendered DSH patch, per-run event streams, sessions/
    runs/               smoke/eval specs and their outcomes
    guard/              file-guard baseline
    report.md           fidelity + results report
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import HarnessError, load

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")   # must be a valid EasyTSAD class name


class Phase(str, Enum):
    FETCH = "FETCH"            # deterministic: download paper + code
    EVIDENCE = "EVIDENCE"      # agent: read paper/code -> reproduction contract
    IMPLEMENT = "IMPLEMENT"    # agent: method folder + uv env + import check
    SMOKE = "SMOKE"            # agent: run small pipeline case, repair until it passes
    EVALUATE = "EVALUATE"      # deterministic: full `scripts/tsad exp`
    REPORT = "REPORT"          # deterministic: catalog entry, export, report.md
    DONE = "DONE"

    @property
    def next(self) -> "Phase":
        order = list(Phase)
        return order[min(order.index(self) + 1, len(order) - 1)]


AGENT_PHASES = (Phase.EVIDENCE, Phase.IMPLEMENT, Phase.SMOKE)


class Status(str, Enum):
    READY = "ready"            # phase can be started
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"   # review gate or a question from the agent
    FAILED = "failed"          # budget exhausted / validation failed; needs a human
    DONE = "done"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, splitting on '\\n' only (json.dumps(ensure_ascii=False) may emit
    U+2028 etc. raw, which str.splitlines() would split on). Undecodable lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


_HELD: dict[str, object] = {}   # task dir -> open lock file; lock() is re-entrant within a process


@dataclass
class Task:
    dir: Path
    spec: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- creation
    @classmethod
    def create(cls, method: str, *, paper: dict, code_url: str | None = None,
               env_hint: str | None = None, family: str | None = None) -> "Task":
        if not ID_RE.match(method):
            raise HarnessError(f"method name {method!r} must match {ID_RE.pattern}")
        cfg = load()
        if (cfg.methods_dir / method).exists():
            raise HarnessError(f"EasyTSAD/Methods/{method} already exists")
        d = cfg.tasks_dir / method
        if d.exists():
            raise HarnessError(f"task {method} already exists at {d}")
        for sub in ("paper", "code", "dsh", "runs", "guard"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        spec = {"method": method, "created_at": now(), "paper": paper,
                "code_url": code_url or "", "env_hint": env_hint or "",
                "family": family or cfg["report"]["default_family"]}
        (d / "task.toml").write_text(_dump_toml(spec))
        t = cls(d, spec, {"phase": Phase.FETCH.value, "status": Status.READY.value,
                          "session_id": None, "history": [], "counters": {}})
        t.save()
        t.log("task/created", spec=spec)
        return t

    @classmethod
    def open(cls, task_id: str) -> "Task":
        d = load().tasks_dir / task_id
        if not (d / "state.json").exists():
            raise HarnessError(f"no task {task_id!r} in {d.parent}")
        with open(d / "task.toml", "rb") as f:
            spec = tomllib.load(f)
        return cls(d, spec, json.loads((d / "state.json").read_text()))

    @staticmethod
    def all() -> list["Task"]:
        root = load().tasks_dir
        return [Task.open(p.parent.name) for p in sorted(root.glob("*/state.json"))]

    # -------------------------------------------------------------- accessors
    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def method(self) -> str:
        return self.spec["method"]

    @property
    def method_dir(self) -> Path:
        return load().methods_dir / self.method

    @property
    def phase(self) -> Phase:
        return Phase(self.state["phase"])

    @property
    def status(self) -> Status:
        return Status(self.state["status"])

    def counter(self, name: str) -> int:
        return self.state["counters"].get(name, 0)

    def bump(self, name: str) -> int:
        self.state["counters"][name] = self.counter(name) + 1
        self.save()
        return self.state["counters"][name]

    # ------------------------------------------------------------ transitions
    def set(self, phase: Phase | None = None, status: Status | None = None, **extra) -> None:
        old = (self.state["phase"], self.state["status"])
        if phase is not None:
            self.state["phase"] = phase.value
        if status is not None:
            self.state["status"] = status.value
        self.state.update(extra)
        new = (self.state["phase"], self.state["status"])
        if new != old:
            self.state["history"].append({"at": now(), "phase": new[0], "status": new[1],
                                          **({"reason": extra["reason"]} if "reason" in extra else {})})
            self.log("state/change", old=list(old), new=list(new), **extra)
        self.save()

    def reload(self) -> "Task":
        """Re-read state.json (the MCP tool server or another CLI process may have written it)."""
        self.state = json.loads((self.dir / "state.json").read_text())
        return self

    def save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp, self.dir / "state.json")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def lock(self):
        """Exclusive, non-blocking flock on <task>/.lock (held by `harness run` and CLI mutations)."""
        key = str(self.dir.resolve())
        if key in _HELD:                       # already held by this process: nest
            yield self
            return
        f = open(self.dir / ".lock", "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise HarnessError(f"task {self.id} is busy (another harness process holds the lock)") from None
        _HELD[key] = f
        try:
            yield self
        finally:
            del _HELD[key]
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    # ---------------------------------------------------------------- journal
    def log(self, event: str, **data) -> None:
        rec = {"at": now(), "event": event, "phase": self.state.get("phase"), **data}
        with open(self.dir / "journal.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    read_jsonl = staticmethod(read_jsonl)

    def journal(self) -> list[dict]:
        return read_jsonl(self.dir / "journal.jsonl")


def _dump_toml(d: dict, prefix: str = "") -> str:
    """Tiny TOML writer for flat task specs (str/int/float/bool/list[str] + one level of tables)."""
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return json.dumps(str(v), ensure_ascii=False)
    lines, tables = [], []
    for k, v in d.items():
        (tables if isinstance(v, dict) else lines).append((k, v))
    out = "".join(f"{k} = {val(v)}\n" for k, v in lines)
    for k, v in tables:
        out += f"\n[{prefix}{k}]\n" + _dump_toml(v, prefix=f"{prefix}{k}.").lstrip("\n")
    return out
