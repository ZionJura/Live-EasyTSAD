"""Module 4: human correction, approval and records.

Every human intervention (correction, approval note, answer to an agent
question) is appended to `<task>/corrections.jsonl` together with a context
snapshot (what the agent last did, git changes, latest smoke result) so that
interventions can be analysed later. Un-consumed corrections are injected at
the top of the next agent prompt (`format_for_prompt`) and then marked
consumed by the pipeline (`mark_consumed`).

corrections.jsonl record:
    {id, at, author, kind, phase, target_phase, note, question_id?, consumed,
     consumed_at?, snapshot}
"""
from __future__ import annotations

import getpass
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from .config import HarnessError, ROOT
from .task import Phase, Status, Task, now, read_jsonl

MAX_STR = 600          # truncation length for strings in event snapshots


# ----------------------------------------------------------------- helpers
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))
    os.replace(tmp, path)


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _trunc(v, n: int = MAX_STR):
    """Recursively truncate long strings inside an event payload."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"... [+{len(v) - n} chars]"
    if isinstance(v, dict):
        return {k: _trunc(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_trunc(x, n) for x in v[:50]]
    return v


def _corrections_path(task: Task) -> Path:
    return task.dir / "corrections.jsonl"


def _questions_path(task: Task) -> Path:
    return task.dir / "questions.jsonl"


def _author(author: str | None) -> str:
    if author:
        return author
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "human")


# ---------------------------------------------------------------- snapshot
def _latest_events(task: Task, n: int) -> tuple[str | None, list[dict]]:
    runs = sorted((task.dir / "dsh" / "runs").glob("*.ndjson"))
    if not runs:
        return None, []
    keep = {"tool_call", "tool_result", "text", "final", "error"}
    events = [_trunc(e) for e in read_jsonl(runs[-1]) if e.get("type") in keep]
    return str(runs[-1].relative_to(task.dir)), events[-n:]


def _git_status(task: Task) -> list[str]:
    paths = [str(task.method_dir.relative_to(ROOT)) if task.method_dir.is_relative_to(ROOT)
             else str(task.method_dir), "envs"]
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *paths],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [f"<git status failed: {e}>"]
    return r.stdout.splitlines()[:200] if r.returncode == 0 else [f"<git status failed: {r.stderr.strip()[:300]}>"]


def _latest_smoke(task: Task) -> dict | None:
    files = sorted((task.dir / "runs").glob("smoke*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return {"file": str(files[-1].relative_to(task.dir)),
                **_trunc(json.loads(files[-1].read_text()), 2000)}
    except (json.JSONDecodeError, OSError) as e:
        return {"file": str(files[-1].relative_to(task.dir)), "error": str(e)}


def open_questions(task: Task) -> list[dict]:
    """Questions from `ask_human` that have not been answered yet."""
    return [q for q in read_jsonl(_questions_path(task)) if not q.get("answered")]


def snapshot(task: Task, n: int = 30) -> dict:
    """Context at the moment of a human intervention (stored with each correction)."""
    events_file, events = _latest_events(task, n)
    last_end = next((e for e in reversed(task.journal()) if e.get("event") == "agent/end"), None)
    return {
        "at": now(),
        "phase": task.state.get("phase"),
        "status": task.state.get("status"),
        "session_id": task.state.get("session_id"),
        "events_file": events_file,
        "events": events,
        "last_agent_end": _trunc(last_end, 3000) if last_end else None,
        "git_status": _git_status(task),
        "smoke": _latest_smoke(task),
        "open_questions": open_questions(task),
    }


# ------------------------------------------------------------- corrections
def _corrections(task: Task) -> list[dict]:
    return read_jsonl(_corrections_path(task))


def _add_correction(task: Task, kind: str, note: str, *, author: str | None,
                    target_phase: str | None = None, consumed: bool = False, **extra) -> dict:
    rows = _corrections(task)
    rec = {"id": f"c{len(rows) + 1:03d}", "at": now(), "author": _author(author), "kind": kind,
           "phase": task.state.get("phase"), "target_phase": target_phase, "note": note,
           **extra, "consumed": consumed, "snapshot": snapshot(task)}
    _append_jsonl(_corrections_path(task), rec)
    return rec


def check_eval(task: Task, target: Phase | None, kill_eval: bool) -> None:
    """Refuse to touch a task whose full evaluation is still running, unless the human
    rewinds (`target` given) and explicitly asks to kill it (`kill_eval`)."""
    from . import runner
    if task.phase is not Phase.EVALUATE or not runner.eval_alive(task):
        return
    pid = task.state["eval"]["pid"]
    if target is None or not kill_eval:
        raise HarnessError(f"task {task.id}: evaluation still running (pid {pid}); wait for it, or rewind "
                           f"with --phase <PHASE> --kill-eval to stop it")
    try:
        os.killpg(pid, signal.SIGTERM)    # launched with start_new_session: pid == process group id
    except ProcessLookupError:
        pass
    task.log("human/kill_eval", pid=pid, target_phase=target.value)


def correct(task: Task, note: str, *, phase: Phase | str | None = None,
            author: str | None = None, kill_eval: bool = False) -> dict:
    """Record a human correction, optionally rewind to `phase`, and set status READY."""
    if not note.strip():
        raise HarnessError("empty correction note")
    target = Phase(phase) if phase is not None else None
    check_eval(task, target, kill_eval)
    rec = _add_correction(task, "correct", note, author=author,
                          target_phase=target.value if target else None)
    task.log("human/correct", id=rec["id"], author=rec["author"], note=note,
             target_phase=rec["target_phase"])
    for k in ("pending_question", "review_pending"):
        task.state.pop(k, None)
    task.set(phase=target, status=Status.READY, reason=f"human correction {rec['id']}")
    return rec


def approve(task: Task, note: str = "", *, author: str | None = None, force: bool = False) -> dict:
    """Clear a review gate: phase -> next, status -> READY. A non-empty note reaches the agent.
    Without an open gate (`review_pending`) only `force=True` skips the phase (logged as forced)."""
    if not force and not task.state.get("review_pending"):
        raise HarnessError(f"task {task.id} has no open review gate ({task.phase.value}/{task.status.value}); "
                           f"use `harness approve --force` to skip the phase anyway")
    if task.status not in (Status.AWAITING_HUMAN, Status.FAILED, Status.READY):
        raise HarnessError(f"task {task.id} is {task.status.value}; nothing to approve")
    cur = task.phase
    rec = None
    if note.strip():
        rec = _add_correction(task, "approve", note, author=author, target_phase=cur.next.value)
    forced = force and not task.state.get("review_pending")
    task.log("human/force_approve" if forced else "human/approve", author=_author(author), note=note,
             approved_phase=cur.value, id=rec["id"] if rec else None,
             **({"status": task.status.value, "reason": task.state.get("reason")} if forced else {}))
    for k in ("review_pending", "pending_question"):
        task.state.pop(k, None)
    task.set(phase=cur.next, status=Status.READY, reason=f"{'force-' if forced else ''}approved {cur.value}")
    return {"approved": cur.value, "phase": task.phase.value, "forced": forced,
            "correction": rec["id"] if rec else None}


def pending_corrections(task: Task) -> list[dict]:
    """Corrections not yet delivered to the agent (oldest first)."""
    return [c for c in _corrections(task) if not c.get("consumed")]


def mark_consumed(task: Task, ids: list[str]) -> None:
    """Mark corrections as delivered to an agent run."""
    want = set(ids)
    if not want:
        return
    rows = _corrections(task)
    at = now()
    for r in rows:
        if r.get("id") in want and not r.get("consumed"):
            r["consumed"], r["consumed_at"] = True, at
    _write_jsonl(_corrections_path(task), rows)
    task.log("human/consumed", ids=sorted(want))


def answer(task: Task, question_id: str, text: str, *, author: str | None = None) -> dict:
    """Answer an `ask_human` question; the answer is stored as a correction (kind='answer')."""
    qs = read_jsonl(_questions_path(task))
    q = next((x for x in qs if x.get("id") == question_id), None)
    if q is None:
        raise HarnessError(f"no question {question_id!r} in {_questions_path(task)}")
    if q.get("answered"):
        raise HarnessError(f"question {question_id} already answered")
    note = f"Q ({question_id}): {q.get('question', '')}\nA: {text}"
    rec = _add_correction(task, "answer", note, author=author, question_id=question_id)
    q.update(answered=True, answered_at=rec["at"], answer=text, correction_id=rec["id"])
    _write_jsonl(_questions_path(task), qs)
    task.log("human/answer", id=rec["id"], question_id=question_id, answer=text)
    task.state.pop("pending_question", None)
    if task.status == Status.AWAITING_HUMAN and not open_questions(task):
        task.set(status=Status.READY, reason=f"answered {question_id}")
    else:
        task.save()
    return rec


def format_for_prompt(corrections: list[dict]) -> str:
    """Render pending corrections as the top block of the next agent prompt ('' if none)."""
    if not corrections:
        return ""
    lines = ["## HUMAN CORRECTIONS (highest priority)",
             "A human reviewed your work. These instructions override the paper, the contract "
             "and your previous decisions. Follow them, and record a note if one conflicts with "
             "the paper.", ""]
    for c in corrections:
        head = f"[{c['id']}] {c.get('kind', 'correct')} by {c.get('author', '?')} at {c.get('at', '')}"
        if c.get("phase"):
            head += f" (during {c['phase']})"
        if c.get("target_phase"):
            head += f" -> phase {c['target_phase']}"
        body = "\n".join("    " + l for l in str(c.get("note", "")).splitlines())
        lines += [f"- {head}", body]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- history
def history(task: Task, n: int = 50) -> str:
    """Human-readable timeline (last `n` entries) from the journal, enriched with corrections."""
    by_id = {c["id"]: c for c in _corrections(task)}
    skip = ("at", "event", "phase")
    lines = []
    for d in task.journal():
        ev = d.get("event", "?")
        if ev == "state/change":
            msg = f"{'/'.join(d.get('old', []))} -> {'/'.join(d.get('new', []))}"
            msg += f"  ({d['reason']})" if d.get("reason") else ""
        elif ev == "agent/start":
            msg = f"run {d.get('run')} [{d.get('label')}]"
        elif ev == "agent/end":
            msg = (f"run {d.get('run')} [{d.get('label')}] {d.get('reason')} "
                   f"steps={d.get('steps')} tools={d.get('tool_calls')}")
            msg += f" error={str(d['error'])[-160:]!r}" if d.get("error") else ""
        elif ev in ("human/approve", "human/force_approve"):
            msg = f"{d.get('approved_phase')} by {d.get('author')}" + (f": {d['note']}" if d.get("note") else "")
        elif ev in ("human/correct", "human/answer"):
            c = by_id.get(d.get("id"), d)
            flag = "consumed" if c.get("consumed") else "PENDING"
            tgt = f" -> {c['target_phase']}" if c.get("target_phase") else ""
            note = " | ".join(str(c.get("note", "")).splitlines())
            msg = f"[{d.get('id')} {flag}] {c.get('author')}{tgt}: {_trunc(note, 300)}"
        elif ev == "human/consumed":
            msg = f"delivered {', '.join(d.get('ids', []))} to the agent"
        else:
            msg = ", ".join(f"{k}={_trunc(v, 80)}" for k, v in d.items() if k not in skip)[:200]
        lines.append(f"{d.get('at', '')}  {d.get('phase') or '-':<9} {ev:<16} {msg}")
    return "\n".join(lines[-n:])


__all__ = ["snapshot", "check_eval", "correct", "approve", "pending_corrections", "mark_consumed",
           "answer", "history", "format_for_prompt", "open_questions"]
