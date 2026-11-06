"""Deterministic phase driver: decides what the agent is asked and when a phase ends.

    FETCH -> EVIDENCE -> [review gate] -> IMPLEMENT -> SMOKE -> EVALUATE -> REPORT -> DONE

Agent phases run one DSH turn, then the harness checks the file guard and the
phase validator. Failed checks are fed back as repair rounds (bounded by
`[budget].max_repair_rounds`); exhausted budgets stop the task for a human.
"""
from __future__ import annotations

import contextlib
import json
import time
import traceback

from . import guard, papers, report, review, runner, validate
from .agent import run_agent
from .config import PROMPTS_DIR, ROOT, load
from .task import AGENT_PHASES, Phase, Status, Task


def _prompt(name: str, **kw) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text().format(**kw)


def phase_prompt(task: Task, phase: Phase) -> str:
    cfg = load()
    paper = task.spec.get("paper", {})
    meta_path = task.dir / "paper" / "meta.json"
    full = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {**paper, **(full.get("paper") or {})}
    code = full.get("code") or {}
    common = dict(method=task.method, task_dir=task.dir,
                  method_dir=task.method_dir.relative_to(ROOT))
    if phase is Phase.EVIDENCE:
        code_info = (f"{code.get('url')} cloned at {code.get('dir')} (commit {code.get('commit')})"
                     if code.get("dir") else "none found; rely on the paper")
        return _prompt("evidence", **common, title=meta.get("title") or paper.get("title", "?"),
                       paper_ref=meta.get("arxiv_id") or meta.get("url") or paper.get("arxiv_id", ""),
                       code_info=code_info,
                       contract_example=json.dumps(validate.CONTRACT_EXAMPLE, indent=2, ensure_ascii=False))
    if phase is Phase.IMPLEMENT:
        return _prompt("implement", **common)
    if phase is Phase.SMOKE:
        s = cfg["smoke"]
        return _prompt("smoke", **common,
                       smoke_desc=f"{s['schema']} schema, {s['dataset']} curves {', '.join(s['curves'])}, "
                                  f"<= {s['max_epochs']} epochs",
                       smoke_left=max(0, int(cfg['budget']['max_smoke_runs']) - task.counter("smoke_runs")))
    raise ValueError(f"{phase} is not an agent phase")


def _with_corrections(task: Task, prompt: str) -> tuple[str, list[str]]:
    pending = review.pending_corrections(task)
    if not pending:
        return prompt, []
    return review.format_for_prompt(pending) + "\n\n" + prompt, [c["id"] for c in pending]


VALIDATORS = {Phase.EVIDENCE: validate.contract, Phase.IMPLEMENT: validate.implementation,
              Phase.SMOKE: validate.smoke}


def _guard_stop(task: Task, violations: list[str]) -> None:
    task.log("guard/violation", paths=violations)
    task.set(status=Status.AWAITING_HUMAN,
             reason="protected files changed\n" + guard.restore_hint(violations[:20], task.id))


def _agent_phase(task: Task, phase: Phase) -> None:
    cfg = load()
    max_rounds = int(cfg["budget"]["max_repair_rounds"])
    rnd = 0
    transient_left = int(cfg["budget"].get("max_transient_resumes", 3))
    guard.baseline(task)               # idempotent: the first agent phase snapshots, later calls keep it
    violations = guard.check(task)
    if violations:                     # unresolved earlier violation: never run on a tampered tree
        return _guard_stop(task, violations)
    prompt = phase_prompt(task, phase)
    while True:
        prompt, consumed = _with_corrections(task, prompt)
        task.state.pop("pending_question", None)   # the flag must reflect only this round's questions
        task.set(status=Status.RUNNING)
        res = run_agent(task, prompt, label=f"{phase.value.lower()}-r{rnd}")
        if consumed and res.reached_model:
            review.mark_consumed(task, consumed)
        task.reload()
        if task.phase is not phase or task.status is not Status.RUNNING:
            task.log("pipeline/superseded", expected=phase.value,
                     found=[task.state["phase"], task.state["status"]])
            return
        violations = guard.check(task)
        if violations:
            return _guard_stop(task, violations)
        if task.state.get("pending_question"):
            task.set(status=Status.AWAITING_HUMAN, reason="agent asked a question (see `harness log`)")
            return
        if not res.ok:
            if _transient(res) and transient_left > 0:
                # Upstream hiccup after DSH's own retries: continue the same session once more.
                transient_left -= 1
                task.log("pipeline/transient_retry", reason=res.reason, error=res.error[:500])
                prompt = _prompt("resume", phase=phase.value)
                continue
            task.set(status=Status.FAILED, reason=f"agent run ended with {res.reason}: {res.error[:500]}")
            return
        errors = VALIDATORS[phase](task)
        task.log("validate", target=phase.value, round=rnd, errors=errors)
        if not errors:
            break
        rnd += 1
        if rnd > max_rounds:
            task.set(status=Status.FAILED, reason=f"{phase.value} exit check still failing after "
                                                  f"{max_rounds} repair rounds: {errors[:5]}")
            return
        prompt = _prompt("repair", phase=phase.value, round=rnd, max_rounds=max_rounds,
                         errors="\n".join(f"- {e}" for e in errors))
    if phase.value in cfg["human"].get("review_after", []):
        # phase stays; `harness approve` moves it forward, `harness correct` re-runs it
        task.set(status=Status.AWAITING_HUMAN, review_pending=True,
                 reason=f"review gate after {phase.value}: `harness approve {task.id}` or `harness correct`")
    else:
        task.set(phase=phase.next, status=Status.READY)


TRANSIENT_MARKERS = ("PI_AI_ERROR", "stream failed", "TRANSPORT", "TIMEOUT", "RATE_LIMIT", "SERVER",
                     "ECONNRESET", "socket hang up")


def _transient(res) -> bool:
    """Model-side failure worth resuming (not a budget stop, tool outage or agent crash)."""
    return res.reason in ("error", "crash") and any(m in res.error for m in TRANSIENT_MARKERS)


def step(task: Task, *, gpus: str | None = None) -> None:
    """Advance the task by (at most) one phase."""
    phase = task.phase
    if task.status in (Status.AWAITING_HUMAN, Status.FAILED, Status.DONE):
        return
    if phase is Phase.FETCH:
        meta = papers.fetch(task)
        if not (task.dir / "paper" / "paper.md").exists() or not (task.dir / "paper" / "paper.md").read_text().strip():
            task.set(status=Status.FAILED, reason="paper text could not be extracted")
            return
        task.set(phase=Phase.EVIDENCE, status=Status.READY,
                 paper_title=(meta.get("paper") or {}).get("title") or task.spec.get("paper", {}).get("title"))
    elif phase in AGENT_PHASES:
        _agent_phase(task, phase)
    elif phase is Phase.EVALUATE:
        if task.status is Status.READY:
            if runner.eval_alive(task):      # an earlier launch is still running: track it, never relaunch
                task.set(status=Status.RUNNING, reason="re-attached to the running evaluation")
                return
            info = runner.launch_eval(task, gpus or load()["eval"]["gpus"])
            task.set(status=Status.RUNNING, eval=info)
            return
        st = runner.eval_status(task)
        if st.get("running"):
            return
        errors = validate.evaluation(task)
        task.log("validate", target="EVALUATE", errors=errors)
        if errors:
            task.set(status=Status.FAILED, reason=f"evaluation failed: {errors[:5]}")
        else:
            task.set(phase=Phase.REPORT, status=Status.READY)
    elif phase is Phase.REPORT:
        report.write_catalog(task)
        bm = report.export()
        path = report.write_report(task)
        ids = {m["id"] for m in json.loads(bm.read_text()).get("methods", [])}
        if task.method not in ids:
            task.set(status=Status.FAILED, reason=f"{task.method} missing from benchmark.json after export")
            return
        task.set(phase=Phase.DONE, status=Status.DONE, report=str(path))


def run(task: Task, *, until: Phase | None = None, wait: bool = False, gpus: str | None = None,
        poll_s: int = 60) -> Task:
    """Step until DONE / a human is needed / `until` is reached (evaluation: poll when `wait`).

    Holds the task lock for the whole run, so CLI mutations from other shells fail fast."""
    with task.lock():
        task.reload()
        if task.status is Status.RUNNING and task.phase in AGENT_PHASES:
            # we hold the lock, so no other `harness run` is driving this agent phase
            task.set(status=Status.READY, reason="resumed after an interrupted agent run")
        while True:
            task.reload()
            if until is not None and list(Phase).index(task.phase) >= list(Phase).index(until):
                return task
            before = (task.phase, task.status)
            try:
                step(task, gpus=gpus)
            except Exception as e:  # noqa: BLE001 - never leave the task RUNNING without a reason
                with contextlib.suppress(Exception):
                    task.reload()
                task.log("pipeline/error", step=before[0].value, error=repr(e), traceback=traceback.format_exc())
                task.set(status=Status.FAILED, reason=f"harness error in {before[0].value}: {e!r}"[:800])
                return task
            if task.status in (Status.AWAITING_HUMAN, Status.FAILED, Status.DONE):
                return task
            if task.phase is Phase.EVALUATE and task.status is Status.RUNNING:
                if not wait:
                    return task
                time.sleep(poll_s)
            elif (task.phase, task.status) == before:
                return task
