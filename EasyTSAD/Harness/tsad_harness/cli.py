"""Command line interface: `scripts/harness <command>` (see README.md)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import HarnessError, ROOT, load
from .task import Phase, Status, Task


def _print_task(t: Task, verbose: bool = False) -> None:
    s = t.state
    print(f"{t.id:<24} {s['phase']:<10} {s['status']:<15} runs={t.counter('agent_runs')} "
          f"smoke={t.counter('smoke_runs')} session={s.get('session_id') or '-'}")
    if s.get("reason") and t.status is not Status.READY:
        print(f"  reason: {s['reason']}")
    if verbose:
        for h in s["history"][-10:]:
            print(f"  {h['at']}  {h['phase']:<10} {h['status']:<15} {h.get('reason', '')[:120]}")


def cmd_search(a) -> None:
    from . import papers
    for i, c in enumerate(papers.search(a.query, limit=a.limit, sources=tuple(a.sources.split(",")))):
        print(f"[{i}] {c.get('title')}  ({c.get('year') or '?'}, {c.get('venue') or c.get('source')})")
        print(f"    arxiv={c.get('arxiv_id') or '-'} url={c.get('url') or '-'}")
        if c.get("code_urls"):
            print(f"    code: {', '.join(c['code_urls'][:3])}")


def cmd_new(a) -> None:
    from . import papers, pipeline
    paper = papers.resolve(arxiv=a.arxiv, title=a.title, pdf=a.pdf, url=a.url)
    code = a.code or next(iter(paper.get("code_urls") or []), None)
    t = Task.create(a.name, paper=paper, code_url=code, env_hint=a.env, family=a.family)
    print(f"created task {t.id} at {t.dir}")
    if not a.no_run:
        pipeline.run(t, until=Phase.EVIDENCE)
        _print_task(t)


def cmd_run(a) -> None:
    from . import pipeline
    t = Task.open(a.task)
    until = Phase(a.until) if a.until else None
    pipeline.run(t, until=until, wait=a.wait, gpus=a.gpus)
    _print_task(t, verbose=True)


def cmd_status(a) -> None:
    tasks = [Task.open(a.task)] if a.task else Task.all()
    if not tasks:
        print(f"no tasks in {load().tasks_dir}")
    for t in tasks:
        if t.phase is Phase.EVALUATE and t.status is Status.RUNNING:
            from . import runner
            st = runner.eval_status(t)
            print(f"  eval: {json.dumps(st, ensure_ascii=False)[:300]}")
        _print_task(t, verbose=bool(a.task))


def cmd_approve(a) -> None:
    from . import review
    t = Task.open(a.task)
    with t.lock():
        t.reload()
        review.approve(t, a.message or "", force=a.force)
    _print_task(t)


def cmd_correct(a) -> None:
    from . import review
    t = Task.open(a.task)
    phase = Phase(a.phase) if a.phase else None
    with t.lock():
        t.reload()
        rec = review.correct(t, a.message, phase=phase, author=a.author, kill_eval=a.kill_eval)
    print(f"recorded correction {rec['id']}; the next agent run will receive it")
    _print_task(t)
    if a.run:
        cmd_run(argparse.Namespace(task=a.task, until=None, wait=False, gpus=None))


def cmd_answer(a) -> None:
    from . import review
    t = Task.open(a.task)
    with t.lock():
        t.reload()
        review.answer(t, a.question, a.message)
    _print_task(t)


def cmd_log(a) -> None:
    from . import review
    print(review.history(Task.open(a.task), n=a.n))


def cmd_reset(a) -> None:
    from . import review
    t = Task.open(a.task)
    with t.lock():
        t.reload()
        review.check_eval(t, Phase(a.phase), a.kill_eval)
        for k in ("pending_question", "review_pending"):
            t.state.pop(k, None)
        t.set(phase=Phase(a.phase), status=Status.READY, reason=f"manual reset to {a.phase}")
    _print_task(t)


def cmd_guard(a) -> int:
    """Show, restore or accept changes to protected files (see guard.py)."""
    from . import guard
    t = Task.open(a.task)
    if a.restore or a.accept:
        if a.accept and not (a.message or "").strip():
            raise HarnessError("--accept needs -m NOTE explaining why the change is acceptable")
        with t.lock():
            t.reload()
            if a.restore:
                for line in guard.restore(t):
                    print(line)
            else:
                guard.accept(t, a.message)
            left = guard.check(t)
            if (not left and t.status is Status.AWAITING_HUMAN
                    and str(t.state.get("reason", "")).startswith("protected files changed")):
                t.set(status=Status.READY, reason=f"guard {'restored' if a.restore else 'accepted'}")
    left = guard.check(t)
    print("\n".join(left) if left else "file guard clean")
    _print_task(t)
    return 1 if left else 0


def cmd_report(a) -> None:
    from . import report
    t = Task.open(a.task)
    print(report.write_report(t))


def cmd_publish(a) -> None:
    from . import report
    print(json.dumps(report.publish(build=not a.no_build, deploy=a.deploy), indent=2, ensure_ascii=False))


def cmd_doctor(a) -> None:
    """Check binary, key, endpoint and that the composed DSH profile boots."""
    import httpx
    cfg = load()
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{'ok ' if good else 'FAIL'} {name:<22} {detail}")

    line("dsh binary", cfg.dsh_bin.exists(), str(cfg.dsh_bin))
    try:
        key = cfg.api_key()
        line("api key", True, f"from ${cfg['llm']['api_key_env']} or {cfg['llm'].get('api_key_file')}")
    except HarnessError as e:
        key = ""
        line("api key", False, str(e))
    if key:
        env = cfg.network_env()
        proxy = None if cfg["llm"].get("bypass_proxy") else env.get("https_proxy")
        try:
            r = httpx.get(cfg["llm"]["base_url"].rstrip("/") + "/models", timeout=30, proxy=proxy,
                          trust_env=False, headers={"Authorization": f"Bearer {key}"})
            ids = [m.get("id") for m in r.json().get("data", [])]
            line("endpoint /models", r.status_code == 200 and cfg["llm"]["model"] in ids,
                 f"HTTP {r.status_code}, model {cfg['llm']['model']} {'listed' if cfg['llm']['model'] in ids else 'NOT listed'}")
        except Exception as e:  # noqa: BLE001 - diagnostics only
            line("endpoint /models", False, repr(e)[:200])
    import tempfile
    from pathlib import Path
    from .agent import render_patch
    with tempfile.TemporaryDirectory() as d:
        fake = Task(Path(d) / "Doctor", {"method": "Doctor"}, {"phase": "EVIDENCE", "counters": {}})
        (fake.dir / "dsh").mkdir(parents=True)
        patch = render_patch(fake)
        env = cfg.agent_env() if key else cfg.network_env()
        env["DSH_HOME"] = str(cfg.dsh_home)
        p = subprocess.run([str(cfg.dsh_bin), "--profile", cfg["dsh"]["profile"], "--patch",
                            str(cfg.path("dsh", "trim_patch")), "--patch", str(patch), "--dump-config"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        line("profile composes", p.returncode == 0, (p.stderr.strip().splitlines() or [""])[-1][:200])
    raise SystemExit(0 if ok else 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="TSAD paper reproduction harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="search papers (arXiv + LLM web search)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--sources", default="arxiv,llm", help="comma list: arxiv,llm,github")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("new", help="create a task for one paper and fetch it")
    p.add_argument("name", help="EasyTSAD method / class name, e.g. AnomalyTransformer2")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--arxiv")
    src.add_argument("--title")
    src.add_argument("--pdf")
    src.add_argument("--url")
    p.add_argument("--code", help="official code repository URL")
    p.add_argument("--env", help="environment hint for the agent (e.g. torch2)")
    p.add_argument("--family", help="leaderboard family (深度学习 / 传统方法 / 预训练模型)")
    p.add_argument("--no-run", action="store_true", help="only create the task")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("run", help="advance a task until it needs a human or is done")
    p.add_argument("task")
    p.add_argument("--until", choices=[x.value for x in Phase])
    p.add_argument("--wait", action="store_true", help="wait for the full evaluation to finish")
    p.add_argument("--gpus", help="GPU ids for the evaluation, e.g. 0,1")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="show tasks")
    p.add_argument("task", nargs="?")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("approve", help="pass a review gate")
    p.add_argument("task")
    p.add_argument("-m", "--message")
    p.add_argument("--force", action="store_true",
                   help="skip the phase even without an open review gate (logged as forced)")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("correct", help="record a human correction (fed to the next agent run)")
    p.add_argument("task")
    p.add_argument("-m", "--message", required=True)
    p.add_argument("--phase", choices=[x.value for x in Phase], help="rewind to this phase")
    p.add_argument("--author")
    p.add_argument("--run", action="store_true", help="continue the task right away")
    p.add_argument("--kill-eval", action="store_true", help="with --phase: stop a running evaluation")
    p.set_defaults(fn=cmd_correct)

    p = sub.add_parser("answer", help="answer a question the agent asked")
    p.add_argument("task")
    p.add_argument("question", help="question id, e.g. q001")
    p.add_argument("-m", "--message", required=True)
    p.set_defaults(fn=cmd_answer)

    p = sub.add_parser("log", help="timeline of a task (journal + corrections)")
    p.add_argument("task")
    p.add_argument("-n", type=int, default=50)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("reset", help="force a task back to a phase")
    p.add_argument("task")
    p.add_argument("phase", choices=[x.value for x in Phase])
    p.add_argument("--kill-eval", action="store_true", help="stop a running evaluation first")
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("guard", help="check / restore / accept changes to protected files")
    p.add_argument("task")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="list violations (default)")
    g.add_argument("--restore", action="store_true", help="restore protected files from the baseline")
    g.add_argument("--accept", action="store_true", help="accept the changes as the new baseline (-m NOTE)")
    p.add_argument("-m", "--message")
    p.set_defaults(fn=cmd_guard)

    p = sub.add_parser("report", help="(re)write report.md for a task")
    p.add_argument("task")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("publish", help="export results, refresh leaderboard data, build the site")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--deploy", action="store_true", help="also run the deploy script")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("doctor", help="check installation, API key, endpoint and profile")
    p.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except HarnessError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
