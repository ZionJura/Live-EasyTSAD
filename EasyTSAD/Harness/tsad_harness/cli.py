"""Command line interface: `scripts/harness <command>` (see README.md)."""
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import sys

from .config import HARNESS_DIR, HarnessError, ROOT, load
from .task import Phase, Status, Task


def _print_task(t: Task, verbose: bool = False) -> None:
    s = t.state
    print(f"{t.id:<24} {s['phase']:<10} {s['status']:<15} runs={t.counter('agent_runs')} "
          f"session={s.get('session_id') or '-'}")
    if s.get("reason") and t.status is not Status.READY:
        print(f"  reason: {s['reason']}")
    if verbose:
        for h in s["history"][-10:]:
            print(f"  {h['at']}  {h['phase']:<10} {h['status']:<15} {h.get('reason', '')[:120]}")


def cmd_tui(a) -> None:
    import os
    entry = str(ROOT / "scripts" / "harness")
    arguments = [entry, "tui"]
    if getattr(a, "resume", None):
        arguments.extend(["--resume", a.resume])
    if a.task:
        arguments.append(a.task)
    if a.gpus is not None:
        arguments.extend(["--gpus", a.gpus])
    arguments.extend(["--refresh", str(a.refresh)])
    os.execv(entry, arguments)


def cmd_search(a) -> None:
    from . import papers
    for i, c in enumerate(papers.search(a.query, limit=a.limit, sources=tuple(a.sources.split(",")))):
        print(f"[{i}] {c.get('title')}  ({c.get('year') or '?'}, {c.get('venue') or c.get('source')})")
        print(f"    arxiv={c.get('arxiv_id') or '-'} url={c.get('url') or '-'}")
        if c.get("code_urls"):
            print(f"    code: {', '.join(c['code_urls'][:3])}")


def cmd_new(a) -> None:
    from .bridge import RemoteClient
    from .store import identity
    paper = {k: getattr(a, k) for k in ("arxiv", "title", "pdf", "url", "code", "env", "family") if getattr(a, k)}
    paper["name"] = a.name
    options = {}
    if getattr(a, "evaluation_spec", None):
        import tomllib
        with open(a.evaluation_spec, "rb") as source:
            options["evaluation"] = tomllib.load(source)
    result = RemoteClient().ensure().call("POST", "/batches", {"papers": [paper], "start": not a.no_run,
        "afterJobs": getattr(a, "after", None) or [], **options, "requestId": identity("cli")})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_run(a) -> None:
    from .bridge import RemoteClient
    import time
    client = RemoteClient().ensure()
    result = client.call("POST", f"/jobs/{a.task}/run", {"until": a.until, "gpus": a.gpus})
    print(json.dumps(result, ensure_ascii=False))
    if a.wait:
        while True:
            result = client.call("GET", f"/jobs/{a.task}")
            if result["status"] in ("done", "failed", "paused", "cancelled"):
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            time.sleep(2)


def cmd_status(a) -> None:
    from .bridge import RemoteClient
    import httpx
    try:
        path = "/jobs/" + a.task if a.task else "/jobs"
        print(json.dumps(RemoteClient().call("GET", path), ensure_ascii=False, indent=2))
        return
    except (httpx.ConnectError, HarnessError):
        pass
    tasks = [Task.open(a.task)] if a.task else Task.all()
    if not tasks:
        print(f"no tasks in {load().tasks_dir}")
    for task in tasks:
        _print_task(task, verbose=bool(a.task))


def cmd_approve(a) -> None:
    from . import review
    t = Task.open(a.task)
    with t.lock():
        t.reload()
        review.approve(t, a.message or "", force=a.force)
    _print_task(t)


def cmd_correct(a) -> None:
    from .bridge import RemoteClient
    result = RemoteClient().ensure().call("POST", f"/jobs/{a.task}/correct", {"message": a.message,
        "phase": a.phase, "author": a.author, "run": a.run})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_answer(a) -> None:
    from . import review
    t = Task.open(a.task)
    with t.lock():
        t.reload()
        review.answer(t, a.question, a.message)
    _print_task(t)


def cmd_log(a) -> None:
    from .bridge import RemoteClient
    import httpx
    try:
        print(RemoteClient().call("GET", f"/jobs/{a.task}/logs").get("text", ""))
    except (httpx.ConnectError, HarnessError):
        from . import review
        print(review.history(Task.open(a.task), n=a.n))


def cmd_reset(a) -> None:
    from .bridge import RemoteClient
    client = RemoteClient().ensure()
    print(json.dumps(client.call("POST", f"/jobs/{a.task}/correct", {"message": f"Reset to {a.phase}",
        "phase": a.phase, "run": True}), indent=2, ensure_ascii=False))


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
    print(json.dumps(report.results_view(t), ensure_ascii=False, indent=2))


def cmd_publish(a) -> None:
    from .bridge import RemoteClient
    line = "/publish" + (" --build" if not a.no_build else "") + (" --deploy" if a.deploy else "")
    print(json.dumps(RemoteClient().ensure().call("POST", "/commands", {"line": line}), indent=2, ensure_ascii=False))


def cmd_serve(a) -> None:
    from .service import serve
    serve(a.host, a.port)


def cmd_chat(a) -> None:
    from .bridge import RemoteClient
    client = RemoteClient(a.url).ensure()
    sid = a.session or client.call("POST", "/sessions", {}).get("sessionId")
    if not a.message:
        print(sid)
        return
    from .store import identity
    print(json.dumps(client.call("POST", f"/sessions/{sid}/messages", {"text": a.message,
        "mode": "queue", "requestId": identity("cli")}), ensure_ascii=False))


def cmd_job_control(a) -> None:
    from .bridge import RemoteClient
    options = {"afterJobs": a.after} if getattr(a, "after", None) is not None else {}
    print(json.dumps(RemoteClient().ensure().call("POST", f"/jobs/{a.task}/{a.cmd}", options), ensure_ascii=False))


def cmd_doctor(a) -> None:
    """Check binary, key, endpoint and that the composed DSH profile boots."""
    import httpx
    cfg = load()
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{'ok ' if good else 'FAIL'} {name:<22} {detail}")

    host = HARNESS_DIR / "dsh-host"
    entry = host / "launch.mjs"
    line("pinned DSH host", (host / "node_modules/@deepseek-ai/dsh/package.json").exists(), str(entry))
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
    env = cfg.agent_env() if key else cfg.network_env()
    env.update(TSAD_MODEL_PROVIDER=cfg["llm"]["provider"], TSAD_MODEL=cfg["llm"]["model"],
               TSAD_MODEL_BASE_URL=cfg["llm"]["base_url"], TSAD_MODEL_API_KEY_ENV=cfg.dsh_key_env,
               TSAD_MODEL_API="openai-responses" if cfg["llm"]["wire_api"] == "responses" else "openai-completions")
    try:
        result = subprocess.run([shutil.which("node") or "node", str(entry), "--check"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        line("profile composes", result.returncode == 0, (result.stderr.strip().splitlines() or [""])[-1][:200])
    except (OSError, subprocess.TimeoutExpired) as error:
        line("profile composes", False, str(error))
    raise SystemExit(0 if ok else 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="TSAD paper reproduction harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="persistent terminal control service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("chat", help="send a normal message to a persistent Agent")
    p.add_argument("message", nargs="?")
    p.add_argument("--session")
    p.add_argument("--url")
    p.set_defaults(fn=cmd_chat)

    for action in ("pause", "resume", "stop", "retry"):
        p = sub.add_parser(action, help=f"{action} a domain job")
        p.add_argument("task")
        if action == "resume":
            p.add_argument("--after", action="append", help="wait for a prerequisite job; repeat as needed")
        p.set_defaults(fn=cmd_job_control)

    p = sub.add_parser("tui", help="Agent chat, native commands and live experiments")
    p.add_argument("--resume", metavar="SESSION_ID|latest", help="restore a session instead of creating an independent one")
    p.add_argument("task", nargs="?", help="initially selected task")
    p.add_argument("--gpus", help="default GPUs for runs: auto, cpu, or 0,1")
    p.add_argument("--refresh", type=float, default=2, help="refresh interval in seconds (default: 2)")
    p.set_defaults(fn=cmd_tui)

    p = sub.add_parser("search", help="search papers (arXiv + LLM web search)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--sources", default="arxiv,github", help="comma list: arxiv,github (legacy optional: llm)")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("new", help="create a task and run integration, evaluation and reporting")
    p.add_argument("name", help="EasyTSAD method / class name, e.g. AnomalyTransformer2")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--arxiv")
    src.add_argument("--title")
    src.add_argument("--pdf")
    src.add_argument("--url")
    p.add_argument("--code", help="official code repository URL")
    p.add_argument("--after", action="append", help="wait until this job finishes; repeat for multiple prerequisites")
    p.add_argument("--evaluation-spec", help="EasyTSAD experiment TOML overriding this task's evaluation defaults")
    p.add_argument("--env", help="environment hint for the agent (e.g. torch2)")
    p.add_argument("--family", help="leaderboard family (深度学习 / 传统方法 / 预训练模型)")
    p.add_argument("--no-run", action="store_true", help="only create the task")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("run", help="run integration, automatic repairs, evaluation and reporting")
    p.add_argument("task")
    p.add_argument("--until", choices=[x.value for x in Phase])
    waiting = p.add_mutually_exclusive_group()
    waiting.add_argument("--wait", dest="wait", action="store_true", help="wait through evaluation (default)")
    waiting.add_argument("--detach", dest="wait", action="store_false", help="return after launching evaluation")
    p.set_defaults(wait=True)
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
    p.add_argument("--new-session", action="store_true",
                   help="start a fresh agent session, preserving task artifacts and earlier traces")
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("guard", help="check / restore / accept changes to protected files")
    p.add_argument("task")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="list violations (default)")
    g.add_argument("--restore", action="store_true", help="restore protected files from the baseline")
    g.add_argument("--accept", action="store_true", help="accept the changes as the new baseline (-m NOTE)")
    p.add_argument("-m", "--message")
    p.set_defaults(fn=cmd_guard)

    p = sub.add_parser("report", help="read native EasyTSAD results for a task")
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
