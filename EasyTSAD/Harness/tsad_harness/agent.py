"""Drive the trimmed DeepSeek Harness (`dsh --profile headless`) for one agent run.

The harness renders a per-role patch with the LLM route and a scoped, per-turn
loopback HTTP MCP server, then launches `dsh` with `--json`. Session history lives
under the configured DSH home; prompts, events and process ledgers live under
`<method>/reproduction/agents/<role>/`. The outer launcher enforces step and wall
budgets and owns the lifetimes of both CLI and MCP server processes.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import shutil
import os
import signal
import socket
import time
import uuid
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import HARNESS_DIR, PROMPTS_DIR, ROOT, HarnessError, load
from .workers import ROLES, relative
from .task import Task, now
from .process_identity import process_identity as _process_identity


class WorkerStartupError(HarnessError):
    pass


class WorkerTransportError(HarnessError):
    pass


class WorkerOutputLimitError(WorkerTransportError):
    pass


@dataclass
class RunResult:
    session_id: str | None
    reason: str                    # completed | error | aborted | max-tokens | budget-steps | budget-time | crash
    final_text: str = ""
    steps: int = 0
    tool_calls: int = 0
    usage: dict = field(default_factory=dict)
    error: str = ""
    events_path: Path | None = None
    reached_model: bool = False    # True once any step_start was seen
    saw_turn_end: bool = False

    @property
    def ok(self) -> bool:
        return self.reason == "completed"


def render_system_prompt(task: Task) -> str:
    text = (PROMPTS_DIR / "system.md").read_text()
    return text.format(root=ROOT, task_dir=task.dir, method=task.method,
                       method_dir=task.method_dir.relative_to(ROOT))


def _loopback_env(env: dict) -> dict:
    """Keep the private MCP connection direct regardless of proxy defaults."""
    exclusions = list(dict.fromkeys(host.strip() for key in ("NO_PROXY", "no_proxy")
                                   for host in env.get(key, "").split(",") if host.strip()))
    exclusions.extend(host for host in ("127.0.0.1", "localhost", "::1") if host not in exclusions)
    return {**env, "NO_PROXY": ",".join(exclusions), "no_proxy": ",".join(exclusions)}


def _tool_environment(task: Task, role: str, write_enabled: bool) -> dict:
    cfg = load()
    tool_env = {
        "PYTHONPATH": str(HARNESS_DIR),
        "TSAD_TASK_DIR": str(task.dir),
        "TSAD_ROOT": str(ROOT),
        "TSAD_WORKER_ROLE": role,
        "TSAD_WORKER_READ_ONLY": "0" if write_enabled else "1",
        "TSAD_HARNESS_CONFIG": str(Path(os.environ.get("TSAD_HARNESS_CONFIG", HARNESS_DIR / "config.toml"))),
        # mcp-client scrubs the ambient env; pass what the tools need explicitly
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if os.environ.get(var):
            tool_env[var] = os.environ[var]
    tool_env.update({k: v for k, v in cfg.network_env().items() if k in ("NO_PROXY", "no_proxy")})
    return _loopback_env(tool_env)


def worker_session_store(cfg, session_id: str | None = None) -> tuple[Path, str]:
    root, compression = cfg.dsh_home / "worker-sessions", "none"
    if session_id:
        for parent in (cfg.dsh_home / "worker-sessions", cfg.dsh_home / "worker-sessions-zstd",
                       cfg.dsh_home / "sessions"):
            matches = list(parent.glob(f"*/{session_id}"))
            if not matches:
                continue
            source = matches[0]
            compression = "zstd" if any(source.glob("*.jsonl.zstd")) else "none"
            root = cfg.dsh_home / ("worker-sessions-zstd" if compression == "zstd" else "worker-sessions")
            destination = root / source.relative_to(parent)
            if source != destination and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
            break
    return root, compression


def render_patch(task: Task, role: str = "implementer", *, write_enabled: bool = True,
                 mcp_url: str | None = None, session_id: str | None = None,
                 max_output_tokens: int | None = None) -> Path:
    """Write the method's per-role runtime patch, applied after trim.patch.yml."""
    cfg = load()
    llm = cfg["llm"]
    model = {"id": llm["model"], "name": llm["model"], "input": ["text"],
             "contextWindow": int(llm["context_window"]),
             "maxTokens": int(max_output_tokens or llm["max_output_tokens"])}
    if llm.get("reasoning_effort"):
        model["reasoningEfforts"] = {e: e for e in ("low", "medium", "high")}
    default_model = {"provider": llm["provider"], "model": llm["model"]}
    if llm.get("reasoning_effort"):
        default_model["reasoningEffort"] = llm["reasoning_effort"]
    tool_env = _tool_environment(task, role, write_enabled)
    session_root, compression = worker_session_store(cfg, session_id)
    transport = ({"transport": "streamable-http", "url": mcp_url} if mcp_url else {
        "transport": "stdio", "command": str(cfg.python_bin),
        "args": ["-m", "tsad_harness.worker_tools"], "env": tool_env, "cwd": str(ROOT)})
    patch = [
        {"id": "llm-pi-ai", "config": {"providers": {llm["provider"]: {
            "displayName": llm.get("display_name", llm["provider"]),
            "apiKeyEnv": cfg.dsh_key_env,
            "api": {"responses": "openai-responses", "chat": "openai-completions"}[llm["wire_api"]],
            "baseURL": llm["base_url"],
            "models": [model],
            # Proxies surface dropped HTTP/2 streams as PI_AI_ERROR, which DSH does not retry by default.
            "retryPolicy": {"mode": "normal", "maxRetries": int(cfg["llm"].get("max_retries", 6)),
                            "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TIMEOUT",
                                               "TRANSPORT", "PI_AI_ERROR"]},
        }}}},
        {"id": "agent-default-model", "config": default_model},
        {"id": "persona", "config": {"prefix": f"You are the TSAD {role} worker. {ROLES[role]} Use only scoped MCP tools."}},
        {"id": "session-persistence-jsonl",
         "config": {"root": str(session_root), "compression": compression}},
        {"insert": [{"id": "tsad-tools", "name": "@deepseek-ai/dsh-mcp-client", "config": {
            "serverName": "tsad", **transport,
            "toolCallTimeoutMs": int(cfg["dsh"]["mcp_tool_timeout_ms"]),
            "failOnStartupError": True,
        }}]},
    ]
    # Disable the shipped generic mutation and shell tools. The MCP server is
    # the sole writer and checks the role's exact artifact allowlist.
    patch.extend({"id": name, "disabled": True} for name in
                 ("tool-fs", "tool-fs-search", "tool-bash", "tool-jobs", "tool-job", "tool-apply-patch", "tool-terminal"))
    out = task.method_dir / "reproduction" / "agents" / role / "runtime.patch.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# generated by tsad_harness.agent.render_patch; do not edit\n"
                   + yaml.safe_dump(patch, sort_keys=False, allow_unicode=True))
    return out


def _write_json(path: Path, data: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _permit_owner(path: Path, pid: int, lease_id: str | None = None):
    if lease_id is None:
        try:
            lease_id = json.loads((path / "owner.json").read_text()).get("leaseId")
        except (OSError, ValueError):
            pass
    _write_json(path / "owner.json", {"pid": pid, "identity": _process_identity(pid),
                "startTime": time.time(), "leaseId": lease_id or uuid.uuid4().hex})


def _stale_permit(path: Path) -> bool:
    try:
        owner = json.loads((path / "owner.json").read_text())
        actual = _process_identity(owner.get("pid", 0))
        return actual is None or actual != owner.get("identity")
    except (OSError, ValueError):
        try:
            return time.time() - path.stat().st_mtime > 30
        except OSError:
            return False


def _discard_permit(path: Path):
    tombstone = path.with_name("stale-" + uuid.uuid4().hex)
    try:
        path.rename(tombstone)
    except FileNotFoundError:
        return
    shutil.rmtree(tombstone, ignore_errors=True)


@contextlib.contextmanager
def _permit_lock(slot: Path):
    lock = slot.with_name(slot.name + ".reclaim")
    acquired = False
    try:
        try:
            lock.mkdir()
        except FileExistsError:
            if _stale_permit(lock):
                _discard_permit(lock)
            yield False
            return
        acquired = True
        _permit_owner(lock, os.getpid())
        yield True
    finally:
        if acquired:
            _discard_permit(lock)


@contextlib.contextmanager
def model_gate(cancel=None):
    """Directory permits shared with the native host, including stale PID checks."""
    cfg = load()
    interactive = cfg.raw.get("interactive", {})
    runtime = Path(os.environ.get("TSAD_HARNESS_RUNTIME") or interactive.get("runtime_dir", "runtime/harness-workbench"))
    if not runtime.is_absolute():
        runtime = ROOT / runtime
    directory = runtime / "model-permits"
    directory.mkdir(parents=True, exist_ok=True)
    chosen = None
    lease_id = uuid.uuid4().hex
    while chosen is None:
        if cancel is not None and cancel.is_set():
            raise HarnessError("Worker cancelled while waiting for a model permit")
        for number in range(int(interactive.get("max_model_requests", 4))):
            slot = directory / str(number)
            with _permit_lock(slot) as locked:
                if not locked:
                    continue
                if slot.exists():
                    if _stale_permit(slot):
                        _discard_permit(slot)
                    else:
                        continue
                slot.mkdir()
                _permit_owner(slot, os.getpid(), lease_id)
                chosen = slot
                break
        if chosen is None:
            if cancel is not None:
                cancel.wait(.1)
            else:
                time.sleep(.1)
    try:
        yield chosen
    finally:
        while True:
            with _permit_lock(chosen) as locked:
                if locked:
                    try:
                        owner = json.loads((chosen / "owner.json").read_text())
                    except (OSError, ValueError):
                        break
                    if owner.get("leaseId") == lease_id:
                        _discard_permit(chosen)
                    break
            time.sleep(.05)


def run_agent(task: Task, prompt: str, *, label: str, role: str = "implementer",
              session_id: str | None = None, cancel=None, on_process=None,
              on_event=None, stage_token: str | None = None, write_enabled: bool = True,
              max_output_tokens: int | None = None, fresh_session: bool = False) -> RunResult:
    with model_gate(cancel) as permit:
        return _run_agent(task, prompt, label=label, role=role, session_id=session_id, cancel=cancel,
                          on_process=on_process, on_event=on_event, stage_token=stage_token,
                          write_enabled=write_enabled, permit=permit,
                          max_output_tokens=max_output_tokens, fresh_session=fresh_session)


def _run_agent(task: Task, prompt: str, *, label: str, role: str = "implementer",
              session_id: str | None = None, cancel=None, on_process=None,
              on_event=None, stage_token: str | None = None, write_enabled: bool = True, permit=None,
              max_output_tokens: int | None = None, fresh_session: bool = False) -> RunResult:
    cfg = load()
    deadline = time.monotonic() + 60 * float(cfg["budget"]["max_wall_minutes_per_run"])
    ledger_path = task.method_dir / "reproduction/agents" / role / "process.json"
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text())
        if previous.get("status") == "running":
            current = _process_identity(previous.get("pid", 0))
            if current and current == previous.get("processIdentity"):
                raise HarnessError(f"The {role} worker is already running; refusing duplicate launch")
    owned = None
    try:
        with _worker_mcp(cfg, task, role, write_enabled=write_enabled, cancel=cancel,
                         on_process=on_process, deadline=deadline, stage_token=stage_token) as owned:
            return _run_agent_cli(task, prompt, label=label, role=role, session_id=session_id,
                                  cancel=cancel, on_process=on_process, on_event=on_event,
                                  stage_token=stage_token, write_enabled=write_enabled,
                                  permit=permit, deadline=deadline, mcp_process=owned,
                                  max_output_tokens=max_output_tokens, fresh_session=fresh_session)
    except HarnessError as error:
        if owned is None:
            raise WorkerStartupError(str(error)) from error
        raise
    finally:
        if owned is not None and ledger_path.exists():
            process = json.loads(ledger_path.read_text())
            process["ownedProcesses"] = [dict(owned)]
            if process.get("kind") == "worker-startup":
                process.update(status="failed", finishedAt=now(), reason="error")
            _write_json(ledger_path, process)
            if on_process:
                on_process(dict(process))


def _run_agent_cli(task: Task, prompt: str, *, label: str, role: str,
                   session_id: str | None, cancel, on_process, on_event,
                   stage_token: str | None, write_enabled: bool, permit,
                   deadline: float, mcp_process: dict, max_output_tokens: int | None = None,
                   fresh_session: bool = False) -> RunResult:
    """Run the exact host-pinned published CLI, with a durable per-role ledger.

    Role files are independent so the three research workers never mutate
    state.json concurrently. The scheduler persists callbacks before model input.
    """
    cfg = load()
    budget = cfg["budget"]
    session_path = task.method_dir / "reproduction/agents" / role / "session.json"
    if fresh_session:
        session_id = None
    elif not session_id and session_path.exists():
        session_id = json.loads(session_path.read_text()).get("sessionId")
    patch = render_patch(task, role, write_enabled=write_enabled, mcp_url=mcp_process["url"],
                         session_id=session_id, max_output_tokens=max_output_tokens)
    run_id = uuid.uuid4().hex
    directory = patch.parent
    events_path = directory / f"{run_id}-{label}.ndjson"
    (directory / f"{run_id}-{label}.prompt.md").write_text(prompt)
    ledger_path = directory / "process.json"
    node = HARNESS_DIR / "dsh-host/node_modules/node/bin/node"
    command = [str(node) if node.exists() else "node", str(HARNESS_DIR / "dsh-host/launch.mjs"),
               "--worker", "--profile", cfg["dsh"]["profile"],
               "--patch", str(cfg.path("dsh", "trim_patch")), "--patch", str(patch), "--json"]
    if session_id:
        command += ["--session-id", session_id]
    command.append("-")
    cfg.dsh_home.mkdir(parents=True, exist_ok=True)
    result = RunResult(session_id=session_id, reason="crash", events_path=events_path)
    stderr_path = events_path.with_suffix(".stderr")
    recorded_argv = [Path(arg).relative_to(ROOT).as_posix()
                     if Path(arg).is_absolute() and Path(arg).is_relative_to(ROOT) else arg
                     for arg in command]
    recorded_log, recorded_stderr = relative(events_path), relative(stderr_path)
    child_env = _loopback_env(cfg.agent_env())
    timed_out = threading.Event()
    finished = threading.Event()
    stop = None
    with open(events_path, "w") as events, open(stderr_path, "w") as errors:
        proc = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=errors, text=True, start_new_session=True)
        process = {"pid": proc.pid, "processIdentity": _process_identity(proc.pid),
                   "kind": "worker", "role": role, "status": "running", "runId": run_id,
                   "sessionId": session_id, "stageToken": stage_token,
                   "maxOutputTokens": int(max_output_tokens or cfg.raw.get("llm", {}).get("max_output_tokens", 8192)),
                   "freshSession": fresh_session,
                   "ownedProcesses": [dict(mcp_process)],
                   "argv": recorded_argv,
                   "cwd": ".", "log": recorded_log, "stderr": recorded_stderr,
                   "startedAt": now()}

        def watch():
            while not finished.wait(.2):
                if cancel is not None and cancel.is_set():
                    _terminate(proc)
                    return
                if time.monotonic() >= deadline:
                    timed_out.set()
                    _terminate(proc)
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        try:
            if permit is not None:
                _permit_owner(permit, proc.pid)
            _write_json(ledger_path, process)
            watcher.start()
        except BaseException:
            finished.set()
            _terminate(proc)
            proc.wait()
            _close_pipes(proc)
            raise
        try:
            if on_process:
                on_process(dict(process))
            if cancel is not None and cancel.is_set():
                stop = "aborted"
                _terminate(proc)
            else:
                proc.stdin.write(prompt)
                proc.stdin.close()
                for line in proc.stdout:
                    events.write(line)
                    events.flush()
                    _consume(result, line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if result.session_id and result.session_id != process.get("sessionId"):
                        process["sessionId"] = result.session_id
                        _write_json(session_path, {"sessionId": result.session_id})
                        _write_json(ledger_path, process)
                        if on_process:
                            on_process(dict(process))
                    if on_event:
                        on_event(event)
                    if result.steps >= int(budget["max_steps_per_run"]):
                        stop = "budget-steps"
                        _terminate(proc)
                        break
        except KeyboardInterrupt:
            stop = "aborted"
            _terminate(proc)
        except (BrokenPipeError, OSError):
            pass
        except BaseException:
            _terminate(proc)
            raise
        finally:
            finished.set()
            proc.wait()
            watcher.join(timeout=2)
            _close_pipes(proc)
            errors.close()
            if timed_out.is_set():
                stop = "budget-time"
            elif cancel is not None and cancel.is_set():
                stop = "aborted"
            if stop:
                result.reason = stop
            elif not result.saw_turn_end or (proc.returncode != 0 and result.reason == "completed"):
                result.reason = "crash"
            elif _tools_failed(stderr_path):
                result.reason = "error"
                result.error = "Scoped MCP tools did not initialize; refusing worker completion.\n" + stderr_path.read_text()[-3000:]
            if not result.ok and not result.error:
                result.error = f"dsh exit code {proc.returncode}\n" + stderr_path.read_text()[-4000:]
            process.update(status="completed" if result.ok else "failed", finishedAt=now(),
                           sessionId=result.session_id, reason=result.reason, error=result.error[-3000:],
                           exitCode=proc.returncode, reachedModel=result.reached_model,
                           steps=result.steps, toolCalls=result.tool_calls, usage=result.usage,
                           finalText=result.final_text[-12000:])
            _write_json(ledger_path, process)
            if on_process:
                on_process(dict(process))
    task.log("agent/end", role=role, run=run_id, reason=result.reason, steps=result.steps,
             tool_calls=result.tool_calls, usage=result.usage, events=relative(events_path))
    return result


@contextlib.contextmanager
def _worker_mcp(cfg, task: Task, role: str, *, write_enabled: bool, cancel=None,
                on_process=None, deadline: float, stage_token: str | None = None):
    """Start the scoped server before the pinned client's 60s initialize window.

    A dedicated loopback HTTP server belongs to exactly one CLI turn. Waiting
    for its MCP endpoint keeps cold Python imports outside the MCP handshake.
    """
    directory = task.method_dir / "reproduction/agents" / role
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "mcp-server.log"
    ledger_path = directory / "process.json"
    mcp_ledger = directory / "mcp-process.json"
    if mcp_ledger.exists():
        previous = json.loads(mcp_ledger.read_text())
        if previous.get("status") == "running":
            current = _process_identity(previous.get("pid", 0))
            if current and current == previous.get("processIdentity"):
                raise HarnessError(f"The {role} scoped MCP server is already running; refusing duplicate launch")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    endpoint = "/mcp/" + uuid.uuid4().hex
    command = [str(cfg.python_bin), "-m", "tsad_harness.worker_tools", "--http-port", str(port)]
    argv = [Path(arg).relative_to(ROOT).as_posix()
            if Path(arg).is_absolute() and Path(arg).is_relative_to(ROOT) else arg for arg in command]
    log = relative(log_path)
    script = relative(HARNESS_DIR / "tsad_harness/worker_tools.py")
    tool_env = {**_tool_environment(task, role, write_enabled), "TSAD_WORKER_MCP_HTTP_PATH": endpoint}
    timeout = min(deadline, time.monotonic() + float(cfg["dsh"].get("mcp_startup_timeout_ms", 300000)) / 1000)
    error = None
    with open(log_path, "w") as output:
        proc = subprocess.Popen(command, cwd=ROOT, env=tool_env,
                                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        process = {"pid": proc.pid, "processIdentity": _process_identity(proc.pid), "kind": "worker-tools",
                   "role": role, "status": "running", "runId": uuid.uuid4().hex,
                   "stageToken": stage_token, "argv": argv, "cwd": ".", "log": log, "script": script,
                   "url": f"http://127.0.0.1:{port}{endpoint}", "transport": "streamable-http",
                   "startedAt": now()}
        startup = {**process, "kind": "worker-startup", "ownedProcesses": [dict(process)]}
        try:
            _write_json(mcp_ledger, process)
            _write_json(ledger_path, startup)
            if on_process:
                on_process(dict(startup))
            while True:
                if cancel is not None and cancel.is_set():
                    raise HarnessError("Scoped MCP initialization cancelled")
                if time.monotonic() >= timeout:
                    raise HarnessError("Scoped MCP initialization exceeded its startup or run budget")
                if proc.poll() is not None:
                    raise HarnessError("Scoped MCP server exited before readiness:\n" + log_path.read_text()[-3000:])
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=.3)
                try:
                    connection.request("GET", endpoint, headers={"Accept": "application/json, text/event-stream"})
                    if connection.getresponse().status in (200, 400, 405, 406):
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                if cancel is not None:
                    cancel.wait(.1)
                else:
                    time.sleep(.1)
            yield process
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            _terminate(proc)
            proc.wait()
            output.close()
            process.update(status="failed" if error else "stopped", error=error,
                           finishedAt=now(), reason="error" if error else "completed")
            _write_json(mcp_ledger, process)
            current = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            if current.get("kind") == "worker-startup":
                current.update(status="failed", finishedAt=now(), reason="error", error=error,
                               ownedProcesses=[dict(process)])
                _write_json(ledger_path, current)
                if on_process:
                    on_process(dict(current))


def _tools_failed(stderr_path: Path) -> bool:
    """True when DSH reports that the TSAD MCP entry did not activate."""
    text = stderr_path.read_text() if stderr_path.exists() else ""
    return "mcp-client(tsad): initial connection or tool synchronization failed" in text or (
        "tsad-tools" in text and ("did not activate" in text or "entry did not activate" in text))


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None and not pipe.closed:
            with contextlib.suppress(BrokenPipeError, OSError):
                pipe.close()


def _consume(res: RunResult, line: str) -> None:
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return
    t = e.get("type")
    if t == "session":
        res.session_id = e.get("sessionId")
    elif t == "status" and e.get("phase") == "step_start":
        res.steps += 1
        res.reached_model = True
    elif t == "status" and e.get("phase") == "step_end" and e.get("usage"):
        for k, v in e["usage"].items():
            if isinstance(v, (int, float)):
                res.usage[k] = res.usage.get(k, 0) + v
    elif t == "status" and e.get("phase") == "turn_end":
        r = e.get("reason") or {}
        res.saw_turn_end = True
        res.reason = r.get("kind", "error")
        if r.get("error"):
            res.error = json.dumps(r["error"], ensure_ascii=False)[:4000]
    elif t == "tool_call":
        res.tool_calls += 1
    elif t == "final":
        res.final_text = e.get("text", "")
    elif t == "error":
        res.error = e.get("message", "")


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


__all__ = ["RunResult", "run_agent", "render_patch", "now"]
