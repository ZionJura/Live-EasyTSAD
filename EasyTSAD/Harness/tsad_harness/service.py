"""Authenticated loopback control API and durable events for terminal clients."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import secrets
import shlex
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .bridge import Bridge
from .config import ROOT, HarnessError, load
from .scheduler import Scheduler, process_identity
from .store import Store, identity
from .task import Task

API = "/api/harness/v1"
COMMANDS = [
    ("reproduce", "<paper IDs, titles, URLs or PDF paths>", "Create and run a batch of papers"),
    ("papers", "search <query>", "Search papers and official implementations"),
    ("tasks", "", "List jobs and current stages"), ("run", "<job or batch>", "Run an existing task"),
    ("pause", "<job>", "Pause a training job"), ("resume", "<job>", "Resume a paused job"),
    ("stop", "<job>", "Cancel a training job (distinct from stopping a chat turn)"),
    ("retry", "<job>", "Retry with a new attempt"), ("correct", "<job> <message>", "Record a correction"),
    ("ps", "[job]", "Show managed Agent, environment and evaluation processes"),
    ("agents", "", "Show the Host and CLI Agent tree"), ("agent", "<session>", "Select a continuable Agent conversation"),
    ("logs", "<job>", "Read experiment logs"), ("scores", "<job>", "View live and final scores"),
    ("report", "<job>", "Read canonical avg/all references and metrics"), ("publish", "[--build] [--deploy]", "Export results; deploy only when explicit"),
    ("help", "", "Discover TSAD and native commands"),
]


def runtime_dir() -> Path:
    configured = load().raw.get("interactive", {}).get("runtime_dir", "runtime/harness-workbench")
    path = Path(os.environ.get("TSAD_HARNESS_RUNTIME", configured)).expanduser()
    return path if path.is_absolute() else ROOT / path


def access_token(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "access-token"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    except FileExistsError:
        pass
    os.chmod(path, 0o600)
    return path.read_text().strip()


def paper_inputs(text: str) -> list[str]:
    """Multiline titles remain whole; quoted/ID/URL lists can share a line."""
    import re
    lines = [line.strip() for line in text.replace(";", "\n").splitlines() if line.strip()]
    out = []
    for line in lines:
        values = shlex.split(line)
        if len(values) > 1 and all(re.match(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|https?://|/|\./|~/)", x) for x in values):
            out.extend(values)
        else:
            out.append(line.strip('"'))
    return out


def redact_worker_endpoints(value):
    """Hide per-turn MCP capability paths while retaining useful process URLs."""
    if isinstance(value, dict):
        return {key: redact_worker_endpoints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_worker_endpoints(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(/mcp/)[a-f0-9]{32}(?![a-f0-9])", r"\1[redacted]", value)
    return value


def create_app(*, directory: Path | None = None, bridge=None, start_runtime=True, port: int | None = None):
    directory = directory or runtime_dir()
    workspace_root = ROOT.resolve()
    token = access_token(directory)
    store = Store(directory / "state.sqlite3")
    port = int(port if port is not None else load().raw.get("interactive", {}).get("port", 8767))
    bridge = bridge or Bridge(directory, f"http://127.0.0.1:{port}", token)
    scheduler = Scheduler(store, bridge)
    mutation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifetime(app):
        lock = open(directory / "service.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise HarnessError("A workbench service already owns this state directory") from None
        startup = None
        recovery = None
        if start_runtime:
            async def recover():
                try:
                    def recover_benchmark():
                        from LiveLeaderBoard.pipeline.export import recover_pending
                        return recover_pending()
                    await asyncio.to_thread(recover_benchmark)
                except Exception as error:
                    store.event("benchmark.recovery_failed", error=str(error))
            async def initialize():
                await bridge.start()
                await scheduler.start()
            recovery = asyncio.create_task(recover())
            startup = asyncio.create_task(initialize())
        try:
            yield
        finally:
            if startup and not startup.done():
                startup.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await startup
            if recovery:
                await recovery
            await scheduler.close()
            await bridge.close()
            store.close()
            lock.close()

    app = FastAPI(title="TSAD Harness Control API", version="1.0", lifespan=lifetime,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.scheduler, app.state.bridge, app.state.token = store, scheduler, bridge, token

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path
        protected = path.startswith(API) or path.startswith("/internal/")
        if protected and path != API + "/health":
            authorization = request.headers.get("authorization", "")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            if not secrets.compare_digest(supplied, token):
                return JSONResponse({"error": "Connect with the local access token"}, status_code=401)
            origin = request.headers.get("origin")
            if origin and request.method != "GET":
                from urllib.parse import urlparse
                parsed = urlparse(origin)
                if parsed.hostname not in ("localhost", "127.0.0.1", request.url.hostname):
                    return JSONResponse({"error": "Origin is not allowed"}, status_code=403)
        try:
            return await call_next(request)
        except HarnessError as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(HarnessError)
    async def harness_error(request, error):
        return JSONResponse({"error": str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def bad_request(request, error):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.get(API + "/health")
    async def health():
        return {"ok": True, "version": 1, "runtime": "@deepseek-ai/dsh@0.2.2-rc.1",
                "connected": not bool(bridge.error), "runtimeError": bridge.error,
                "eventCursor": store.cursor(), "maxPapers": scheduler.max_papers}

    @app.get(API + "/sessions")
    async def sessions():
        result = await native_collection("/sessions", "sessions")
        rows = {row.get("sessionId") or row.get("id"): row for row in result.get("sessions", [])}
        for binding in store.all("bindings"):
            if is_cli(binding):
                row = cli_agent(binding)
                rows[row["sessionId"]] = row
        return redact_worker_endpoints({**result, "sessions": list(rows.values())})

    @app.post(API + "/sessions")
    async def new_session(request: Request):
        body = await request.json()
        result = await bridge.call("POST", "/sessions", {**body, "cwd": str(ROOT)})
        sid = result.get("sessionId") or result["id"]
        scheduler.bind(sid, "coordinator")
        store.event("session.created", sessionId=sid)
        return result

    @app.get(API + "/sessions/{session_id}")
    async def session(session_id: str, beforeSeq: int | None = None, maxMessages: int | None = None):
        snapshot = await asyncio.to_thread(scheduler.worker_snapshot, session_id)
        if snapshot is not None:
            return redact_worker_endpoints({**snapshot, **cli_agent(snapshot)})
        from urllib.parse import urlencode
        query = {key: value for key, value in {"beforeSeq": beforeSeq, "maxMessages": maxMessages}.items() if value is not None}
        return await bridge.call("GET", "/sessions/" + session_id + ("?" + urlencode(query) if query else ""), timeout=8)

    @app.get(API + "/sessions/{session_id}/events")
    async def session_events(session_id: str, after: int = 0, epoch: str = ""):
        snapshot = await asyncio.to_thread(scheduler.worker_snapshot, session_id)
        if snapshot is not None:
            events = snapshot.get("events", [])
            event_epoch = snapshot.get("eventEpoch") or session_id
            events_start = int(snapshot.get("eventsStart", 0))
            cursor = max(0, after - events_start) if epoch == event_epoch else 0
            return redact_worker_endpoints({"events": events[cursor:], "cursor": snapshot.get("eventCursor", events_start + len(events)),
                    "epoch": event_epoch})
        from urllib.parse import urlencode
        return await bridge.call("GET", f"/sessions/{session_id}/events?" + urlencode({"after": max(0, after), "epoch": epoch}))

    @app.post(API + "/sessions/{session_id}/messages")
    async def message(session_id: str, request: Request):
        body = await request.json()
        if not isinstance(body.get("text"), str) or not body["text"].strip():
            raise HarnessError("Message text is required")
        if body["text"].lstrip().startswith("/"):
            return redact_worker_endpoints(await command({"sessionId": session_id, "line": body["text"], "requestId": body.get("requestId")}))
        if body.get("mode", "queue") not in ("queue", "steer"):
            raise HarnessError("Message mode must be queue or steer")
        request_id = body.get("requestId") or identity("msg")
        if is_cli(store.get("bindings", session_id) or {}):
            result = await scheduler.worker_message(session_id, body["text"], request_id)
        else:
            result = await bridge.call("POST", f"/sessions/{session_id}/messages", {**body, "requestId": request_id})
        store.event("message.accepted", sessionId=session_id, text=body["text"])
        return result

    @app.post(API + "/sessions/{session_id}/cancel")
    async def cancel(session_id: str):
        if is_cli(store.get("bindings", session_id) or {}):
            result = await scheduler.worker_cancel(session_id)
        else:
            result = await bridge.call("POST", f"/sessions/{session_id}/cancel", {})
        store.event("session.interrupted", sessionId=session_id)
        return result

    def is_cli(binding):
        return binding.get("transport") == "cli" or binding.get("kind") == "cli"

    async def native_collection(path, key):
        try:
            return await bridge.call("GET", path)
        except (HarnessError, httpx.HTTPError) as error:
            return {key: [], "runtimeError": str(error)}

    def cli_agent(binding):
        job = store.get("jobs", binding.get("jobId", "")) or {}
        return {"sessionId": binding["sessionId"], "name": binding.get("title") or binding.get("name") or
                f"{job.get('method', binding.get('jobId', 'Worker'))}: {binding.get('role', 'worker')}",
                "role": binding.get("role"), "jobId": binding.get("jobId"), "transport": "cli",
                "parentSessionId": binding.get("parentSessionId") or job.get("leadSessionId") or job.get("sessionId"),
                "running": binding.get("status") == "running", "status": binding.get("status", "idle"),
                "continuable": True, "mode": "continuable", "phase": job.get("stage"),
                "binding": {**binding, "kind": "cli"}}

    @app.get(API + "/agents")
    async def agents(sessionId: str = ""):
        selected = next((b for b in store.all("bindings") if b.get("sessionId") == sessionId), {})
        result = ({"agents": [], **({"runtimeError": bridge.error} if bridge.error else {})}
                  if is_cli(selected) else await native_collection("/agents?sessionId=" + sessionId, "agents"))
        bindings = {b["sessionId"]: b for b in store.all("bindings")}
        rows = {}
        for agent in result.get("agents", []):
            sid = agent.get("sessionId") or agent.get("id")
            if sid in bindings:
                agent.update(binding=bindings[sid])
            rows[sid] = agent
        for binding in bindings.values():
            if is_cli(binding):
                row = cli_agent(binding)
                rows[row["sessionId"]] = row
        return redact_worker_endpoints({**result, "agents": list(rows.values())})

    @app.get(API + "/commands")
    async def commands(sessionId: str = ""):
        native = []
        if not bridge.error and not is_cli(store.get("bindings", sessionId) or {}):
            with contextlib.suppress(HarnessError):
                native = (await bridge.call("GET", "/commands?sessionId=" + sessionId)).get("commands", [])
        own = [{"name": name, "command": "/" + name, "usage": usage, "description": description, "source": "tsad"} for name, usage, description in COMMANDS]
        names = {n[0] for n in COMMANDS}
        return {"commands": own + [n for n in native if n.get("name", "").lstrip("/") not in names]}

    def get_job(job_id):
        job = store.get("jobs", job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    def scores(job_id, *, job=None):
        job = get_job(job_id) if job is None else job
        metrics = [m for m in store.all("metrics") if m["jobId"] == job_id]
        recorded = {(m.get("schema"), m.get("dataset"), m.get("runId")) for m in metrics}
        for result in job.get("datasetResults", {}).values():
            key = (result.get("schema"), result.get("dataset"), result.get("run_id"))
            if result.get("status") == "success" and key not in recorded:
                metrics.append({"jobId": job_id, "dataset": key[1], "schema": key[0],
                    "dataset_type": result.get("dataset_type"), "runId": key[2], "provisional": False,
                    "data": {"summary": result.get("metrics_summary", {})}, "manifest": result.get("manifest")})
        return {"metrics": metrics}

    def workspace_file(value, *, base=ROOT):
        if not value or not isinstance(value, (str, Path)):
            return None
        path = Path(value)
        path = (path if path.is_absolute() else base / path).resolve()
        return path if path.is_relative_to(workspace_root) else None

    def relative_file(value, *, base=ROOT):
        path = workspace_file(value, base=base)
        return str(path.relative_to(workspace_root)) if path else None

    def redact(value, secret_values=None):
        if secret_values is None:
            values = [token] + [item for key, item in os.environ.items() if len(item) >= 6 and
                      re.search(r"(?:api[_-]?key|token|secret|password)", key, re.I)]
            with contextlib.suppress(HarnessError):
                values.append(load().api_key())
            secret_values = sorted(set(values), key=len, reverse=True)
        # Process argv, scripts and logs can carry credentials from launch-time env.
        if isinstance(value, dict):
            return {key: "[redacted]" if re.search(r"(?:api[_-]?key|token|secret|password|authorization)", key, re.I)
                    else redact(item, secret_values) for key, item in value.items()}
        if isinstance(value, list):
            result, hide_next = [], False
            for item in value:
                result.append("[redacted]" if hide_next else redact(item, secret_values))
                hide_next = isinstance(item, str) and bool(re.fullmatch(r"--?(?:api[-_]key|token|secret|password|authorization)", item, re.I))
            return result
        if not isinstance(value, str):
            return value
        value = redact_worker_endpoints(value)
        for secret in secret_values:
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)(--(?:api[-_]key|token|secret|password|authorization)\s+)(?:\"[^\"]*\"|'[^']*'|[^\s;]+)", r"\1[redacted]", value)
        value = re.sub(r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1[redacted]@", value)
        value = re.sub(r"(?i)((?:[\w-]*(?:api[_-]?key|token|secret|password))[\w-]*\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s;]+)", r"\1[redacted]", value)
        return value

    def processes(job_id="", *, job=None, task=None):
        selected_jobs = [job] if job is not None else [get_job(job_id)] if job_id else store.all("jobs")
        result = []
        for job in selected_jobs:
            records = []
            for role, record in (job.get("workerProcesses") or {}).items():
                records.append((role, record, ROOT))
                for owned in record.get("ownedProcesses") or []:
                    if isinstance(owned, dict):
                        owned_role = f"{owned.get('role') or role}:{owned.get('kind') or 'owned'}"
                        records.append((owned_role, {**owned, "parentPid": record.get("pid")}, ROOT))
            if job.get("evaluation"):
                current_task = task or (Task.open(job["taskId"]) if job.get("taskId") else None)
                records.append(("evaluation", job["evaluation"], current_task.dir if current_task else ROOT))
            for role, original, base in records:
                record = dict(original)
                running = record.get("status") == "running" or role == "evaluation" and not record.get("status")
                if running and record.get("pid"):
                    actual = process_identity(record["pid"])
                    running = actual is not None and (not record.get("processIdentity") or record["processIdentity"] == actual)
                kind = "evaluation" if role == "evaluation" else "environment" if record.get("kind") == "environment" else "agent"
                row = {key: record[key] for key in ("pid", "parentPid", "processIdentity", "sessionId", "returncode", "finishedAt", "url") if key in record}
                row.update(id=record.get("id") or f"{job['id']}:{role}:{record.get('pid', '')}", jobId=job["id"], kind=kind,
                           role=role, processKind=record.get("kind") or kind, argv=record.get("argv") or record.get("command") or [],
                           cwd=relative_file(record.get("cwd") or "."), startedAt=record.get("startedAt") or record.get("started_at"),
                           status="running" if running else "finished", outcome=record.get("status"),
                           gpus=record.get("gpus", job.get("gpus")))
                for field in ("log", "stderr", "script"):
                    value = record.get(field) or (record.get("spec") if field == "script" else None)
                    if value:
                        row[field] = relative_file(value, base=base)
                if row.get("script"):
                    script = workspace_file(row["script"])
                    if script and script.is_file():
                        with script.open(errors="replace") as stream:
                            row["scriptContent"] = stream.read(100000)
                result.append(row)
        return redact({"processes": result}) if result else {"processes": []}

    @app.get(API + "/processes")
    def get_processes(jobId: str = ""):
        return processes(jobId)

    def logs(job_id):
        job = get_job(job_id)
        paths = []
        for process in processes(job_id)["processes"]:
            paths.extend(path for key in ("log", "stderr") if (path := workspace_file(process.get(key))))
        task = Task.open(job["taskId"]) if job.get("taskId") else None
        if task:
            paths.extend(sorted((task.dir / "runs").glob("*.out"))[-3:])
        paths = list(dict.fromkeys(path for path in paths if path.is_file()))[-12:]
        return redact({"text": "\n".join(f"{p.relative_to(ROOT)}\n{p.read_text(errors='replace')[-24000:]}" for p in paths),
                "journal": task.journal()[-80:] if task else [], "logs": [str(p.relative_to(ROOT)) for p in paths]})

    def job_report(job_id):
        from . import report
        job = get_job(job_id)
        return {"jobId": job_id, **report.results_view(Task.open(job["taskId"]))} if job.get("taskId") else {"jobId": job_id, "datasets": []}

    @app.get(API + "/jobs")
    async def jobs():
        return redact_worker_endpoints({"jobs": store.all("jobs"), "leases": store.leases()})

    @app.get(API + "/batches")
    async def batches():
        return {"batches": store.all("batches")}

    @app.post(API + "/batches")
    async def create_batch(request: Request):
        body = await request.json()
        return redact_worker_endpoints(scheduler.batches(body.get("papers", []), body.get("requestId") or identity("req"),
            session_id=body.get("sessionId"), start=body.get("start", True), **{k: body[k] for k in
                ("gpus", "until", "afterJobs", "sequential", "evaluation", "evaluationPlan") if k in body}))

    @app.get(API + "/jobs/{job_id}")
    def job_detail(job_id: str):
        job = get_job(job_id)
        task = None
        parameters = {}
        if job.get("taskId"):
            task = Task.open(job["taskId"])
            path = workspace_file(task.method_dir / "config.toml")
            if path and path.is_file() and path.stat().st_size <= 100000:
                with contextlib.suppress(OSError, ValueError):
                    parameters = {"parameters": tomllib.loads(path.read_text()), "parametersPath": str(path.relative_to(ROOT))}
        return {**redact_worker_endpoints({**job, **parameters, "diagnostics": [d for d in store.all("diagnostics") if d["jobId"] == job_id],
                "attempts": [a for a in store.all("attempts") if a["jobId"] == job_id],
                "agents": [b for b in store.all("bindings") if b.get("jobId") == job_id],
                **scores(job_id, job=job)}), **processes(job_id, job=job, task=task)}

    @app.post(API + "/jobs/{job_id}/{action}")
    async def job_action(job_id: str, action: str, request: Request):
        body = await request.json() if await request.body() else {}
        async with mutation_lock:
            return redact_worker_endpoints(await scheduler.control(job_id, action, body))

    @app.post(API + "/batches/{batch_id}/{action}")
    async def batch_action(batch_id: str, action: str, request: Request):
        batch = store.get("batches", batch_id)
        if not batch:
            raise HTTPException(404, "Unknown batch")
        body = await request.json() if await request.body() else {}
        async with mutation_lock:
            return redact_worker_endpoints({"jobs": [await scheduler.control(j, action, body) for j in batch["jobs"]]})

    @app.get(API + "/jobs/{job_id}/logs")
    def get_logs(job_id: str):
        return logs(job_id)

    @app.get(API + "/jobs/{job_id}/scores")
    def get_scores(job_id: str):
        return scores(job_id)

    @app.get(API + "/jobs/{job_id}/report")
    def get_report(job_id: str):
        return job_report(job_id)

    async def command(body):
        line, session_id = body.get("line", "").strip(), body.get("sessionId")
        name, _, raw = line.lstrip("/").partition(" ")
        name, raw = name.lower(), raw.strip()
        args = shlex.split(raw)
        request_id = body.get("requestId") or identity("command")
        if name == "reproduce":
            return scheduler.batches(paper_inputs(raw), request_id, session_id=session_id,
                                     **{key: body[key] for key in ("gpus", "until") if key in body})
        if name in ("help",):
            return await commands(session_id or "")
        if name in ("tasks",):
            return {"jobs": store.all("jobs"), "batches": store.all("batches")}
        if name == "papers" and raw.startswith("search "):
            return await tool("search", {"query": raw[7:]}, session_id)
        if name in ("agents", "agent"):
            if name == "agent" and args:
                state = await session(args[0])
                return {"sessionId": args[0], "session": state, "selected": True}
            return await agents(session_id or "")
        if name == "ps":
            return await asyncio.to_thread(processes, args[0] if args else "")
        if name in ("logs", "scores", "report") and args:
            return await asyncio.to_thread({"logs": logs, "scores": scores, "report": job_report}[name], args[0])
        if name in ("run", "pause", "resume", "stop", "retry", "correct") and args:
            async with mutation_lock:
                batch = store.get("batches", args[0])
                kwargs = {"message": " ".join(args[1:])} if name == "correct" else {}
                kwargs.update({key: body[key] for key in ("gpus", "until") if key in body})
                if batch:
                    return {"jobs": [await scheduler.control(j, name, kwargs) for j in batch["jobs"]]}
                return await scheduler.control(args[0], name, kwargs)
        if name == "publish":
            from . import report
            result = await asyncio.to_thread(report.publish, build="--build" in args or "--deploy" in args, deploy="--deploy" in args)
            store.event("benchmark.updated", result=result)
            return result
        if name in {c[0] for c in COMMANDS}:
            raise HarnessError("Usage: /" + name + " " + next(c[1] for c in COMMANDS if c[0] == name))
        available = await commands(session_id or "")
        native_names = {c.get("name", c.get("command", "")).lstrip("/") for c in available["commands"] if c.get("source") != "tsad"}
        if name not in native_names:
            raise HarnessError(f"Unknown command /{name}; use /help")
        return await bridge.call("POST", "/commands", {"sessionId": session_id, "line": line})

    @app.post(API + "/commands")
    async def execute_command(request: Request):
        return redact_worker_endpoints(await command(await request.json()))

    async def tool(name, args, session_id=None, role=None, job_id=None):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                raise HarnessError("Tool arguments must be a JSON object or a valid JSON object string") from None
        if not isinstance(args, dict):
            raise HarnessError("Tool arguments must be a JSON object or a valid JSON object string")
        if name == "diagnose" and args.get("operation") == "check_write_lease":
            job_id = job_id or args.get("jobId")
            job = store.get("jobs", job_id) if job_id else None
            leases = store.leases()
            # An evidenced diagnosis returns to IMPLEMENT before a repair can
            # write code. The diagnoser itself only writes its diagnosis artifact.
            granted = bool(job and job.get("taskId") and job.get("authorized") and job.get("stage") == "IMPLEMENT"
                and role == "implementer" and any(l["owner"] == job_id and l["resource"] == "method:" + job["method"] and l.get("mode") == "write" for l in leases))
            if not granted:
                return {"leaseGranted": False}
            task = Task.open(job["taskId"])
            return {"leaseGranted": True, "allowedMethodDir": relative_file(task.method_dir),
                    "allowedEnvironmentDir": relative_file(task.method_dir / "environment"),
                    "allowedTaskDir": relative_file(task.dir)}
        if role in ("researcher", "supervisor", "reporter") and name not in ("tasks", "scores", "search", "fetch", "status", "diagnose"):
            raise HarnessError("This role can observe and diagnose, but cannot control jobs")
        if args.get("input") is not None:
            return await command({"sessionId": session_id, "line": "/" + name + " " + args["input"], "requestId": args.get("requestId")})
        target = args.get("jobId") or job_id or args.get("taskId")
        if name == "reproduce":
            values = args.get("papers") or paper_inputs(args.get("text", ""))
            options = {key: args[key] for key in ("evaluation", "evaluationPlan", "afterJobs", "sequential") if key in args}
            return scheduler.batches(values, args.get("requestId") or identity("tool"), session_id=session_id, **options)
        if name == "tasks":
            return {"jobs": store.all("jobs"), "batches": store.all("batches")}
        if name in ("run", "pause", "resume", "stop", "retry", "correct"):
            async with mutation_lock:
                return await scheduler.control(target, name, args)
        if name == "scores":
            return scores(target)
        if name == "status":
            return get_job(target) if target else {"jobs": store.all("jobs"), "leases": store.leases()}
        if name == "diagnose":
            return scheduler.diagnose(target, args)
        if name == "complete":
            job = get_job(target)
            if job["status"] != "done" or not job.get("benchmarkRevision"):
                raise HarnessError("Goal waits for evaluation, report and committed dual export")
            return {"complete": True, "jobId": target}
        if name == "search":
            from . import papers
            result = await asyncio.to_thread(papers.search, args.get("query", ""), limit=int(args.get("limit", 8)), sources=tuple(args.get("sources", ["arxiv", "github"])))
            return {"papers": result}
        if name == "fetch":
            import httpx
            url = args.get("url", "")
            if not url.startswith(("https://", "http://")):
                raise HarnessError("HTTP(S) URL is required")
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                if "pdf" in response.headers.get("content-type", ""):
                    raise HarnessError("Use /reproduce with the PDF URL to download and extract it")
                return {"url": str(response.url), "text": response.text[:100000]}
        raise HarnessError("Unknown domain tool " + name)

    @app.post("/internal/tools/{name}")
    async def internal_tool(name: str, request: Request):
        body = await request.json()
        return redact_worker_endpoints(await tool(name, body.get("arguments", {}), body.get("sessionId"), body.get("role"), body.get("jobId")))

    @app.get(API + "/events")
    async def events(request: Request, after: int = 0):
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            cursor = after
        async def stream():
            nonlocal cursor
            yield "retry: 1000\n\n"
            while not await request.is_disconnected():
                records = store.events(cursor)
                if records:
                    for event in records:
                        cursor = event["id"]
                        yield f"id: {cursor}\nevent: {event['type']}\ndata: {json.dumps(redact_worker_endpoints(event), ensure_ascii=False)}\n\n"
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(.35 if records else 1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get(API + "/snapshot")
    async def snapshot():
        return redact_worker_endpoints({"eventCursor": store.cursor(), "batches": store.all("batches"), "jobs": store.all("jobs"),
                "bindings": store.all("bindings"), "leases": store.leases(), "metrics": store.all("metrics")})

    return app


def serve(host="127.0.0.1", port=None):
    import uvicorn
    port = port or int(load().raw.get("interactive", {}).get("port", 8767))
    print(f"Harness control API: http://{host}:{port}{API}\nConnect with: scripts/harness tui", flush=True)
    uvicorn.run(create_app(port=port), host=host, port=port, log_level="info",
                timeout_graceful_shutdown=5)
