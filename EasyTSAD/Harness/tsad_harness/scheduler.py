"""Code-owned jobs, resource leases and event-triggered supervision."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path

from . import agent, environments, papers, pipeline, report, runner, workers
from .config import ROOT, HarnessError, load
from .store import Store, identity
from .task import Phase, Status, Task, now
from .process_identity import process_identity


def process_tree(process):
    """Include independently launched tool servers owned by a CLI turn."""
    yield process
    for child in process.get("ownedProcesses") or []:
        yield from process_tree(child)


def worker_alive(process):
    expected = process.get("processIdentity")
    return bool(expected and process_identity(process.get("pid", 0)) == expected)


def safe_stop(info: dict):
    pid = info.get("pid")
    expected = info.get("processIdentity")
    current = process_identity(pid) if pid else None
    if not pid or not expected or (current is not None and current != expected):
        raise HarnessError("Process identity cannot be verified; the service will not signal an unrelated process")
    # A PID/starttime match alone is insufficient if legacy state was imported incorrectly.
    if current is not None:
        if sys.platform == "darwin":
            try:
                cmdline = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="],
                                         capture_output=True, text=True, timeout=2, check=True).stdout
            except (OSError, subprocess.SubprocessError) as error:
                raise HarnessError("Cannot verify the experiment command; refusing to signal it") from error
        else:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "tsad" not in cmdline or "exp" not in cmdline:
            raise HarnessError("Recorded process is not a TSAD experiment")
    elif not info.get("commandVerified"):
        raise HarnessError("Cannot verify the exited experiment launcher; leases remain held")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)


class Scheduler:
    def __init__(self, store: Store, bridge):
        self.store, self.bridge = store, bridge
        self.workers: dict[str, asyncio.Task] = {}
        self.shutdown = False
        self.loop_task = None
        self.chat_cursors: dict[str, int] = {}
        self.progress_offsets: dict[str, int] = {}
        self.max_papers = int(load().raw.get("interactive", {}).get("max_papers", 2))
        self.root_session = None
        self.control_locks = {}
        self.cli_cancels = {}
        self.chat_workers = set()
        self.session_locks = {}
        self.model_slots = asyncio.Semaphore(int(load().raw.get("interactive", {}).get("max_model_requests", 4)))
        self.environment_condition = asyncio.Condition()
        self.environment_active = 0
        self.environment_waiting = 0
        self.environment_ceiling = max(1, int(load().raw.get("interactive", {}).get("max_environment_installs", 2)))
        self.environment_policy = store.get("cursors", "environment-policy") or {
            "id": "environment-policy", "limit": 1, "successes": 0}
        self.environment_policy["limit"] = min(self.environment_ceiling, self.environment_policy["limit"])

    async def start(self):
        await asyncio.to_thread(self.import_legacy)
        historical_processes = self.store.all("processes")
        for job in self.store.all("jobs"):
            configurations = dict(job.get("workerConfig") or {})
            model = load().raw.get("llm", {}).get("model", "")
            for process in historical_processes:
                if process.get("jobId") != job["id"] or process.get("status") != "completed" or not process.get("maxOutputTokens"):
                    continue
                role = process.get("role")
                if not role or role.endswith(":chat"):
                    continue
                previous = configurations.get(role) or {}
                if process["maxOutputTokens"] > previous.get("maxOutputTokens", 0):
                    configurations[role] = {"maxOutputTokens": process["maxOutputTokens"], "model": model,
                                            "reason": "recovered_successful_handoff", "updatedAt": now()}
            if configurations != (job.get("workerConfig") or {}):
                job = self.store.update("jobs", job["id"], workerConfig=configurations)
            job = self.reconcile_repair_rounds(job)
            recoverable = job.get("legacy") or job.get("stage") in {"EVALUATE", "DIAGNOSE", "REPORT", "DONE"}
            if job.get("taskId") and not job.get("resultsRecovered") and recoverable and not job.get("unsupported"):
                job = await asyncio.to_thread(self.recover_results, job, Task.open(job["taskId"]))
            # The worker registers its ledger before receiving model input.
            # Recover it even when a crash interrupted the Store callback.
            directory = load().methods_dir / job["method"] / "reproduction/agents"
            processes = dict(job.get("workerProcesses") or {})
            for path in directory.glob("*/process.json"):
                with contextlib.suppress(OSError, ValueError):
                    process = json.loads(path.read_text())
                    role = process.get("role") or path.parent.name
                    prior = processes.get(role, {})
                    if process.get("startedAt", "") >= prior.get("startedAt", ""):
                        processes[role] = process
            for path in directory.glob("*/mcp-process.json"):
                with contextlib.suppress(OSError, ValueError):
                    process = json.loads(path.read_text())
                    if not worker_alive(process):
                        continue
                    role = process.get("role") or path.parent.name
                    prior = processes.get(role)
                    if prior and prior.get("kind") not in ("worker-tools", "worker-startup"):
                        owned = {p.get("pid"): p for p in prior.get("ownedProcesses") or []}
                        owned[process["pid"]] = process
                        processes[role] = {**prior, "ownedProcesses": list(owned.values())}
                    else:
                        processes[role] = process
            if processes != job.get("workerProcesses", {}):
                job = self.store.update("jobs", job["id"], workerProcesses=processes)
            info = job.get("evaluation")
            if not info and job.get("taskId") and (job.get("evaluationLaunching") or ("evaluationLaunching" not in job and job.get("stage") == "EVALUATE")):
                with contextlib.suppress(HarnessError, OSError, ValueError):
                    info = Task.open(job["taskId"]).state.get("eval")
                    if info:
                        job = self.store.update("jobs", job["id"], evaluation=info)
            if info:
                # Never relaunch an attempt merely because its launcher exited during restart.
                # Final manifests must be consumed by evaluate() before clearing evaluation.
                resources = job.get("resources") or self.evaluation_resources(job, info)
                if not self.store.acquire(resources, job["id"], recovered=True, mode="read"):
                    self.store.update("jobs", job["id"], status="paused", authorized=False,
                                      reason="Recovered resource lease conflicts; inspect running jobs")
                else:
                    self.store.update("jobs", job["id"], resources=resources)
            elif job.get("status") == "running" and job.get("authorized"):
                processes = job.get("workerProcesses") or {}
                live = [p for process in processes.values() for p in process_tree(process) if worker_alive(p)]
                if live:
                    # A crashed service cannot prove an old write turn stopped.
                    self.store.update("jobs", job["id"], status="paused", authorized=False,
                                      reason="Recovered an active CLI worker; stop or resume it after it exits")
                else:
                    self.store.update("jobs", job["id"], status="queued")
        self.loop_task = asyncio.create_task(self.loop())

    def evaluation_resources(self, job, info):
        environment = runner.method_env(job["method"]) or job.get("environment") or job["method"]
        return ["method:" + job["method"], "environment:" + environment] + [
            "gpu:" + gpu for gpu in str(info.get("gpus") or "cpu").split(",") if gpu]

    @staticmethod
    def native_running(state):
        return bool(state.get("running") or state.get("status") in ("running", "busy", "queued")
                    or state.get("goalRunning") or state.get("queuedMessages", 0))

    async def cancel_roles(self, job_id, timeout=30):
        for (owner, _), event in list(self.cli_cancels.items()):
            if owner == job_id:
                event.set()
        current_job = self.store.get("jobs", job_id) or {}
        processes = (item for owner in (current_job.get("workerProcesses") or {}).values()
                     for item in process_tree(owner))
        for process in processes:
            if not worker_alive(process):
                continue
            pid, expected = process.get("pid"), process.get("processIdentity")
            actual = process_identity(pid) if pid else None
            if actual is not None and actual != expected:
                raise HarnessError("Worker PID was reused; refusing to signal it")
            if actual is not None:
                if not expected:
                    raise HarnessError("Cannot verify recovered CLI worker process")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + timeout
                while process_identity(pid) == expected:
                    if time.monotonic() > deadline:
                        raise HarnessError("CLI worker has not exited; its write lease remains held")
                    await asyncio.sleep(.1)
        bindings = [b for b in self.store.all("bindings") if b.get("jobId") == job_id and b.get("transport") != "cli"]
        for binding in bindings:
            sid = binding["sessionId"]
            await self.bridge.call("POST", f"/sessions/{sid}/cancel", {})
            deadline = time.monotonic() + timeout
            while self.native_running(await self.bridge.call("GET", f"/sessions/{sid}")):
                if time.monotonic() >= deadline:
                    raise HarnessError("Native Agent has not become idle; write leases remain held")
                await asyncio.sleep(.1)

    async def thread(self, function, *args, **kwargs):
        # Cancellation must not leave an untracked thread mutating a method directory.
        operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            await operation
            raise

    def import_legacy(self):
        for path in sorted(load().tasks_dir.glob("*/state.json")):
            if self.store.get("jobs", path.parent.name):
                continue
            try:
                t = Task.open(path.parent.name)
                active = runner.eval_alive(t)
                info = t.state.get("eval") or {}
                # Legacy processes are observed, never automatically killed/restarted on import.
                self.store.put("jobs", {"id": t.id, "method": t.method, "paper": t.spec.get("paper"),
                    "taskId": t.id, "stage": t.phase.value,
                    "status": "running" if active else "done" if t.status is Status.DONE else "paused",
                    "authorized": False, "legacy": True, "createdAt": t.spec.get("created_at", now()),
                    "evaluation": {**info, "processIdentity": process_identity(info["pid"])} if active else None,
                    "resources": ["method:" + t.method, "environment:" + (runner.method_env(t.method) or t.method)]
                        + ["gpu:" + g for g in str(info.get("gpus", "")).split(",") if g] if active else [],
                    "diagnostics": [], "datasetResults": {}, "repairRounds": 0})
            except Exception as error:
                self.store.event("legacy.import_failed", taskId=path.parent.name, error=str(error))

    @staticmethod
    def result_key(plan, dataset, plans):
        return dataset if len(plans) == 1 else f"{plan.get('dataset_type', 'UTS')}/{plan['schema']}/{dataset}"

    def recover_results(self, job, task):
        results = dict(job.get("datasetResults") or {})
        plans = job.get("evaluationPlan") or pipeline.evaluation_plan(task)
        recorded = list(results.values()) + list((job.get("evaluationStatus") or {}).get("datasets", {}).values())
        for plan in plans:
            for dataset in plan["datasets"]:
                paths = [ROOT / value["manifest"] for value in recorded if value.get("manifest")]
                if job.get("legacy"):
                    directory = runner.find_run(task.method, plan["schema"], dataset)
                    if directory:
                        paths.append(directory / "manifest.json")
                for path in paths:
                    try:
                        manifest = json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                    if manifest.get("status") != "success" or any(manifest.get(name) != value for name, value in (
                            ("method", task.method), ("schema", plan["schema"]), ("dataset", dataset),
                            ("dataset_type", plan.get("dataset_type", "UTS")))):
                        continue
                    key = self.result_key(plan, dataset, plans)
                    results[key] = {"status": "success", "manifest": workers.relative(path),
                        "run_dir": workers.relative(path.parent), "run_id": manifest.get("run_id"),
                        "config_hash": manifest.get("config_hash"), "schema": plan["schema"],
                        "configIdentity": runner.evaluation_identity(task, plan, dataset),
                        "dataset_type": plan.get("dataset_type", "UTS"), "dataset": dataset,
                        "metrics_summary": runner.read_summary(path.parent)}
        return self.store.update("jobs", job["id"], datasetResults=results, evaluationPlan=plans, resultsRecovered=True)

    def batches(self, papers_input: list, request_id: str, session_id=None, start=True, **options):
        if not papers_input or len(papers_input) > 100:
            raise HarnessError("Provide 1–100 papers")
        args = {"papers": papers_input, "sessionId": session_id, "start": start, **options}
        def create():
            batch_id = identity("batch")
            batch = {"id": batch_id, "papers": papers_input, "status": "queued" if start else "paused",
                     "sessionId": session_id, "createdAt": now(), "jobs": [], "sequential": options.get("sequential", False)}
            for value in papers_input:
                paper = value if isinstance(value, dict) else {"input": str(value).strip()}
                if not any(paper.get(k) for k in ("input", "arxiv", "title", "pdf", "url", "taskId")):
                    raise HarnessError("Each paper needs input, arxiv, title, pdf, url or taskId")
                name = paper.get("name") or paper.get("taskId") or ("Paper" + hashlib.sha256(
                    json.dumps(paper, sort_keys=True).encode()).hexdigest()[:12])
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
                    raise HarnessError("Method names must be valid EasyTSAD class names")
                if any(j.get("method") == name and j["status"] not in ("done", "cancelled", "failed")
                       for j in self.store.all("jobs")):
                    raise HarnessError(f"An active job already owns method {name}")
                job_id = name if not self.store.get("jobs", name) else identity(name)
                job = {"id": job_id, "batchId": batch_id, "method": name, "paper": paper,
                       "stage": "FETCH", "status": "queued" if start else "paused", "authorized": start,
                       "createdAt": now(), "sessionId": session_id, "repairRounds": 0,
                       "datasetResults": {}, "diagnostics": [], "evaluationLaunching": False, **options,
                       "environment": "tsad_" + name.lower()[:36] + "_" + hashlib.sha256(job_id.encode()).hexdigest()[:8]}
                if paper.get("taskId"):
                    t = Task.open(paper["taskId"])
                    job.update(taskId=t.id, stage=t.phase.value)
                if options.get("evaluation"):
                    job["evaluationPlan"] = [{**load()["eval"], **options["evaluation"]}]
                if options.get("sequential") and batch["jobs"]:
                    job["afterJobs"] = [batch["jobs"][-1]]
                self.store.put("jobs", job)
                batch["jobs"].append(job_id)
                self.store.event("job.created", jobId=job_id, batchId=batch_id, job=job)
            self.store.put("batches", batch)
            self.store.event("batch.created", batchId=batch_id, batch=batch)
            return batch
        return self.store.once(request_id, "batch.create", args, create)

    async def control(self, job_id, action, body):
        async with self.control_locks.setdefault(job_id, asyncio.Lock()):
            return await self._control(job_id, action, body)

    async def _control(self, job_id, action, body):
        job = self.store.get("jobs", job_id)
        if job is None:
            raise HarnessError("Unknown job " + job_id)
        if action in ("run", "resume", "retry"):
            if action == "resume" and job.get("status") == "failed" and job.get("runtimeRetryKey"):
                job.update(runtimeRetries=0, runtimeRetryKey=None)
            if "afterJobs" in body:
                job["afterJobs"] = body["afterJobs"]
            if job.get("evaluation"):
                # Persisted attempts always resume observation; even an exited process may
                # have completed valid manifests not yet consumed by the scheduler.
                job["stage"] = "EVALUATE"
            elif action == "retry" and job["status"] in ("done", "cancelled", "failed"):
                job.update(datasetResults={}, stage="IMPLEMENT", repairRounds=0, attemptId=None, resultsRecovered=True)
            job.update(status="queued", authorized=True, reason="", until=body.get("until"),
                       gpus=body.get("gpus", job.get("gpus")), nextRunAt=0)
        elif action in ("pause", "stop"):
            # Authoritative intent is committed before awaiting any asynchronous stop.
            job = self.store.update("jobs", job_id,
                status="paused" if action == "pause" else "cancelled", authorized=False)
            worker = self.workers.get(job_id)
            if worker and worker is not asyncio.current_task() and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            job = self.store.get("jobs", job_id)
            try:
                await self.cancel_roles(job_id)
                info = job.get("evaluation")
                if info:
                    if self.group_alive(info):
                        safe_stop(info)
                        await self.wait_exit(info, timeout=30)
                    job.update(evaluation=None, resources=[])
                    if job.get("taskId"):
                        Task.open(job["taskId"]).set(eval=None)
                self.store.release(job_id)
            except BaseException as error:
                self.store.update("jobs", job_id, reason=str(error)[:2000])
                raise
        elif action == "correct":
            note = body.get("message") or body.get("note") or body.get("text")
            if not note:
                raise HarnessError("Correction message is required")
            diagnostic = self.diagnose(job_id, {"category": "human_correction", "message": note,
                                               "action": "report_only", "evidence": []})
            if job.get("taskId"):
                task = Task.open(job["taskId"])
                with open(task.dir / "corrections.jsonl", "a") as out:
                    out.write(json.dumps({"id": diagnostic["id"], "at": now(), "note": note,
                        "target_phase": "IMPLEMENT", "author": body.get("author", "user")}, ensure_ascii=False) + "\n")
            if body.get("run") or body.get("phase"):
                await self._control(job_id, "pause", {})
                job = self.store.get("jobs", job_id)
                job.update(stage=body.get("phase") or "IMPLEMENT", status="queued", authorized=True,
                           evaluation=None, attemptId=None)
        else:
            raise HarnessError("Unknown job action " + action)
        result = self.store.put("jobs", job)
        self.store.event("job." + action, jobId=job_id, job=result)
        return result

    def diagnose(self, job_id, body):
        job = self.store.get("jobs", job_id)
        if not job:
            raise HarnessError("Unknown job")
        evidence = body.get("evidence") or []
        category = body.get("category", "observation")
        permitted = category in ("dimension", "alignment", "preprocessing", "interface", "execution", "oom", "environment")
        action = "repair" if permitted and workers.repair_evidence(evidence) and body.get("action") == "repair" else "report_only"
        result = self.store.put("diagnostics", {**body, "id": identity("diag"), "jobId": job_id,
            "attemptId": job.get("attemptId"), "createdAt": now(), "category": category,
            "evidence": evidence, "action": action})
        if job.get("taskId"):
            task = Task.open(job["taskId"])
            path = task.method_dir / "reproduction/diagnostics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.store.event("diagnostic.created", jobId=job_id, diagnostic=result)
        return result

    @staticmethod
    def group_alive(info):
        pid, expected = info.get("pid"), info.get("processIdentity")
        if not pid or not expected:
            raise HarnessError("Cannot verify experiment process identity; leases remain held")
        current = process_identity(pid)
        if current is not None and current != expected:
            raise HarnessError("Experiment PID was reused; leases remain held for inspection")
        # Check the entire original session/process group, not just the launcher.
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(") ", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == pid and int(fields[3]) == pid:
                    if int(fields[19]) < int(expected):
                        raise HarnessError("Unverifiable process group member")
                    return True
            except (OSError, IndexError, ValueError):
                continue
        return False

    async def wait_exit(self, info, timeout):
        deadline = time.monotonic() + timeout
        while self.group_alive(info):
            if time.monotonic() >= deadline:
                raise HarnessError("Training process group has not exited; method and GPU leases remain held")
            await asyncio.sleep(.2)

    async def loop(self):
        while not self.shutdown:
            try:
                await self.poll_chats()
                for job_id, task in list(self.workers.items()):
                    if task.done():
                        with contextlib.suppress(asyncio.CancelledError):
                            error = task.exception()
                            if error:
                                self.store.event("scheduler.error", jobId=job_id, error=str(error))
                        del self.workers[job_id]
                jobs = self.store.all("jobs")
                candidates = [j for j in jobs if j.get("authorized") and j["status"] in ("queued", "running", "waiting_gpu")]
                for job in candidates:
                    if job.get("nextRunAt", 0) > time.time():
                        continue
                    if any((self.store.get("jobs", dependency) or {}).get("status") not in ("done", "failed", "cancelled")
                           for dependency in job.get("afterJobs", [])):
                        continue
                    if len(self.workers) >= self.max_papers:
                        break
                    if job["id"] not in self.workers:
                        self.workers[job["id"]] = asyncio.create_task(self.work(job["id"]))
                # Imported live jobs remain visible, with their results streamed, without taking over ownership.
                for job in jobs:
                    if job.get("evaluation"):
                        await self.poll_progress(job)
                for batch in self.store.all("batches"):
                    members = [self.store.get("jobs", i) for i in batch["jobs"]]
                    status = "done" if all(j["status"] in ("done", "failed", "cancelled") for j in members) else "running" if any(j["status"] in ("running", "queued", "waiting_gpu") for j in members) else "paused"
                    if batch["status"] != status:
                        self.store.update("batches", batch["id"], status=status)
                        self.store.event("batch.updated", batchId=batch["id"], status=status)
            except Exception as error:
                self.store.event("scheduler.error", error=str(error))
            await asyncio.sleep(.5)

    async def poll_chats(self):
        if self.bridge.error:
            return
        for binding in self.store.all("bindings"):
            if binding.get("transport") == "cli":
                continue
            sid = binding["sessionId"]
            try:
                after = binding.get("eventCursor", 0)
                reply = await self.bridge.call("GET", f"/sessions/{sid}/events?after={after}&epoch={binding.get('eventEpoch', '')}", timeout=2)
                events = reply.get("events", [])
                if events:
                    with self.store.transaction():
                        for event in events:
                            self.store.event("session.event", sessionId=sid, jobId=binding.get("jobId"), data=event)
                        self.store.update("bindings", binding["id"], eventCursor=reply.get("cursor", events[-1].get("seq", after)), eventEpoch=reply.get("epoch", ""))
                elif reply.get("reset"):
                    self.store.update("bindings", binding["id"], eventCursor=reply.get("cursor", 0), eventEpoch=reply.get("epoch", ""))
                    self.store.event("session.snapshot_required", sessionId=sid)
            except Exception:
                continue

    async def root(self):
        if self.root_session:
            return self.root_session
        previous = next((b for b in self.store.all("bindings") if b.get("role") == "coordinator"), None)
        if previous:
            self.root_session = previous["sessionId"]
        else:
            result = await self.bridge.call("POST", "/sessions", {"title": "TSAD Coordinator", "cwd": str(ROOT),
                "systemPrompt": "You are the TSAD coordinator. Ordinary messages are normal conversation. Explicit reproduction requests may call tsad_reproduce with structured papers. Never start an experiment from an ambiguous general question. Domain jobs own training, resource allocation and publication. The fixed workflow uses short CLI workers; report actual controller events."})
            self.root_session = result.get("sessionId") or result["id"]
            self.bind(self.root_session, "coordinator")
        return self.root_session

    def bind(self, session_id, role, job_id=None, parent=None):
        return self.store.put("bindings", {"id": session_id, "sessionId": session_id, "role": role,
            "jobId": job_id, "parentSessionId": parent, "createdAt": now(), "eventCursor": 0})

    def change(self, job_id, **fields):
        job = self.store.update_active("jobs", job_id, **fields)
        if job is None:
            raise asyncio.CancelledError()
        self.store.event("job.updated", jobId=job_id, job=job)
        return job

    def _record_worker_process(self, job_id, role, process):
        current = self.store.get("jobs", job_id)
        if not current:
            return
        processes = {**current.get("workerProcesses", {}), role: process}
        updates = {"workerProcesses": processes}
        if role in ("hyperparameters", "interface", "environment"):
            branches = dict(current.get("branches") or {})
            branches[role] = {"status": process["status"], "sessionId": process.get("sessionId"),
                              "artifact": workers.relative(workers.layout(Task.open(current["taskId"]))[role]),
                              "reason": process.get("error", "")}
            updates["branches"] = branches
        self.store.update("jobs", job_id, **updates)
        process_id = process.get("id") or f"{job_id}:{role}:{process.get('runId', process.get('pid'))}"
        self.store.put("processes", {**process, "id": process_id, "jobId": job_id, "role": role})
        sid = process.get("sessionId")
        if sid:
            binding = self.store.get("bindings", sid) or {"id": sid, "sessionId": sid,
                "jobId": job_id, "role": role, "transport": "cli", "createdAt": now(), "eventCursor": 0}
            binding.update(process=process, status=process["status"], stepCount=process.get("steps", 0),
                           eventsPath=process.get("log"), title=f"{current['method']}: {role}")
            if process["status"] == "completed":
                binding["completedStageToken"] = process.get("stageToken")
            self.store.put("bindings", binding)
        self.store.event("worker.process", jobId=job_id, role=role, process=process)

    async def run_worker(self, job, task, role, context=None, *, read_only=False, message=None):
        async with self.session_locks.setdefault((job["id"], role), asyncio.Lock()):
            async with self.model_slots:
                return await self._run_worker(job, task, role, context, read_only=read_only, message=message)

    async def _run_worker(self, job, task, role, context=None, *, read_only=False, message=None):
        slot = role + ":chat" if read_only else role
        token = f"{job['stage']}:{job.get('attemptId') or 'initial'}:{job.get('repairRounds', 0)}"
        current_job = self.store.get("jobs", job["id"])
        previous = (current_job.get("workerProcesses") or {}).get(role, {})
        if read_only:
            token = "chat:" + identity("turn")
        if any(worker_alive(process) for process in process_tree(previous)):
            raise HarnessError(f"The {role} CLI worker or its tool server is still running; refusing a duplicate turn")
        handoff_error = None
        researching = role in ("hyperparameters", "interface", "environment")
        refresh_environment = role == "environment" and current_job.get("environmentError")
        if (not read_only and not refresh_environment and previous.get("kind") not in ("worker-startup", "worker-tools")
                and previous.get("status") == "completed" and (previous.get("stageToken") == token or researching)):
            try:
                validated = workers.validate(task, role)
            except (HarnessError, ValueError, OSError) as error:
                handoff_error = str(error)[:2000]
            else:
                self._record_worker_process(job["id"], role, previous)
                return validated
        loop = asyncio.get_running_loop()
        cancel = threading.Event()
        self.cli_cancels[job["id"], slot] = cancel

        def dispatch(function, *args):
            # Workers never write Store. Marshal state/event writes to its owner loop
            # and wait for durable process registration before delivering model input.
            future = concurrent.futures.Future()
            def invoke():
                try:
                    future.set_result(function(*args))
                except BaseException as error:
                    future.set_exception(error)
            loop.call_soon_threadsafe(invoke)
            return future.result()

        def record_event(event):
            latest = (self.store.get("jobs", job["id"]).get("workerProcesses") or {}).get(role, {})
            sid = latest.get("sessionId")
            self.store.event("session.event", sessionId=sid, jobId=job["id"], role=role, data=event)
            if sid and event.get("type") == "status" and event.get("phase") == "step_start":
                binding = self.store.get("bindings", sid)
                if binding:
                    self.store.update("bindings", sid, stepCount=binding.get("stepCount", 0) + 1)

        queued = [row for row in self.store.all("messages") if row.get("jobId") == job["id"] and
                  row.get("role") == role and row.get("status") == "queued"] if not read_only else []
        combined = {**(context or {}), "user_messages": [row["text"] for row in queued]}
        if refresh_environment:
            combined["installation_error"] = current_job["environmentError"]
        if role == "diagnoser":
            combined["human_corrections"] = task.read_jsonl(task.dir / "corrections.jsonl")[-5:]
        prompts = workers.prompt(task, role, combined)
        if handoff_error:
            prompts += (
                "\nYour previous handoff could not be read by the controller: " + handoff_error
                + "\nCorrect only the assigned handoff using the source already inspected, publish it "
                "and stop. Do not investigate the algorithm or Harness internals again.\n")
        if previous.get("reason") == "max-tokens":
            prompts += (
                "\nThe previous response hit its output limit. Continue from the source already inspected "
                "in this session; do not restart the investigation. Publish a concise version of the "
                "assigned deliverable now, recording any unresolved detail in it, then stop.\n")
        recovery = dict((current_job.get("workerRecovery") or {}).get(role) or {})
        if recovery.get("stageToken") != token:
            recovery = {"stageToken": token, "outputLimits": 0, "noProgress": 0, "freshSessions": 0}
            if previous.get("reason") == "max-tokens" and previous.get("stageToken") == token:
                recovery.update(outputLimits=1, noProgress=int(not previous.get("toolCalls")))
        llm = load().raw.get("llm", {})
        base_tokens = int(llm.get("max_output_tokens", 8192))
        limit_tokens = max(base_tokens, int(llm.get("max_recovery_output_tokens", 32768)))
        learned = (current_job.get("workerConfig") or {}).get(role) or {}
        learned_tokens = learned.get("maxOutputTokens", base_tokens) if learned.get("model", "") == llm.get("model", "") else base_tokens
        if previous.get("status") == "completed":
            learned_tokens = max(learned_tokens, previous.get("maxOutputTokens") or base_tokens)
        output_tokens = min(limit_tokens, max(learned_tokens, base_tokens * 2 ** min(recovery["outputLimits"], 2)))
        fresh_session = (not read_only and recovery["noProgress"] >= 2 and recovery["freshSessions"] < 2)
        assigned = workers.write_paths(task, role)
        before = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in assigned if path.is_file()}
        if fresh_session:
            recovery.update(freshSessions=recovery["freshSessions"] + 1, noProgress=0)
            self.store.update("jobs", job["id"], workerRecovery={**(current_job.get("workerRecovery") or {}), role: recovery})
            task.log("worker/recovery", role=role, action="fresh_session", previous_session=previous.get("sessionId"),
                     max_output_tokens=output_tokens, existing_files=list(before))
            prompts += "\nContinue the integration from the existing files on disk. Implement the missing assigned files in small tool writes and finish.\n"
        if read_only:
            prompts = "This is a read-only continuation of your existing worker session. Explain and inspect artifacts. For changes, ask the user to submit a job correction.\n" + str(message or "")
        for row in queued:
            self.store.update("messages", row["id"], status="delivering")
        operation = asyncio.create_task(asyncio.to_thread(agent.run_agent, task, prompts, label=role,
            role=role, session_id=None if fresh_session else previous.get("sessionId"), cancel=cancel,
            stage_token=token, write_enabled=not read_only,
            max_output_tokens=output_tokens, fresh_session=fresh_session,
            on_process=lambda process: dispatch(self._record_worker_process, job["id"], slot, process),
            on_event=lambda event: dispatch(record_event, event)))
        try:
            result = await asyncio.shield(operation)
            recovered_handoff = None
            if not result.ok:
                if result.reason == "max-tokens":
                    after = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in assigned if path.is_file()}
                    changed = after != before
                    recovery.update(outputLimits=recovery["outputLimits"] + 1,
                                    noProgress=0 if changed else recovery["noProgress"] + 1,
                                    toolCalls=result.tool_calls, filesChanged=changed)
                    latest = self.store.get("jobs", job["id"])
                    configuration = {"maxOutputTokens": min(limit_tokens, output_tokens * 2),
                                     "model": llm.get("model", ""), "reason": "output_limit", "updatedAt": now()}
                    self.store.update("jobs", job["id"], workerRecovery={**(latest.get("workerRecovery") or {}), role: recovery},
                                      workerConfig={**(latest.get("workerConfig") or {}), role: configuration})
                    task.log("worker/recovery", role=role, action="output_limit", **recovery,
                             max_output_tokens=output_tokens)
                    if role == "implementer" and changed:
                        with contextlib.suppress(HarnessError, ValueError, OSError):
                            delivered = workers.validate(task, role)
                            self.store.event("worker.handoff_recovered", jobId=job["id"], role=role, files=delivered)
                            recovered_handoff = delivered
                    if recovered_handoff is None:
                        raise agent.WorkerOutputLimitError(f"{role}: model output limit; adapt output budget/session from recorded file progress")
                elif re.search(r"(?:API error|HTTP|status(?: code)?)[^\n]{0,30}\b(?:429|5\d\d)\b|Upstream connection failed|ECONNRESET|ETIMEDOUT", result.error, re.I):
                    raise agent.WorkerTransportError(f"{role}: model transport: {result.error[-2000:]}")
                elif not result.reached_model and result.reason != "aborted":
                    raise agent.WorkerStartupError(f"{role}: runtime startup: {result.error[-2000:]}")
                elif recovered_handoff is None:
                    raise HarnessError(f"{role}: {result.reason}: {result.error[-2000:]}")
            if recovered_handoff is not None:
                validated = recovered_handoff
            elif read_only:
                validated = {"finalText": result.final_text, "sessionId": result.session_id}
            else:
                try:
                    validated = workers.validate(task, role)
                except (HarnessError, ValueError, OSError) as error:
                    raise workers.WorkerHandoffError(f"{role}: unreadable handoff: {error}") from error
            latest = self.store.get("jobs", job["id"])
            recoveries = dict(latest.get("workerRecovery") or {})
            recoveries.pop(role, None)
            configurations = dict(latest.get("workerConfig") or {})
            if not read_only:
                configurations[role] = {"maxOutputTokens": output_tokens, "model": llm.get("model", ""),
                                        "reason": "successful_handoff", "updatedAt": now()}
                task.log("config/feedback", setting="worker_output_tokens", role=role, selected=output_tokens,
                         reason="successful_handoff")
            self.store.update("jobs", job["id"], runtimeRetries=0, runtimeRetryKey=None, nextRunAt=0,
                              workerRecovery=recoveries, workerConfig=configurations)
            for row in queued:
                self.store.update("messages", row["id"], status="delivered")
            return validated
        except asyncio.CancelledError:
            cancel.set()
            await operation
            raise
        finally:
            self.cli_cancels.pop((job["id"], slot), None)
            for row in queued:
                current_message = self.store.get("messages", row["id"])
                if current_message.get("status") == "delivering":
                    self.store.update("messages", row["id"], status="queued")
            if not self.shutdown and not cancel.is_set():
                for row in self.store.all("messages"):
                    if row.get("jobId") == job["id"] and row.get("role") == role and row.get("status") == "queued":
                        self.store.update("messages", row["id"], status="continuing")
                        self.continue_worker_message(row, job, role)

    def worker_snapshot(self, session_id):
        binding = next((b for b in self.store.all("bindings") if b.get("sessionId") == session_id and b.get("transport") == "cli"), None)
        if not binding:
            return None
        path = ROOT / binding.get("eventsPath", "")
        events = []
        if path.is_file():
            for line in path.read_text(errors="replace").split("\n"):
                with contextlib.suppress(ValueError):
                    events.append(json.loads(line))
        total = len(events)
        snapshot = {**binding, "running": binding.get("status") == "running", "events": events[-400:],
                    "eventsStart": max(0, total - 400), "eventCursor": total,
                    "eventEpoch": binding.get("eventsPath", "")}
        final = binding.get("process", {}).get("finalText")
        if final:
            snapshot["history"] = [{"role": "assistant", "content": final}]
        snapshot["messages"] = [message for message in self.store.all("messages")
                                if message.get("sessionId") == session_id]
        return snapshot

    async def worker_message(self, session_id, text, request_id):
        binding = next((b for b in self.store.all("bindings") if b.get("sessionId") == session_id and b.get("transport") == "cli"), None)
        if not binding:
            raise HarnessError("Unknown CLI worker session")
        job = self.store.get("jobs", binding["jobId"])
        role = binding["role"].split(":", 1)[0]
        process = binding.get("process") or {}
        active = process.get("status") == "running" and process_identity(process.get("pid", 0)) == process.get("processIdentity") and process.get("processIdentity")
        def save():
            self.store.put("messages", {"id": request_id, "sessionId": session_id, "jobId": job["id"],
                "role": role, "text": text, "status": "queued" if active else "continuing", "createdAt": now()})
            self.store.event("worker.message", sessionId=session_id, jobId=job["id"], text=text)
            self.store.event("session.event", sessionId=session_id, jobId=job["id"],
                             data={"type": "user", "text": text})
            return {"accepted": True, "requestId": request_id, "delivery": "next_role_turn" if active else "session_continuation"}
        result = self.store.once(request_id, "worker.message", {"sessionId": session_id, "text": text}, save)
        row = self.store.get("messages", request_id)
        if not active and row.get("status") == "continuing":
            self.continue_worker_message(row, job, role)
        return result

    def continue_worker_message(self, row, job, role):
        request_id, session_id = row["id"], row["sessionId"]
        if any(getattr(turn, "request_id", None) == request_id for turn in self.chat_workers):
            return
        async def continue_session():
            try:
                task = Task.open(job["taskId"])
                await self.run_worker(job, task, role, read_only=True, message=row["text"])
                self.store.update("messages", request_id, status="delivered")
            except asyncio.CancelledError:
                self.store.update("messages", request_id, status="queued")
                raise
            except Exception as error:
                self.store.update("messages", request_id, status="failed", error=str(error))
                self.store.event("worker.error", sessionId=session_id, jobId=job["id"], error=str(error))
        turn = asyncio.create_task(continue_session())
        turn.request_id = request_id
        self.chat_workers.add(turn)
        turn.add_done_callback(self.chat_workers.discard)

    async def worker_cancel(self, session_id):
        binding = next((b for b in self.store.all("bindings") if b.get("sessionId") == session_id and b.get("transport") == "cli"), None)
        if not binding:
            raise HarnessError("Unknown CLI worker session")
        # Cancelling a reply does not cancel a detached experiment.
        for key in (binding["role"], binding["role"].split(":", 1)[0] + ":chat"):
            event = self.cli_cancels.get((binding["jobId"], key))
            if event:
                event.set()
        return {"accepted": True}

    def environment_capacity(self):
        try:
            memory = {line.split(":", 1)[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
            cache = Path(load().raw.get("paths", {}).get("uv_cache_dir", "runtime/uv-cache")).expanduser()
            if not cache.is_absolute():
                cache = ROOT / cache
            while not cache.exists():
                cache = cache.parent
            disk = shutil.disk_usage(cache)
            if memory["MemAvailable"] < max(4 * 1024 ** 2, memory["MemTotal"] // 10) or disk.free < 8 * 1024 ** 3:
                return 1
        except (OSError, KeyError, ValueError):
            return 1
        return self.environment_policy["limit"]

    async def install_environment(self, job_id, task, *, recovery=None):
        self.change(job_id, status="queued", activeOperation="environment_wait", reason="Waiting for dependency installation slot")
        async with self.environment_condition:
            self.environment_waiting += 1
            try:
                await self.environment_condition.wait_for(lambda: self.environment_active < self.environment_capacity())
                self.environment_active += 1
            finally:
                self.environment_waiting -= 1
        succeeded = False
        try:
            self.change(job_id, status="running", activeOperation="environment_install", reason="")
            loop = asyncio.get_running_loop()
            cancel = threading.Event()
            self.cli_cancels[job_id, "environment-install"] = cancel
            def persist_process(process):
                future = concurrent.futures.Future()
                def persist():
                    try:
                        self._record_worker_process(job_id, "environment-install", process)
                        future.set_result(None)
                    except BaseException as error:
                        future.set_exception(error)
                loop.call_soon_threadsafe(persist)
                future.result()
            try:
                options = {"recovery": recovery} if recovery else {}
                operation = asyncio.create_task(asyncio.to_thread(pipeline.install_environment, task,
                    cancel=cancel, on_process=persist_process, **options))
                try:
                    result = await asyncio.shield(operation)
                    succeeded = True
                    return result
                except asyncio.CancelledError:
                    cancel.set()
                    await operation
                    raise
            finally:
                self.cli_cancels.pop((job_id, "environment-install"), None)
                if (self.store.get("jobs", job_id) or {}).get("authorized"):
                    self.change(job_id, activeOperation=None)
        finally:
            async with self.environment_condition:
                self.environment_active -= 1
                policy = self.environment_policy
                previous_limit = policy["limit"]
                policy["successes"] = policy.get("successes", 0) + 1 if succeeded else 0
                if not succeeded:
                    policy["limit"] = max(1, policy["limit"] // 2)
                elif policy["successes"] >= 2:
                    policy.update(limit=min(self.environment_ceiling, policy["limit"] + 1), successes=0)
                self.store.put("cursors", policy)
                task.log("config/feedback", setting="environment_install_concurrency", previous=previous_limit,
                         selected=policy["limit"], effective=self.environment_capacity(), success=succeeded)
                self.environment_condition.notify_all()

    def route_runtime_failure(self, job_id, task, findings):
        job = self.store.get("jobs", job_id)
        normalized = []
        for finding in findings:
            error = "\n".join(str(evidence.get("error", "")) for evidence in finding.get("evidence") or [])
            category = pipeline.failure_category(error, task.method)
            normalized.append({**finding, "category": category} if category in (
                "environment", "resource", "publication") else finding)
        findings = normalized
        errors = "\n".join(str(evidence.get("error", "")) for finding in findings
                           for evidence in finding.get("evidence") or [])
        if re.search(r"CUDA out of memory|CUDA error: out of memory|torch\.OutOfMemoryError", errors, re.I):
            resources = job.get("lastEvaluationResources") or {}
            selected = set(str(resources.get("gpus", "")).split(","))
            devices = [device for device in resources.get("devices", []) if str(device["index"]) in selected]
            total = max((device.get("total_memory_mib") or 0 for device in devices), default=0)
            if total:
                previous = job.get("gpuFeedback") or {}
                retries = previous.get("retries", 0) + 1
                feedback = {"retries": retries, "min_gpu_memory_mib": int(total * 1.25) + 1,
                            "reason": "CUDA out of memory with official method parameters"}
                if retries > 3:
                    self.change(job_id, status="failed", authorized=False,
                                reason="GPU memory remains insufficient after three resource changes; official training parameters retained")
                else:
                    self.change(job_id, stage="EVALUATE", status="queued", gpuFeedback=feedback,
                                evaluationFailed=False, repairTarget=None,
                                reason="Selecting a GPU with more available memory after CUDA OOM")
                    task.set(phase=Phase.EVALUATE, status=Status.READY, eval=None)
                task.log("config/feedback", setting="gpu_memory", previous_gpus=resources.get("gpus"), **feedback)
                return True
        for finding in findings:
            for evidence in finding.get("evidence") or []:
                if finding.get("category") != "publication" and not runner.publication_failure({"error": evidence.get("error", "")}):
                    continue
                path = ROOT / evidence.get("file", "")
                try:
                    manifest = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if not manifest.get("schema") or not manifest.get("dataset"):
                    continue
                key = f"{manifest['schema']}:{manifest['dataset']}:{manifest.get('attempt_id')}"
                previous = job.get("publicationRecovery") or {}
                retries = (previous.get("retries", 0) if previous.get("key") == key else 0) + 1
                recovery = {"key": key, "retries": retries, "dataset": manifest["dataset"],
                            "schema": manifest["schema"], "attemptId": manifest.get("attempt_id")}
                if retries > 5:
                    self.change(job_id, status="failed", authorized=False,
                                reason="Harness publication failed repeatedly; computed scores and successful datasets retained")
                    return True
                self.change(job_id, stage="EVALUATE", status="queued", evaluationFailed=False,
                            publicationRecovery=recovery, nextRunAt=time.time() + min(120, 5 * retries),
                            reason="Recovering publication from existing scores", repairTarget=None)
                task.set(phase=Phase.EVALUATE, status=Status.READY, eval=None)
                task.log("runtime/recovery", category="publication", **recovery)
                return True
        environment_errors = [finding for finding in findings if finding.get("category") == "environment"]
        if environment_errors and job.get("environmentRepairs", 0) < 3:
            error = json.dumps(environment_errors, ensure_ascii=False)
            previous = job.get("environmentRecovery") or {}
            self.change(job_id, stage="RESEARCH", status="queued", environmentError=error,
                        environmentRepairs=job.get("environmentRepairs", 0) + 1, evaluationFailed=False,
                        environmentRecovery=previous, repairTarget=None,
                        reason="Recovering runtime dependencies before continuing the method")
            task.set(phase=Phase.RESEARCH, status=Status.READY, eval=None)
            task.log("runtime/recovery", category="environment", error=error)
            return True
        if environment_errors:
            task.set(status=Status.FAILED, reason="Runtime dependencies still fail after three environment repairs")
            self.change(job_id, status="failed", authorized=False,
                        reason="Runtime dependencies still fail after three environment repairs")
            return True
        resource_errors = [finding for finding in findings if finding.get("category") == "resource"]
        if resource_errors:
            reason = "Execution resource limit prevents continuing with official method parameters"
            task.set(status=Status.FAILED, reason=reason)
            self.change(job_id, status="failed", authorized=False, reason=reason,
                        diagnosticFindings=resource_errors)
            task.log("runtime/recovery", category="resource", action="resource_blocked", findings=resource_errors)
            return True
        return False

    def reconcile_repair_rounds(self, job):
        if job.get("repairCounterVersion") == 2:
            return job
        count = job.get("repairRounds", 0)
        diagnostics = sorted((row for row in self.store.all("diagnostics")
                              if row.get("jobId") == job["id"] and row.get("action") == "repair"),
                             key=lambda row: row.get("createdAt", ""))[:count]
        runtime = []
        if len(diagnostics) == count:
            for row in diagnostics:
                error = "\n".join(str(item.get("error", "")) for item in row.get("evidence") or [])
                category = pipeline.failure_category(error, job["method"])
                if row.get("category") == "environment" or category in ("environment", "resource", "publication") or re.search(
                        r"(?:git clone failed|UV installation failed|Disk quota exceeded|No space left on device|HTTP (?:407|429|5\d\d))", error, re.I):
                    runtime.append(row["id"])
        fields = {"repairCounterVersion": 2}
        if runtime:
            fields.update(repairRounds=max(0, count - len(runtime)),
                          legacyRepairRounds={"count": count, "reclassifiedDiagnostics": runtime, "at": now()})
            self.store.event("job.repair_counter_reconciled", jobId=job["id"], previous=count,
                             selected=fields["repairRounds"], diagnostics=runtime)
        return self.store.update("jobs", job["id"], **fields)

    async def finish_implementation(self, job_id, task):
        check = await self.thread(runner.check_method, task)
        if not check.get("ok"):
            finding = {"category": pipeline.failure_category(check.get("error", ""), task.method),
                       "message": "Method import failed", "evidence": [check]}
            task.set(phase=Phase.DIAGNOSE, status=Status.READY,
                     implementation_errors=[check], evaluation_findings=[finding])
            self.change(job_id, stage="DIAGNOSE", diagnosticFindings=[finding], evaluationFailed=True, repairTarget="IMPLEMENT")
            self.route_runtime_failure(job_id, task, [finding])
            return False
        pipeline.snapshot_implementation(task)
        task.set(phase=Phase.EVALUATE, status=Status.READY, eval=None)
        job = self.store.get("jobs", job_id)
        if job.get("attemptId"):
            self.store.update("attempts", job["attemptId"], sourceHash=runner.source_hash(task.method_dir), status="evaluating")
        recovery = job.get("environmentRecovery")
        self.change(job_id, stage="EVALUATE", evaluation=None, diagnosticFindings=[],
                    environmentError=None, environmentRecovery={**recovery, "verifiedImport": True} if recovery else None)
        return True

    async def repair_environment(self, job_id, task):
        job = self.store.get("jobs", job_id)
        previous = job.get("environmentRecovery") or {}
        plan = await self.thread(environments.plan_recovery, task, job.get("environmentError", ""), previous=previous)
        recovery = {**previous, **plan}
        self.change(job_id, environmentRecovery=recovery)
        task.log("runtime/recovery_action", category="environment", **recovery)
        if plan["mode"] == "research":
            await self.run_worker(self.store.get("jobs", job_id), task, "environment", {"installation_error": job["environmentError"]})
        await self.install_environment(job_id, task, recovery=plan)
        self.change(job_id, environmentError=None)
        try:
            workers.validate(task, "implementer")
        except (HarnessError, ValueError, OSError):
            task.set(phase=Phase.IMPLEMENT, status=Status.READY)
            self.change(job_id, stage="IMPLEMENT", environmentRecovery=None)
        else:
            await self.finish_implementation(job_id, task)

    async def work(self, job_id):
        try:
            while not self.shutdown:
                job = self.store.get("jobs", job_id)
                if not job or not job.get("authorized") or job["status"] in ("done", "failed", "paused", "cancelled"):
                    return
                if not job.get("taskId"):
                    source = job["paper"]
                    value = source.get("input", "")
                    kw = {k: source[k] for k in ("arxiv", "title", "pdf", "url") if source.get(k)}
                    if not kw:
                        kw = {"arxiv": value} if papers.parse_arxiv_id(value) else {"url": value} if value.startswith(("http://", "https://")) else {"pdf": value} if value.lower().endswith(".pdf") else {"title": value}
                    paper = await self.thread(papers.resolve, **kw)
                    task = Task.create(job["method"], paper=paper, code_url=source.get("code"),
                                       env_hint=job.get("environment") or source.get("env"), family=source.get("family"))
                    job = self.change(job_id, taskId=task.id)
                task = Task.open(job["taskId"])
                if not job.get("resultsRecovered") and job.get("legacy"):
                    job = await self.thread(self.recover_results, job, task)
                if job.get("evaluationPlan") and not task.state.get("evaluation_plan"):
                    task.set(evaluation_plan=job["evaluationPlan"])
                stage = job["stage"]
                if stage in ("EVIDENCE", "SMOKE"):
                    stage = "RESEARCH" if stage == "EVIDENCE" else "EVALUATE"
                    job = self.change(job_id, stage=stage)
                if job.get("until") == stage:
                    self.change(job_id, status="paused", authorized=False, reason="Reached requested phase")
                    return
                self.change(job_id, status="running", reason="")
                if stage in ("FETCH", "RESEARCH", "IMPLEMENT"):
                    resources = ["method:" + task.method, "environment:" + workers.relative(task.method_dir / "environment")]
                    if not self.store.acquire(resources, job_id, mode="write", stage=stage):
                        self.change(job_id, status="queued", reason="Waiting for method/environment writer")
                        return
                    try:
                        if stage == "FETCH":
                            task.set(phase=Phase.FETCH, status=Status.RUNNING)
                            discovery = None
                            with contextlib.suppress(HarnessError, ValueError, OSError):
                                discovery = await self.thread(workers.validate, task, "fetcher")
                            if not discovery:
                                meta = await self.thread(pipeline.fetch_paper, task)
                                await self.run_worker(job, task, "fetcher", {"paper_metadata": meta,
                                    "diagnostics": job.get("diagnosticFindings", [])})
                            await self.thread(pipeline.prepare_source, task)
                            task.set(phase=Phase.RESEARCH, status=Status.READY)
                            self.change(job_id, stage="RESEARCH", runtimeRetries=0, runtimeRetryKey=None, nextRunAt=0)
                        elif stage == "RESEARCH":
                            if job.get("environmentError"):
                                await self.repair_environment(job_id, task)
                                continue
                            task.set(phase=Phase.RESEARCH, status=Status.RUNNING)
                            self.change(job_id, branches={role: {"status": "queued"} for role in
                                                         ("hyperparameters", "interface", "environment")})
                            turns = [asyncio.create_task(self.run_worker(job, task, role, {"diagnostics": job.get("diagnosticFindings", [])})) for role in
                                     ("hyperparameters", "interface", "environment")]
                            try:
                                outcomes = await asyncio.gather(*turns, return_exceptions=True)
                            except BaseException:
                                for turn in turns:
                                    turn.cancel()
                                await asyncio.gather(*turns, return_exceptions=True)
                                raise
                            for outcome in outcomes:
                                if isinstance(outcome, BaseException):
                                    raise outcome
                            plans = await self.thread(pipeline.evaluation_plan, task)
                            job = self.change(job_id, evaluationPlan=plans)
                            environment = await self.install_environment(job_id, task)
                            task.set(phase=Phase.IMPLEMENT, status=Status.READY)
                            self.change(job_id, stage="IMPLEMENT", environment=environment.get("environment"), repairTarget=None,
                                        environmentError=None)
                        else:
                            previous = self.store.get("attempts", job.get("attemptId", ""))
                            attempt_id = previous["id"] if previous and previous.get("status") == "running" else identity("attempt")
                            if not previous or attempt_id != previous["id"]:
                                self.store.put("attempts", {"id": attempt_id, "jobId": job_id,
                                    "number": job.get("repairRounds", 0) + 1, "stage": "IMPLEMENT", "createdAt": now(), "status": "running"})
                            job = self.change(job_id, attemptId=attempt_id)
                            task.set(phase=Phase.IMPLEMENT, status=Status.RUNNING)
                            await self.run_worker(job, task, "implementer", pipeline.handoff(task))
                            await self.install_environment(job_id, task)
                            await self.finish_implementation(job_id, task)
                    finally:
                        await self.release_if_idle(job_id)
                elif stage == "EVALUATE":
                    await self.evaluate(job, task)
                    return
                elif stage == "DIAGNOSE":
                    if self.route_runtime_failure(job_id, task, job.get("diagnosticFindings", [])):
                        return
                    task.set(phase=Phase.DIAGNOSE, status=Status.RUNNING)
                    diagnosis = await self.run_worker(job, task, "diagnoser", {"findings": job.get("diagnosticFindings", [])})
                    diagnostic = self.diagnose(job_id, diagnosis)
                    if self.route_runtime_failure(job_id, task, [diagnostic]):
                        return
                    budget = min(4, int(load()["budget"]["max_repair_rounds"]))
                    if diagnostic["action"] == "repair" and job.get("repairRounds", 0) < budget:
                        target = job.get("repairTarget") or "IMPLEMENT"
                        task.set(phase=Phase(target), status=Status.READY, implementation_errors=[diagnostic], eval=None)
                        if job.get("attemptId"):
                            self.store.update("attempts", job["attemptId"], status="repair_required")
                        self.change(job_id, stage=target, repairRounds=job.get("repairRounds", 0) + 1,
                                    evaluation=None, status="queued", repairTarget=None)
                    elif job.get("evaluationFailed"):
                        task.set(status=Status.FAILED, reason="No evidenced repair or repair budget exhausted")
                        self.change(job_id, status="failed", authorized=False, reason="No evidenced repair or repair budget exhausted")
                        return
                    else:
                        # Diagnosis concerns the completed dataset. Continue the
                        # configured queue; only evaluate() can establish that
                        # every dataset has finished before entering REPORT.
                        task.set(phase=Phase.EVALUATE, status=Status.READY)
                        self.change(job_id, stage="EVALUATE", status="queued")
                elif stage == "REPORT":
                    task.set(phase=Phase.REPORT, status=Status.RUNNING)
                    benchmark = await self.thread(report.finalize, task)
                    manifest = json.loads(benchmark.read_text())
                    task.set(phase=Phase.DONE, status=Status.DONE)
                    self.change(job_id, status="done", stage="DONE", authorized=False, benchmarkRevision=manifest.get("revision"))
                    if job.get("attemptId"):
                        self.store.update("attempts", job["attemptId"], status="done")
                    return
                elif stage == "DONE":
                    self.change(job_id, status="done", authorized=False)
                    return
                else:
                    raise HarnessError("Unknown job stage " + stage)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            job = self.store.get("jobs", job_id)
            if not job or not job.get("authorized"):
                return
            if isinstance(error, environments.EnvironmentInstallError):
                self.route_runtime_failure(job_id, task, [{"category": "environment", "message": str(error),
                    "evidence": [{"error": str(error)}]}])
                return
            if isinstance(error, (agent.WorkerTransportError, papers.SourceFetchError, workers.WorkerHandoffError,
                                  environments.EnvironmentStorageError)):
                capacity = isinstance(error, agent.WorkerTransportError) and "no available accounts" in str(error).lower()
                storage = isinstance(error, environments.EnvironmentStorageError)
                kind = ("source_download" if isinstance(error, papers.SourceFetchError)
                        else "environment_storage" if storage
                        else "worker_handoff" if isinstance(error, workers.WorkerHandoffError)
                        else "model_output_limit" if isinstance(error, agent.WorkerOutputLimitError)
                        else "model_capacity" if capacity else "model_transport")
                retry_key = f"{job['stage']}:{kind}"
                retries = (job.get("runtimeRetries", 0) if job.get("runtimeRetryKey") == retry_key else 0) + 1
                if retries <= 5 or capacity or storage:
                    retry_at = time.time() + min(120, 15 * 2 ** min(retries - 1, 3))
                    self.change(job_id, status="queued", runtimeRetries=retries, runtimeRetryKey=retry_key, nextRunAt=retry_at,
                                reason=str(error)[:2000])
                    self.store.event("worker.runtime_retry", jobId=job_id, retry=retries,
                                     nextRunAt=retry_at, retryKey=retry_key, error=str(error))
                    return
            evidence = {"file": workers.relative(load().tasks_dir / job.get("taskId", "") / "journal.jsonl"), "error": str(error)}
            if job.get("taskId"):
                task = Task.open(job["taskId"])
                task.log("scheduler/error", stage=job["stage"], error=str(error))
            startup = isinstance(error, (agent.WorkerStartupError, agent.WorkerTransportError,
                                         papers.SourceFetchError, workers.WorkerHandoffError))
            finding = {"category": "unsupported" if isinstance(error, pipeline.UnsupportedPaper) else "harness_runtime" if startup else "execution",
                       "message": str(error), "action": "report_only", "evidence": [evidence]}
            self.diagnose(job_id, finding)
            if job.get("evaluation"):
                # An observer failure never authorizes repair alongside a
                # potentially live training process. Keep exact ownership.
                self.change(job_id, status="paused", reason=str(error)[:2000], authorized=False)
            elif job.get("taskId") and not startup and not isinstance(error, pipeline.UnsupportedPaper) and job.get("stage") in (
                    "FETCH", "RESEARCH", "IMPLEMENT", "EVALUATE", "REPORT"):
                task.set(phase=Phase.DIAGNOSE, status=Status.READY, evaluation_findings=[finding])
                self.change(job_id, stage="DIAGNOSE", status="queued", evaluationFailed=True,
                            diagnosticFindings=[finding], repairTarget=job["stage"] if job["stage"] in (
                                "FETCH", "RESEARCH", "REPORT") else "IMPLEMENT")
            else:
                self.change(job_id, status="failed", reason=str(error)[:2000], authorized=False,
                            unsupported=isinstance(error, pipeline.UnsupportedPaper))
        finally:
            job = self.store.get("jobs", job_id)
            if job and not job.get("evaluation"):
                await self.release_if_idle(job_id)

    async def release_if_idle(self, job_id):
        job = self.store.get("jobs", job_id)
        if job and job.get("evaluation"):
            return False
        try:
            for binding in self.store.all("bindings"):
                if binding.get("jobId") == job_id and binding.get("transport") != "cli":
                    state = await self.bridge.call("GET", f"/sessions/{binding['sessionId']}")
                    if self.native_running(state):
                        return False
        except Exception:
            return False  # Unreachable native host is not evidence of quiescence.
        for owner in (job.get("workerProcesses") or {}).values() if job else []:
            if any(worker_alive(process) for process in process_tree(owner)):
                return False
        self.store.release(job_id)
        return True

    async def evaluate(self, job, task):
        job_id = job["id"]
        plans = job.get("evaluationPlan") or pipeline.evaluation_plan(task)
        results = {key: dict(value) for key, value in job.get("datasetResults", {}).items()}
        for plan in plans:
            for dataset in plan["datasets"]:
                key = self.result_key(plan, dataset, plans)
                result = results.get(key)
                if result:
                    identity = runner.evaluation_identity(task, plan, dataset)
                    if result.get("configIdentity", identity) != identity:
                        del results[key]
                    else:
                        result["configIdentity"] = identity
        if results != job.get("datasetResults", {}):
            job = self.change(job_id, datasetResults=results)
        pending = [(plan, dataset) for plan in plans for dataset in plan["datasets"]
                   if results.get(self.result_key(plan, dataset, plans), {}).get("status") != "success"]
        if pending:
            selected_plan, selected_dataset = pending[0]
        else:
            selected_plan, selected_dataset = plans[0], None
        remaining = [dataset for plan, dataset in pending if plan == selected_plan]
        if not pending:
            self.change(job_id, stage="REPORT", evaluation=None, status="queued")
            return
        if not job.get("evaluation"):
            launch_plan = dict(selected_plan)
            feedback = job.get("gpuFeedback") or {}
            if feedback.get("min_gpu_memory_mib"):
                launch_plan["min_gpu_memory_mib"] = max(int(selected_plan.get("min_gpu_memory_mib", 0)),
                                                        feedback["min_gpu_memory_mib"])
            recovery = job.get("publicationRecovery") or {}
            if (recovery.get("dataset") == remaining[0] and recovery.get("schema") == selected_plan["schema"]
                    and recovery.get("attemptId") == job.get("attemptId")):
                launch_plan["recover_publication"] = True
            try:
                discovered = await asyncio.to_thread(runner.evaluation_resources, launch_plan, job.get("gpus"))
            except HarnessError as error:
                self.change(job_id, status="waiting_gpu", reason=str(error))
                return
            # One dataset per lease; deterministic FIFO queue prevents one paper hoarding all GPUs.
            available = [g for g in discovered["gpus"].split(",") if g]
            selected = None
            env_name = runner.method_env(task.method) or task.method
            for g in available or ["cpu"]:
                resources = ["method:" + task.method, "environment:" + env_name, "gpu:" + g]
                if self.store.acquire(resources, job_id, mode="read", dataset=remaining[0]):
                    selected = g
                    break
            if selected is None:
                self.change(job_id, status="waiting_gpu", reason="Waiting for a resource lease")
                return
            self.change(job_id, evaluationLaunching=True, pendingDataset=remaining[0])
            task.log("config/selection", setting="gpu", selected=selected, resources=discovered, feedback=feedback)
            try:
                task.set(evaluation=selected_plan)
                launch = asyncio.create_task(asyncio.to_thread(runner.launch_eval, task, selected,
                    datasets=[remaining[0]], attempt_id=job.get("attemptId"), evaluation=launch_plan))
                interrupted = False
                try:
                    info = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    # Persist the result before propagating cancellation. Pause
                    # can now stop this exact process instead of losing its PID.
                    interrupted = True
                    info = await launch
                info["processIdentity"] = info.get("processIdentity") or process_identity(info["pid"])
                task.set(phase=Phase.EVALUATE, status=Status.RUNNING, eval=info)
                self.store.update("jobs", job_id, evaluation=info, resources=resources, evaluationLaunching=False)
                if interrupted:
                    raise asyncio.CancelledError()
                job = self.change(job_id, evaluation=info, resources=resources, status="running")
            except BaseException:
                if not (self.store.get("jobs", job_id) or {}).get("evaluation"):
                    self.store.update("jobs", job_id, evaluationLaunching=False)
                    self.store.release(job_id)
                raise
        info = job["evaluation"]
        if task.state.get("eval") != info:
            task.set(eval=info)
        while not self.shutdown:
            current = self.store.get("jobs", job_id)
            if not current.get("authorized"):
                return
            if info.get("processIdentity") and process_identity(info["pid"]) not in (None, info["processIdentity"]):
                raise HarnessError("Experiment PID was reused; refusing process ownership")
            await self.poll_progress(current)
            state = await asyncio.to_thread(runner.eval_status, task.reload())
            if not state["running"] and info.get("processIdentity") and self.group_alive(info):
                state["running"] = True
            self.change(job_id, heartbeat=now(), evaluationStatus=state)
            for dataset, value in state.get("datasets", {}).items():
                if value.get("status") == "success":
                    observed_plan = (info.get("evaluation") or selected_plan)
                    results[self.result_key(observed_plan, dataset, plans)] = {**value,
                        "schema": observed_plan["schema"], "dataset_type": observed_plan.get("dataset_type", "UTS"),
                        "configIdentity": runner.evaluation_identity(task, observed_plan, dataset),
                        "dataset": dataset, "run_id": state.get("run_id")}
            if results != (self.store.get("jobs", job_id).get("datasetResults") or {}):
                self.change(job_id, datasetResults=results)
            if not state["running"]:
                break
            # Silence and low GPU utilization are observations, never kill conditions.
            await asyncio.sleep(1)
        if self.shutdown:
            return
        findings = pipeline.evaluation_findings(state)
        next_stage = "DIAGNOSE" if findings else "REPORT" if all(
            results.get(self.result_key(plan, dataset, plans), {}).get("status") == "success"
            for plan in plans for dataset in plan["datasets"]) else "EVALUATE"
        # Results, the completed attempt and lease release are one durable
        # transition. A crash cannot clear ownership before recording success.
        with self.store.transaction():
            self.change(job_id, stage=next_stage, datasetResults=results, evaluation=None,
                        resources=[], evaluationLaunching=False, diagnosticFindings=findings,
                        lastEvaluationResources=info.get("resources", {}),
                        evaluationFailed=not state.get("done"), status="queued", repairTarget="IMPLEMENT" if findings else None)
            for finding in findings:
                self.diagnose(job_id, {**finding, "action": "report_only"})
            self.store.release(job_id)
        task.set(phase=Phase(next_stage), status=Status.READY, eval=None, evaluation_findings=findings)
        if findings:
            self.route_runtime_failure(job_id, task, findings)
        else:
            self.change(job_id, publicationRecovery=None, gpuFeedback=None, environmentRepairs=0, environmentRecovery=None)

    def publish_metric(self, job, dataset, run_id, data, provisional):
        evaluation = (job.get("evaluation") or {}).get("evaluation") or load()["eval"]
        schema, dtype = evaluation["schema"], evaluation.get("dataset_type", "UTS")
        metric_id = f"{job['id']}:{dtype}:{schema}:{dataset}:{run_id}"
        prior = self.store.get("metrics", metric_id)
        if prior and prior.get("data") == data and prior.get("provisional") == provisional:
            return
        metric = self.store.put("metrics", {"id": metric_id, "jobId": job["id"], "dataset": dataset,
            "schema": schema, "dataset_type": dtype, "runId": run_id, "provisional": provisional, "data": data})
        self.store.event("metric.updated", jobId=job["id"], metric=metric)

    async def poll_progress(self, job):
        if not job.get("taskId"):
            return
        task = Task.open(job["taskId"])
        state = runner.eval_status(task)
        run_id = state.get("run_id")
        if not run_id:
            return
        datasets = (job.get("evaluation") or {}).get("datasets") or list(state.get("datasets", {}))
        for dataset in datasets:
            directory = runner.find_run(task.method, state.get("schema", load()["eval"]["schema"]), dataset,
                                        run_id=run_id, attempt_id=job.get("attemptId"))
            if directory is None:
                continue
            progress_path = directory / "progress.jsonl"
            cursor_id = f"progress:{job['id']}:{job.get('attemptId')}:{dataset}:{run_id}"
            cursor = self.store.get("cursors", cursor_id) or {}
            offset = int(cursor.get("offset", 0))
            pending = []
            if progress_path.exists():
                if progress_path.stat().st_size < offset:
                    offset = 0
                with open(progress_path) as source:
                    source.seek(offset)
                    while True:
                        start = source.tell()
                        line = source.readline()
                        if not line or not line.endswith("\n"):
                            source.seek(start)
                            break
                        try:
                            pending.append(json.loads(line))
                        except ValueError:
                            pass
                    offset = source.tell()
                with self.store.transaction():
                    for event in pending:
                        if not isinstance(event, dict) or "type" not in event:
                            continue
                        self.store.event("experiment." + event["type"], jobId=job["id"],
                                         attemptId=job.get("attemptId"), data=event)
                        payload = event.get("data") or {}
                        current = self.store.get("jobs", job["id"])
                        progress = dict(current.get("progress") or {})
                        progress.update(dataset=dataset, curve=event.get("curve"), stage=event["type"])
                        if "n_curves_scored" in payload:
                            progress["completed"] = payload["n_curves_scored"]
                        elif "n_curves_evaluated" in payload:
                            progress["completed"] = payload["n_curves_evaluated"]
                        if "total_curves" in payload:
                            progress["total"] = payload["total_curves"]
                        if "epoch" in payload:
                            progress["epochs"] = {"current": payload["epoch"], "total": payload.get("epochs")}
                        self.store.update("jobs", job["id"], progress=progress)
                        self.store.event("job.progress", jobId=job["id"], progress=progress)
                        if event["type"] == "curve_evaluated":
                            self.publish_metric(job, dataset, run_id, {
                                "summary": payload.get("summary", {}),
                                "n_curves_evaluated": payload.get("n_curves_evaluated"),
                                "n_curves_scored": payload.get("n_curves_scored"),
                                "total_curves": payload.get("total_curves"),
                                "diagnostics": payload.get("diagnostics", {})}, True)
                        elif event["type"] == "benchmark_updated":
                            self.store.event("benchmark.updated", jobId=job["id"], data=payload)
                    self.store.put("cursors", {"id": cursor_id, "offset": offset,
                        "path": workers.relative(progress_path), "jobId": job["id"]})
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "success" or not manifest.get("canonical_metrics_committed"):
                continue
            average_path = ROOT / manifest.get("avg_path", "")
            if not average_path.is_file():
                continue
            raw = average_path.read_bytes()
            expected = manifest.get("result_sha256", {}).get(manifest.get("avg_path"))
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                continue
            average = json.loads(raw)
            summary = {}
            for stable, protocol in manifest.get("metric_keys", {}).items():
                value = average.get(protocol, average.get(stable))
                if value is not None:
                    summary[stable] = value if isinstance(value, dict) else {"value": value}
            self.publish_metric(job, dataset, run_id, {"summary": summary,
                "n_curves_evaluated": manifest.get("n_curves_evaluated"),
                "n_curves_scored": manifest.get("n_curves_scored"),
                "total_curves": manifest.get("total_curves"),
                "diagnostics": manifest.get("diagnostics", {})}, False)

    async def close(self):
        self.shutdown = True
        if self.loop_task:
            self.loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.loop_task
        # Training processes intentionally survive service restarts; their identities and leases are durable.
        for worker in self.workers.values():
            worker.cancel()
        for turn in self.chat_workers:
            turn.cancel()
        await asyncio.gather(*self.workers.values(), *self.chat_workers, return_exceptions=True)
