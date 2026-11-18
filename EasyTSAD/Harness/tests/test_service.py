"""Domain/API regression with recorded model-independent host events."""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import httpx

from tsad_harness.service import API, create_app, paper_inputs
from tsad_harness.config import ROOT, HarnessError
from tsad_harness.store import Store


class FakeBridge:
    error = ""
    def __init__(self):
        self.calls = []
        self.messages = []
    async def call(self, method, path, body=None, *, timeout=None):
        self.calls.append((method, path, body))
        if path == "/sessions" and method == "POST":
            return {"sessionId": "root"}
        if path == "/sessions":
            return {"sessions": [{"sessionId": "root"}]}
        if path.startswith("/commands"):
            return {"commands": [{"name": "goal"}, {"name": "compact"}]}
        if path.startswith("/agents"):
            return {"agents": [{"sessionId": "child", "parentSessionId": "root", "continuable": True}]}
        if path.endswith("/messages"):
            self.messages.append(body)
            return {"accepted": True, "requestId": body["requestId"]}
        if path.endswith("/cancel"):
            return {"accepted": True}
        if path.startswith("/sessions/"):
            return {"sessionId": path.split("/")[2], "running": False, "history": [{"role": "assistant", "content": "你好"}]}
        return {"ok": True}
    async def close(self):
        pass


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bridge = FakeBridge()
        self.app = create_app(directory=Path(self.tmp.name), bridge=self.bridge, start_runtime=False)
        self.store = self.app.state.store
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://localhost")
        self.headers = {"Authorization": "Bearer " + self.app.state.token}
    async def asyncTearDown(self):
        await self.client.aclose()
        self.store.close()
        self.tmp.cleanup()
    async def request(self, method, path, body=None):
        return await self.client.request(method, API + path, json=body, headers=self.headers)
    async def test_slow_worker_log_read_does_not_block_health(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def read_snapshot(session_id):
            entered.set()
            release.wait(30)
            finished.set()
            return None
        session = next(route.endpoint for route in self.app.routes if route.path == API + "/sessions/{session_id}")
        health = next(route.endpoint for route in self.app.routes if route.path == API + "/health")
        with patch.object(self.app.state.scheduler, "worker_snapshot", side_effect=read_snapshot):
            pending = asyncio.create_task(session("slow"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 30))
                response = await asyncio.wait_for(health(), 10)
                self.assertTrue(response["ok"])
                self.assertFalse(finished.is_set())
            finally:
                release.set()
                await pending
    async def test_auth_and_cross_origin_write(self):
        self.assertEqual((await self.client.get(API + "/jobs")).status_code, 401)
        self.assertEqual((await self.client.get(API + "/jobs", headers={"Cookie": "tsad_session=" + self.app.state.token})).status_code, 401)
        self.assertEqual((await self.client.get(API + "/jobs", headers={"Authorization":self.app.state.token})).status_code, 401)
        self.assertEqual((await self.client.post(API + "/auth", json={"token":self.app.state.token}, headers=self.headers)).status_code, 404)
        self.assertEqual((await self.request("GET", "/jobs")).status_code, 200)
        response = await self.client.post(API + "/batches", json={"papers":["paper"]}, headers={**self.headers, "Origin":"https://evil.example"})
        self.assertEqual(response.status_code, 403)
    async def test_plain_chat_never_starts_job_unknown_command_rejected(self):
        response = await self.request("POST", "/sessions/root/messages", {"text":"你好，解释异常检测", "requestId":"chat1"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.bridge.messages), 1)
        self.assertEqual(self.store.all("jobs"), [])
        response = await self.request("POST", "/commands", {"sessionId":"root", "line":"/unknown-command"})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(len(self.bridge.messages), 1)
    async def test_batch_and_slash_idempotency(self):
        body = {"papers":["2401.12345", "2402.12345"], "requestId":"batch1", "afterJobs":["prior"],
                "evaluation":{"dataset_type":"MTS", "schema":"mts", "datasets":["SMD"]}}
        first = await self.request("POST", "/batches", body)
        second = await self.request("POST", "/batches", body)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(len(self.store.all("jobs")), 2)
        for job in self.store.all("jobs"):
            self.assertEqual(job["afterJobs"], ["prior"])
            self.assertEqual(job["evaluationPlan"][0]["dataset_type"], "MTS")
        conflict = await self.request("POST", "/batches", {**body, "papers":["different"]})
        self.assertEqual(conflict.status_code, 400)
        slash = await self.request("POST", "/commands", {"line":"/reproduce 2501.11111 2502.22222", "requestId":"slash"})
        self.assertEqual(slash.status_code, 200, slash.text)
        self.assertEqual(len(slash.json()["jobs"]), 2)
    async def test_stop_reply_does_not_cancel_training(self):
        self.store.put("jobs", {"id":"training", "status":"running", "authorized":True})
        response = await self.request("POST", "/sessions/child/cancel", {})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get("jobs", "training")["status"], "running")
        self.assertEqual(self.bridge.calls[-1][1], "/sessions/child/cancel")
    async def test_recovered_legacy_success_is_visible_in_scores(self):
        self.store.put("jobs", {"id":"legacy", "datasetResults":{"A":{
            "status":"success", "dataset":"A", "schema":"naive", "dataset_type":"UTS",
            "run_id":"prior", "metrics_summary":{"auprc":{"value":.5}}}}})
        response = await self.request("GET", "/jobs/legacy/scores")
        self.assertEqual(response.status_code, 200)
        metric = response.json()["metrics"][0]
        self.assertFalse(metric["provisional"])
        self.assertEqual(metric["runId"], "prior")
        self.assertEqual(metric["data"]["summary"]["auprc"]["value"], .5)

    async def test_slash_reproduction_and_resume_preserve_gpu_selection(self):
        response = await self.request("POST", "/commands", {
            "line": "/reproduce 2501.12345", "requestId": "gpu-command", "gpus": "2,3"})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["jobs"][0]
        self.assertEqual(self.store.get("jobs", job_id)["gpus"], "2,3")
        control = AsyncMock(return_value={"id": job_id})
        with patch.object(self.app.state.scheduler, "control", control):
            response = await self.request("POST", "/commands", {
                "line": "/resume " + job_id, "gpus": "1"})
        self.assertEqual(response.status_code, 200, response.text)
        control.assert_awaited_once_with(job_id, "resume", {"gpus": "1"})
    async def test_supervisor_cannot_control_and_scientific_diagnostics_report_only(self):
        self.store.put("jobs", {"id":"job", "method":"Method", "status":"running", "authorized":True})
        body = {"role":"supervisor", "jobId":"job", "arguments":{"jobId":"job"}}
        r = await self.client.post("/internal/tools/stop", json=body, headers=self.headers)
        self.assertEqual(r.status_code, 409)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / "adapter.py"
            source.write_text("def align(scores):\n    return scores[1:]\n")
            body["arguments"].update(category="leakage", action="repair",
                evidence=[{"file":str(source.relative_to(ROOT)), "line":2}])
            r = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["action"], "report_only")
            body["arguments"].update(category="alignment")
            r = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
            self.assertEqual(r.json()["action"], "repair")
            body["arguments"].update(evidence=[])
            r = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
            self.assertEqual(r.json()["action"], "report_only")
    async def test_tool_arguments_accept_objects_and_json_object_strings(self):
        self.store.put("jobs", {"id":"job", "status":"running"})
        for arguments in ({"jobId":"job"}, json.dumps({"jobId":"job"})):
            with self.subTest(arguments=arguments):
                response = await self.client.post("/internal/tools/status",
                    json={"arguments":arguments}, headers=self.headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["id"], "job")
                response = await self.client.post("/internal/tools/stop",
                    json={"arguments":arguments, "role":"supervisor"}, headers=self.headers)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("cannot control jobs", response.json()["error"])

    async def test_tool_arguments_reject_invalid_values_without_recursive_decoding(self):
        for arguments in (None, [], 1, True, "", "{", "[]", "null", "1", "true", json.dumps(json.dumps({}))):
            with self.subTest(arguments=arguments):
                response = await self.client.post("/internal/tools/diagnose",
                    json={"arguments":arguments}, headers=self.headers)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("Tool arguments must be a JSON object", response.json()["error"])

    async def test_bound_goal_cannot_complete_before_export(self):
        self.store.put("jobs", {"id":"job", "status":"running"})
        r = await self.client.post("/internal/tools/complete", json={"arguments":{"jobId":"job"}}, headers=self.headers)
        self.assertEqual(r.status_code, 409)

    async def test_native_write_lease_only_allows_implementation_and_returns_scoped_roots(self):
        base = ROOT / "lease-test-fixture"
        task = SimpleNamespace(method_dir=base / "method", dir=base / "task")
        self.store.put("jobs", {"id":"job", "method":"Method", "taskId":"task", "status":"running",
            "stage":"IMPLEMENT", "authorized":True})
        self.assertTrue(self.store.acquire(["method:Method"], "job", mode="write"))
        body = {"role":"implementer", "jobId":"job", "arguments":{"operation":"check_write_lease"}}
        with patch("tsad_harness.service.Task.open", return_value=task):
            response = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
            self.assertEqual(response.json(), {"leaseGranted":True,
                "allowedMethodDir":str(task.method_dir.relative_to(ROOT)),
                "allowedEnvironmentDir":str((task.method_dir / "environment").relative_to(ROOT)),
                "allowedTaskDir":str(task.dir.relative_to(ROOT))})
            for stage in ("FETCH", "RESEARCH", "DIAGNOSE", "EVALUATE", "REPORT"):
                self.store.update("jobs", "job", stage=stage)
                response = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
                self.assertFalse(response.json()["leaseGranted"], stage)
            self.store.update("jobs", "job", stage="IMPLEMENT")
            body["role"] = "diagnoser"
            response = await self.client.post("/internal/tools/diagnose", json=body, headers=self.headers)
            self.assertFalse(response.json()["leaseGranted"])
    async def test_cli_sessions_messages_events_and_cancel_bypass_host(self):
        self.store.put("jobs", {"id":"job", "method":"Method", "status":"running", "stage":"RESEARCH"})
        self.store.put("bindings", {"id":"worker", "sessionId":"worker", "jobId":"job", "role":"interface",
            "transport":"cli", "status":"running", "title":"Method: interface"})
        state = {"sessionId":"worker", "jobId":"job", "role":"interface", "transport":"cli", "status":"running",
            "events":[{"type":"assistant/message", "text":"first"}, {"type":"assistant/message", "text":"second"}],
            "history":[{"role":"assistant", "content":"worker history"}]}
        scheduler = self.app.state.scheduler
        with patch.object(scheduler, "worker_snapshot", side_effect=lambda sid: state if sid == "worker" else None), \
             patch.object(scheduler, "worker_message", new=AsyncMock(return_value={"accepted":True,"delivery":"next_role_turn"})) as message, \
             patch.object(scheduler, "worker_cancel", new=AsyncMock(return_value={"accepted":True})) as cancel:
            listed = (await self.request("GET", "/sessions")).json()["sessions"]
            self.assertIn("worker", [row.get("sessionId") for row in listed])
            response = await self.request("GET", "/sessions/worker")
            self.assertEqual(response.json()["history"], state["history"])
            self.assertTrue(response.json()["continuable"])
            before = len(self.bridge.calls)
            response = await self.request("GET", "/agents?sessionId=worker")
            self.assertEqual(response.json()["agents"][0]["transport"], "cli")
            response = await self.request("GET", "/commands?sessionId=worker")
            self.assertIn("ps", [row["name"] for row in response.json()["commands"]])
            self.assertNotIn("goal", [row["name"] for row in response.json()["commands"]])
            response = await self.request("GET", "/sessions/worker/events?after=1&epoch=worker")
            self.assertEqual(response.json()["events"], state["events"][1:])
            self.assertEqual(response.json()["cursor"], 2)
            state.update(eventsStart=500, eventCursor=502, eventEpoch="events-path")
            response = await self.request("GET", "/sessions/worker/events?after=501&epoch=events-path")
            self.assertEqual(response.json(), {"events":state["events"][1:], "cursor":502, "epoch":"events-path"})
            response = await self.request("POST", "/sessions/worker/messages", {"text":"check alignment","requestId":"followup","mode":"steer"})
            self.assertEqual(response.status_code, 200, response.text)
            message.assert_awaited_once_with("worker", "check alignment", "followup")
            response = await self.request("POST", "/sessions/worker/cancel", {})
            cancel.assert_awaited_once_with("worker")
            self.assertEqual(self.store.get("jobs", "job")["status"], "running")
            self.assertEqual(len(self.bridge.calls), before)

    async def test_native_session_event_cursor_forwarding(self):
        response = await self.request("GET", "/sessions/root/events?after=8&epoch=abc")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.bridge.calls[-1][1], "/sessions/root/events?after=8&epoch=abc")
        response = await self.request("GET", "/sessions/root?beforeSeq=40&maxMessages=10")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.bridge.calls[-1][1], "/sessions/root?beforeSeq=40&maxMessages=10")

    async def test_host_unavailable_keeps_cli_agents_and_jobs_visible(self):
        self.store.put("jobs", {"id":"job", "method":"Method", "status":"running"})
        self.store.put("bindings", {"id":"worker", "sessionId":"worker", "jobId":"job", "role":"interface",
            "transport":"cli", "status":"running"})
        with patch.object(self.bridge, "call", new=AsyncMock(side_effect=HarnessError("native host unavailable"))):
            for path, key in [("/sessions", "sessions"), ("/agents?sessionId=root", "agents")]:
                response = await self.request("GET", path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()[key][0]["sessionId"], "worker")
                self.assertEqual(response.json()["runtimeError"], "native host unavailable")
            self.assertEqual((await self.request("GET", "/jobs")).json()["jobs"][0]["id"], "job")
            response = await self.request("POST", "/sessions", {})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("native host unavailable", response.text)

    async def test_job_parameters_read_actual_method_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            config = base / "config.toml"
            config.write_text("[Model_Params]\nwindow = 48\nbatch_size = 32\n")
            task = SimpleNamespace(method_dir=base, dir=base)
            self.store.put("jobs", {"id":"job", "method":"Method", "taskId":"task", "status":"running"})
            with patch("tsad_harness.service.Task.open", return_value=task):
                response = await self.request("GET", "/jobs/job")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["parameters"], {"Model_Params":{"window":48,"batch_size":32}})
                self.assertEqual(response.json()["parametersPath"], str(config.relative_to(ROOT)))

    async def test_processes_and_logs_are_shared_with_ps_and_sanitize_scripts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            script, log = base / "install.sh", base / "install.log"
            script.write_text("export TEST_API_KEY='test-secret-value'\nuv sync --locked\nhelper --password 'literal-script-key' https://user:literal-url-key@example.com\n")
            log.write_text("Bearer another-credential\nTEST_API_KEY=test-secret-value\ninstallation ready\n")
            self.store.put("jobs", {"id":"proc-job", "method":"Method", "stage":"RESEARCH", "status":"running",
                "branches":{"environment":{"status":"running"}}, "progress":{"completed":2,"total":5},
                "workerProcesses":{"environment-install":{"pid":123,"processIdentity":"owned","kind":"environment",
                    "status":"running", "argv":["uv", "--api-key", "secret-argument", "sync"],
                    "cwd":".","log":str(log.relative_to(ROOT)),"script":str(script.relative_to(ROOT))},
                    "outside":{"pid":124,"status":"finished","script":"/etc/passwd","argv":[]}}})
            with patch("tsad_harness.service.process_identity", return_value="owned"), patch.dict("os.environ", {"TEST_API_KEY":"test-secret-value"}):
                response = await self.request("GET", "/processes?jobId=proc-job")
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["processes"][0]["status"], "running")
                self.assertEqual(data["processes"][0]["argv"][2], "[redacted]")
                self.assertIn("uv sync --locked", data["processes"][0]["scriptContent"])
                self.assertNotIn("test-secret-value", response.text)
                self.assertNotIn("literal-script-key", response.text)
                self.assertNotIn("literal-url-key", response.text)
                self.assertNotIn("scriptContent", data["processes"][1])
                slash = await self.request("POST", "/commands", {"line":"/ps proc-job"})
                self.assertEqual(slash.json(), data)
                detail = (await self.request("GET", "/jobs/proc-job")).json()
                self.assertEqual(detail["processes"], data["processes"])
                self.assertEqual(detail["progress"]["completed"], 2)
                logs = await self.request("GET", "/jobs/proc-job/logs")
                self.assertIn("installation ready", logs.json()["text"])
                self.assertNotIn("test-secret-value", logs.text)
                self.assertNotIn("another-credential", logs.text)

    async def test_owned_worker_tool_servers_are_observable_in_processes_and_logs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            capability = "0123456789abcdef" * 2
            endpoint = "http://127.0.0.1:43210/mcp/" + capability
            log, script = base / "worker-tools.log", base / "worker-tools.py"
            log.write_text("scoped MCP server ready " + endpoint + "\n")
            script.write_text("api_key = 'private-worker-key'\n# scoped tools server\nendpoint = '" + endpoint + "'\n")
            self.store.put("jobs", {"id":"worker-job", "method":"Method", "status":"running",
                "workerProcesses":{"hyperparameters":{"pid":123,"processIdentity":"cli-identity",
                    "kind":"agent", "status":"running", "argv":["dsh","--profile","worker.yml"],
                    "ownedProcesses":[{"pid":124,"processIdentity":"tools-identity","kind":"worker-tools",
                        "role":"hyperparameters","status":"running","argv":["python","worker-tools.py","--endpoint",endpoint],
                        "cwd":".","log":str(log.relative_to(ROOT)),"script":str(script.relative_to(ROOT)),
                        "url":endpoint}]}}})
            self.store.put("bindings", {"id":"worker", "sessionId":"worker", "jobId":"worker-job",
                "role":"hyperparameters", "transport":"cli", "process":{"url":endpoint}})
            self.store.event("worker.process", jobId="worker-job", process={"ownedProcesses":[{"url":endpoint}]})
            with patch("tsad_harness.service.process_identity", side_effect=lambda pid: {123:"cli-identity",124:"tools-identity"}.get(pid)):
                response = await self.request("GET", "/processes?jobId=worker-job")
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(len(data["processes"]), 2)
                tools = next(row for row in data["processes"] if row["processKind"] == "worker-tools")
                self.assertEqual(tools["kind"], "agent")
                self.assertEqual(tools["role"], "hyperparameters:worker-tools")
                self.assertEqual(tools["parentPid"], 123)
                self.assertEqual(tools["status"], "running")
                self.assertEqual(tools["log"], str(log.relative_to(ROOT)))
                self.assertEqual(tools["url"], "http://127.0.0.1:43210/mcp/[redacted]")
                self.assertIn("scoped tools server", tools["scriptContent"])
                self.assertNotIn("private-worker-key", response.text)
                self.assertNotIn(capability, response.text)
                self.assertEqual((await self.request("POST", "/commands", {"line":"/ps worker-job"})).json(), data)
                logs = await self.request("GET", "/jobs/worker-job/logs")
                self.assertIn("scoped MCP server ready", logs.json()["text"])
                self.assertNotIn(capability, logs.text)
                for path in ["/jobs", "/jobs/worker-job", "/snapshot", "/sessions", "/agents?sessionId=worker"]:
                    visible = await self.request("GET", path)
                    self.assertEqual(visible.status_code, 200, visible.text)
                    self.assertNotIn(capability, visible.text)
                    self.assertIn("/mcp/[redacted]", visible.text)
                command = await self.request("POST", "/commands", {"line":"/tasks"})
                self.assertNotIn(capability, command.text)
                event_route = next(route for route in self.app.routes if getattr(route, "path", "") == API + "/events")
                stream = await event_route.endpoint(SimpleNamespace(headers={}, is_disconnected=AsyncMock(side_effect=[False, True])))
                event_text = "".join([packet async for packet in stream.body_iterator])
                self.assertNotIn(capability, event_text)
                self.assertIn("/mcp/[redacted]", event_text)
                self.assertEqual(self.store.get("jobs", "worker-job")["workerProcesses"]["hyperparameters"]["ownedProcesses"][0]["url"], endpoint)

    async def test_report_returns_canonical_references_without_markdown(self):
        self.store.put("jobs", {"id":"result-job", "taskId":"task"})
        data = {"datasets":[{"dataset":"UCR", "avg_path":"Results/Evals/Method/naive/UCR/avg.json",
            "all_path":"Results/Evals/Method/naive/UCR/all.json", "avg":{"event_f1":.4}, "all":{"001":{"event_f1":.4}}}]}
        with patch("tsad_harness.service.Task.open", return_value=object()) as opened, \
             patch("tsad_harness.report.results_view", return_value=data) as view:
            response = await self.request("GET", "/jobs/result-job/report")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), {"jobId":"result-job", **data})
            opened.assert_called_once_with("task")
            view.assert_called_once()

    async def test_browser_pages_are_not_served(self):
        for path in ("/", "/docs", "/redoc", "/openapi.json", "/docs/how-it-works.html", "/docs/deepseek-harness-guide.html", "/benchmark.json", API + "/docs/deepseek-harness-guide.html"):
            response = await self.client.get(path, headers=self.headers)
            self.assertEqual(response.status_code, 404, path)
        response = await self.request("GET", "/health")
        self.assertEqual(response.status_code, 200, response.text)

    async def test_explicit_port_is_used_for_the_host_callback(self):
        with tempfile.TemporaryDirectory() as directory, patch("tsad_harness.service.Bridge") as bridge:
            app = create_app(directory=Path(directory), start_runtime=False, port=9876)
            self.assertEqual(bridge.call_args.args[1], "http://127.0.0.1:9876")
            app.state.store.close()

    async def test_snapshot_and_event_replay(self):
        a = self.store.event("metric.updated", jobId="x", value=.5)
        b = self.store.event("metric.updated", jobId="x", value=.6)
        r = await self.request("GET", "/snapshot")
        self.assertEqual(r.json()["eventCursor"], b)
        self.assertEqual([e["id"] for e in self.store.events(a)], [b])


class StoreTests(unittest.TestCase):
    def test_leases_restart_and_transaction_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = Store(path)
            self.assertTrue(store.acquire(["gpu:0", "method:A"], "a"))
            self.assertFalse(store.acquire(["gpu:1", "method:A"], "b"))
            self.assertNotIn("gpu:1", [l["resource"] for l in store.leases()])
            store.close()
            store = Store(path)
            self.assertFalse(store.acquire(["gpu:0"], "b"))
            store.release("a")
            self.assertTrue(store.acquire(["gpu:0"], "b"))
            store.close()
    def test_result_view_reads_native_avg_and_all_without_writing(self):
        from tsad_harness import report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "Results/Evals/Method/naive/UCR"
            data_dir.mkdir(parents=True)
            average = {"event-based f1":{"f1":.4,"precision":.8}, "AUPRC":.25}
            per_curve = {"001":{"event-based f1":{"f1":.4,"precision":.8},"AUPRC":.25}}
            (data_dir / "avg.json").write_text(json.dumps(average))
            (data_dir / "all.json").write_text(json.dumps(per_curve))
            before = {str(path):path.read_bytes() for path in root.rglob("*") if path.is_file()}
            task = SimpleNamespace(method="Method", method_dir=root / "EasyTSAD/Methods/Method")
            with patch.object(report, "ROOT", root):
                result = report.results_view(task)
            self.assertEqual(result["datasets"][0]["avg"], average)
            self.assertEqual(result["datasets"][0]["all"], per_curve)
            self.assertEqual(result["datasets"][0]["all_path"], "Results/Evals/Method/naive/UCR/all.json")
            self.assertEqual(before, {str(path):path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_result_view_reads_successful_legacy_metrics(self):
        from tsad_harness import report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "LiveLeaderBoard/results/runs/Method/naive/UCR/prior"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status":"success", "schema":"naive", "dataset":"UCR", "run_id":"prior"}))
            (run / "metrics.json").write_text(json.dumps({"summary":{"auprc":{"value":.5}}, "per_curve":{"001":{}}}))
            task = SimpleNamespace(method="Method", method_dir=root / "EasyTSAD/Methods/Method")
            with patch.object(report, "ROOT", root):
                result = report.results_view(task)
            self.assertEqual(result["datasets"][0]["run_id"], "prior")
            self.assertEqual(result["datasets"][0]["avg"]["auprc"]["value"], .5)

    def test_paper_multiline_title_and_ids(self):
        self.assertEqual(paper_inputs("2401.12345 2402.12345\nA paper with spaces"), ["2401.12345", "2402.12345", "A paper with spaces"])

if __name__ == "__main__":
    unittest.main()

class StartupConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_benchmark_recovery_does_not_delay_runtime(self):
        entered, release = threading.Event(), threading.Event()
        def recover():
            entered.set()
            release.wait(10)
        bridge = FakeBridge()
        bridge.start = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory=Path(directory), bridge=bridge)
            app.state.scheduler.start = AsyncMock()
            with patch('LiveLeaderBoard.pipeline.export.recover_pending', side_effect=recover):
                async with app.router.lifespan_context(app):
                    try:
                        for _ in range(100):
                            if entered.is_set() and bridge.start.await_count:
                                break
                            await asyncio.sleep(.01)
                        self.assertTrue(entered.is_set())
                        bridge.start.assert_awaited_once()
                        app.state.scheduler.start.assert_awaited_once()
                        self.assertFalse(release.is_set())
                    finally:
                        release.set()
