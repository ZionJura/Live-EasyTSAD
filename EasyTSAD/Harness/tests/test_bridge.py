"""Connection and host ownership regressions without a model or installed host."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from tsad_harness import bridge
from tsad_harness.config import HarnessError


class FakeConfig:
    def __init__(self, *, startup_timeout=2):
        self.raw = {
            "interactive": {"port": 8767, "bridge_port": 8768,
                            "startup_timeout_seconds": startup_timeout},
            "llm": {"provider": "fixture", "model": "fixture-model", "base_url": "http://localhost:1",
                    "context_window": 8192, "max_output_tokens": 256},
            "budget": {"max_steps_per_run": 4, "max_wall_minutes_per_run": 1},
        }
        self.python_bin = Path(sys.executable)
        self.dsh_key_env = "FIXTURE_API_KEY"

    def __getitem__(self, key):
        return self.raw[key]

    def api_key(self):
        return "fixture-key"

    def network_env(self):
        return dict(os.environ)

    agent_env = network_env


class RemoteClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsad-bridge-")
        self.runtime = Path(self.tmp.name)
        self.children = []
        self.patches = [patch.object(bridge, "load", return_value=FakeConfig()),
                        patch.object(bridge, "_runtime_dir", return_value=self.runtime),
                        patch.dict(os.environ, {"TSAD_HARNESS_URL": "http://127.0.0.1:8767"})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=3)
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def sleeping_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        self.children.append(child)
        return child

    def test_read_timeout_does_not_launch_another_service(self):
        client = bridge.RemoteClient(token="fixture")
        with patch.object(client, "call", side_effect=httpx.ReadTimeout("slow existing server")), \
             patch.object(bridge.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(HarnessError, "same background service"):
                client.ensure()
        launch.assert_not_called()
        self.assertFalse((self.runtime / "service-launch.json").exists())

    def test_cancelled_connection_does_not_probe_or_spawn(self):
        cancel = threading.Event()
        cancel.set()
        client = bridge.RemoteClient(token="fixture")
        with patch.object(client, "call") as probe, patch.object(bridge.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(HarnessError, "cancelled"):
                client.ensure(cancel=cancel)
        probe.assert_not_called()
        launch.assert_not_called()

    def test_concurrent_ensure_launches_once_and_records_real_process_identity(self):
        clients = [bridge.RemoteClient(token="fixture") for _ in range(2)]
        first_probe = threading.Barrier(2)
        seen = threading.local()
        ready_at = [None]
        original_popen = subprocess.Popen

        def probe(_client, method, path, body=None):
            self.assertEqual((method, path), ("GET", "/health"))
            if not getattr(seen, "entered", False):
                seen.entered = True
                first_probe.wait(timeout=3)
                raise httpx.ConnectError("not listening")
            if ready_at[0] is None or time.monotonic() < ready_at[0]:
                raise httpx.ConnectError("still starting")
            return {"ready": True}

        def launch(*args, **kwargs):
            child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
            self.children.append(child)
            ready_at[0] = time.monotonic() + .05
            return child

        with patch.object(bridge.RemoteClient, "call", new=probe), \
             patch.object(bridge.subprocess, "Popen", side_effect=launch) as launched:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = [pool.submit(client.ensure) for client in clients]
                self.assertEqual([item.result(timeout=4) for item in pending], clients)
        self.assertEqual(launched.call_count, 1)
        record = json.loads((self.runtime / "service-launch.json").read_text())
        self.assertEqual(record["pid"], self.children[0].pid)
        self.assertEqual(record["processIdentity"], bridge._process_identity(self.children[0].pid))
        self.assertEqual(record["url"], clients[0].url)

    def test_a_live_startup_ledger_is_reused_until_health_is_ready(self):
        child = self.sleeping_child()
        client = bridge.RemoteClient(token="fixture")
        record = {"pid": child.pid, "processIdentity": bridge._process_identity(child.pid),
                  "url": client.url, "startedAt": time.time()}
        (self.runtime / "service-launch.json").write_text(json.dumps(record))
        with patch.object(client, "_probe", side_effect=[False, False, False, True]), \
             patch.object(bridge.subprocess, "Popen") as launch:
            self.assertIs(client.ensure(), client)
        launch.assert_not_called()
        self.assertIsNone(child.poll())

    def test_cancellation_while_waiting_preserves_the_owned_background_service(self):
        child = self.sleeping_child()
        client = bridge.RemoteClient(token="fixture")
        (self.runtime / "service-launch.json").write_text(json.dumps({
            "pid": child.pid, "processIdentity": bridge._process_identity(child.pid), "url": client.url}))
        cancel = threading.Event()
        probes = 0

        def unavailable():
            nonlocal probes
            probes += 1
            if probes == 4:
                cancel.set()
            return False

        with patch.object(client, "_probe", side_effect=unavailable), \
             patch.object(bridge.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(HarnessError, "cancelled"):
                client.ensure(cancel=cancel)
        launch.assert_not_called()
        self.assertIsNone(child.poll())

    def test_a_failed_launch_ledger_write_reaps_only_the_new_child(self):
        client = bridge.RemoteClient(token="fixture")
        child = self.sleeping_child()
        with patch.object(client, "_probe", return_value=False), \
             patch.object(bridge.subprocess, "Popen", return_value=child), \
             patch.object(Path, "write_text", side_effect=OSError("ledger unavailable")):
            with self.assertRaisesRegex(OSError, "ledger unavailable"):
                client.ensure()
        self.assertIsNotNone(child.poll())
        self.assertFalse((self.runtime / "service-launch.json").exists())

    def test_http_client_uses_route_timeouts_and_tls_only_for_https(self):
        response = Mock(status_code=200)
        response.json.return_value = {"ok": True}
        transport = Mock()
        transport.request.return_value = response
        transport.__enter__ = Mock(return_value=transport)
        transport.__exit__ = Mock(return_value=None)
        with patch.object(bridge.httpx, "Client", return_value=transport) as factory:
            client = bridge.RemoteClient(url="http://127.0.0.1:8767", token="fixture")
            self.assertEqual(client.call("GET", "/health"), {"ok": True})
            client.call("GET", "/jobs")
            client.call("POST", "/sessions", {"text": "hello"})
            secure = bridge.RemoteClient(url="https://harness.example", token="fixture")
            secure.call("GET", "/health")
        self.assertEqual([call.kwargs["timeout"] for call in factory.call_args_list], [8, 10, 120, 8])
        self.assertEqual([call.kwargs["verify"] for call in factory.call_args_list], [False, False, False, True])
        self.assertTrue(all(call.kwargs["trust_env"] is False for call in factory.call_args_list))
        self.assertEqual(transport.request.call_args_list[2].kwargs["json"], {"text": "hello"})


class HostStartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsad-host-start-")
        self.root = Path(self.tmp.name)
        self.cfg = FakeConfig(startup_timeout=.12)
        self.load_patch = patch.object(bridge, "load", return_value=self.cfg)
        self.load_patch.start()
        self.host = bridge.Bridge(self.root / "runtime", "http://127.0.0.1:8767", "fixture")

    async def asyncTearDown(self):
        await self.host.client.aclose()
        if self.host.log:
            self.host.log.close()
        self.load_patch.stop()
        self.tmp.cleanup()

    async def test_call_translates_transport_errors_without_leaking_details_or_retrying(self):
        for method in ("GET", "POST"):
            for error_type in (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                               httpx.PoolTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.HTTPError):
                with self.subTest(method=method, error=error_type.__name__):
                    request = AsyncMock(side_effect=error_type("private-token private-body"))
                    with patch.object(self.host.client, "request", new=request):
                        with self.assertRaises(HarnessError) as caught:
                            await self.host.call(method, "/sessions/fixture?token=private-token", {"text":"private-body"})
                    message = str(caught.exception)
                    self.assertIn("DSH " + method + " /sessions/fixture", message)
                    self.assertIn("dsh-host.log", message)
                    self.assertIn("timed out" if issubclass(error_type, httpx.TimeoutException) else error_type.__name__, message)
                    self.assertNotIn("private-token", message)
                    self.assertNotIn("private-body", message)
                    request.assert_awaited_once()

    async def test_call_preserves_connect_error_messages(self):
        for error, expected in (("DSH host has not started", "DSH host has not started"), ("", "DSH host unavailable")):
            with self.subTest(error=error):
                self.host.error = error
                request = AsyncMock(side_effect=httpx.ConnectError("private connection detail"))
                with patch.object(self.host.client, "request", new=request):
                    with self.assertRaises(HarnessError) as caught:
                        await self.host.call("POST", "/sessions", {})
                self.assertEqual(str(caught.exception), expected)
                request.assert_awaited_once()

    async def test_call_bounds_default_get_timeout_and_preserves_explicit_and_post_timeouts(self):
        response = Mock(status_code=200, content=b'{"ok":true}')
        response.json.return_value = {"ok": True}
        request = AsyncMock(return_value=response)
        with patch.object(self.host.client, "request", new=request):
            self.assertEqual(await self.host.call("GET", "/sessions/fixture"), {"ok": True})
            await self.host.call("get", "/sessions/fixture/events")
            await self.host.call("GET", "/health", timeout=.5)
            await self.host.call("POST", "/sessions", {})
        self.assertEqual([call.kwargs.get("timeout") for call in request.await_args_list], [8, 8, .5, None])
        self.assertNotIn("timeout", request.await_args_list[-1].kwargs)

    async def test_startup_uses_a_monotonic_deadline_and_rejects_ready_false(self):
        process = Mock()
        process.poll.return_value = None
        self.host.process = process
        self.cfg.raw["interactive"]["startup_timeout_seconds"] = 1.2
        clock = [100.0]
        probes = []

        async def unhealthy(method, path, *, timeout):
            probes.append((method, path, timeout))
            clock[0] += min(.3, timeout)
            return {"ready": False}

        async def pause(delay):
            clock[0] += delay

        with patch.object(bridge, "time", SimpleNamespace(monotonic=lambda: clock[0])), \
             patch.object(bridge, "asyncio", SimpleNamespace(sleep=pause)), \
             patch.object(self.host, "call", new=unhealthy), patch.object(self.host, "_launch") as launch:
            await self.host.start()
        self.assertIn("startup timed out", self.host.error)
        self.assertAlmostEqual(clock[0], 101.2)
        self.assertEqual(len(probes), 3)
        for method, path, timeout in probes:
            self.assertEqual((method, path), ("GET", "/health"))
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, .75)
        launch.assert_not_called()

    async def test_probe_timeout_is_bounded_and_a_later_ready_health_completes(self):
        process = Mock()
        process.poll.return_value = None
        self.host.process = process
        self.cfg.raw["interactive"]["startup_timeout_seconds"] = 1
        health = AsyncMock(side_effect=[httpx.ReadTimeout("not ready yet"), {"ready": True}])
        with patch.object(self.host, "call", new=health), patch.object(self.host, "_launch") as launch:
            await self.host.start()
        self.assertEqual(self.host.error, "")
        self.assertEqual(health.await_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] <= .75 for call in health.await_args_list))
        launch.assert_not_called()

    async def test_cancellation_waits_for_launch_ownership_before_returning(self):
        entered = threading.Event()
        release = threading.Event()
        owner_thread = threading.get_ident()
        launch_threads = []
        process = Mock()
        process.poll.return_value = None

        def launch(cfg):
            launch_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test did not release host launch")
            self.host.process = process

        with patch.object(self.host, "_launch", side_effect=launch):
            starting = asyncio.create_task(self.host.start())
            try:
                while not entered.is_set():
                    await asyncio.sleep(.005)
                starting.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(starting.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await starting
                self.assertIs(self.host.process, process)
                self.assertNotEqual(launch_threads, [owner_thread])
            finally:
                release.set()
                if not starting.done():
                    await asyncio.gather(starting, return_exceptions=True)

    async def test_launch_uses_the_pinned_node_and_runtime_environment(self):
        harness = self.root / "harness"
        host = harness / "dsh-host"
        (host / "node_modules/@deepseek-ai/dsh").mkdir(parents=True)
        node = host / "node_modules/node/bin/node"
        node.parent.mkdir(parents=True)
        node.touch()
        with patch.object(bridge, "HARNESS_DIR", harness), patch.object(bridge, "ROOT", self.root), \
             patch.object(bridge.subprocess, "Popen") as launch, \
             patch.object(bridge, "_process_identity", return_value="fixture-identity"):
            launch.return_value.pid = 12345
            self.host._launch(self.cfg)
        self.assertEqual(launch.call_args.args[0], [str(node), str(host / "launch.mjs")])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        environment = launch.call_args.kwargs["env"]
        self.assertEqual(environment["DSH_HOME"], str(self.host.runtime / "dsh"))
        self.assertEqual(environment["TSAD_MODEL"], "fixture-model")
        self.assertEqual(environment["TSAD_BRIDGE_TOKEN"], self.host.token)


class HostOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsad-host-ownership-")
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        self.harness = self.root / "harness"
        host = self.harness / "dsh-host"
        (host / "node_modules/@deepseek-ai/dsh").mkdir(parents=True)
        node = host / "node_modules/node/bin/node"
        node.parent.mkdir(parents=True)
        node.symlink_to("/usr/bin/python3")
        (host / "launch.mjs").write_text(
            "import json, os, signal, socket, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "runtime = Path(os.environ['DSH_HOME']).parent\n"
            "if len(sys.argv) == 1 or sys.argv[-1] != 'child':\n"
            "    subprocess.Popen([sys.executable, __file__, 'child'])\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    while True: time.sleep(1)\n"
            "listener = socket.socket()\n"
            "listener.bind(('127.0.0.1', int((runtime / 'port').read_text())))\n"
            "listener.listen()\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "(runtime / 'listener.json').write_text(json.dumps({'pid': os.getpid(), "
            "'port': listener.getsockname()[1]}))\n"
            "while True: time.sleep(1)\n")
        self.cfg = FakeConfig()
        self.cfg.agent_env = lambda: {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        self.patches = [patch.object(bridge, "load", return_value=self.cfg),
                        patch.object(bridge, "HARNESS_DIR", self.harness),
                        patch.object(bridge, "ROOT", self.root)]
        for item in self.patches:
            item.start()
        self.runtime.mkdir()
        (self.runtime / "port").write_text("0")
        self.host = bridge.Bridge(self.runtime, "http://127.0.0.1:1", "fixture")
        self.children = []
        self.controllers = []
        self.ledger = self.runtime / "dsh-host-launch.json"

    async def asyncTearDown(self):
        await self.host.close()
        if self.ledger.exists():
            record = json.loads(self.ledger.read_text())
            if (isinstance(record, dict) and isinstance(record.get("pid"), int)
                    and isinstance(record.get("processIdentity"), str)
                    and bridge._process_identity(record["pid"]) == record["processIdentity"]):
                with bridge.contextlib.suppress(ProcessLookupError):
                    os.killpg(record["pid"], signal.SIGKILL)
        for child in self.children:
            with bridge.contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
        for controller in self.controllers:
            with bridge.contextlib.suppress(ProcessLookupError):
                os.kill(controller, signal.SIGKILL)
            os.waitpid(controller, 0)
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    async def wait_for(self, condition):
        deadline = time.monotonic() + 15
        while not condition():
            if time.monotonic() >= deadline:
                log = self.runtime / "dsh-host.log"
                self.fail("fixture did not become ready: " + (log.read_text() if log.exists() else "no host log"))
            await asyncio.sleep(.02)

    def sleeping_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        return child

    async def test_killed_controller_host_group_is_recovered_before_rebinding(self):
        controller = os.fork()
        if controller == 0:
            try:
                os.setsid()
                self.host._launch(self.cfg)
                signal.pause()
            finally:
                os._exit(0)
        self.controllers.append(controller)
        await self.wait_for(lambda: self.ledger.exists() and (self.runtime / "listener.json").exists())
        record = json.loads(self.ledger.read_text())
        listener = json.loads((self.runtime / "listener.json").read_text())
        try:
            self.assertEqual(record["ownerPid"], controller)
            self.assertEqual(record["processIdentity"], bridge._process_identity(record["pid"]))
            self.assertEqual(record["runtime"], str(self.runtime.resolve()))
            os.kill(controller, signal.SIGKILL)
            os.waitpid(controller, 0)
            self.controllers.remove(controller)
            self.assertIsNotNone(bridge._process_identity(record["pid"]))
            (self.runtime / "port").write_text(str(listener["port"]))
            (self.runtime / "listener.json").unlink()
            await asyncio.to_thread(self.host._launch, self.cfg)
            await self.wait_for(lambda: (self.runtime / "listener.json").exists())
            replacement = json.loads((self.runtime / "listener.json").read_text())
            self.assertEqual(replacement["port"], listener["port"])
            self.assertNotEqual(replacement["pid"], listener["pid"])
            await self.wait_for(lambda: bridge._process_identity(listener["pid"]) is None)
            self.assertIsNone(bridge._process_identity(record["pid"]))
            self.assertEqual(json.loads(self.ledger.read_text())["pid"], self.host.process.pid)
            await self.host.close()
            self.assertIsNone(bridge._process_identity(self.host.process.pid))
        finally:
            with bridge.contextlib.suppress(ProcessLookupError):
                os.killpg(record["pid"], signal.SIGKILL)

    async def test_recovery_preserves_live_owner_mismatched_identity_and_other_runtime(self):
        child = self.sleeping_child()
        record = {"pid": child.pid, "processIdentity": bridge._process_identity(child.pid),
                  "ownerPid": os.getpid(), "ownerIdentity": bridge._process_identity(os.getpid()),
                  "runtime": str(self.runtime.resolve())}
        self.ledger.write_text(json.dumps(record))
        with self.assertRaisesRegex(HarnessError, "live controller"):
            self.host._launch(self.cfg)
        for changes in ({"processIdentity": "wrong"}, {"runtime": str(self.root / "other")},
                        {"pid": -1}, {"ownerIdentity": None}):
            with self.subTest(changes=changes):
                self.ledger.write_text(json.dumps({**record, **changes}))
                self.host._recover_host(self.ledger)
                self.assertIsNone(child.poll())
        self.ledger.write_text(json.dumps({**record, "ownerIdentity": "previous-owner"}))
        with patch.object(bridge.os, "getpgid", return_value=child.pid + 1), \
             patch.object(bridge.os, "killpg") as kill:
            self.host._recover_host(self.ledger)
            kill.assert_not_called()

    async def test_missing_or_invalid_record_never_signals_an_unrecorded_listener(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            self.host.port = listener.getsockname()[1]
            for content in (None, "{", "[]", "{}"):
                with self.subTest(content=content), patch.object(bridge.os, "killpg") as kill:
                    if content is not None:
                        self.ledger.write_text(content)
                    self.host._recover_host(self.ledger)
                    kill.assert_not_called()
                    self.assertGreater(listener.fileno(), 0)

    async def test_failed_ownership_write_reaps_new_launcher(self):
        with patch.object(Path, "write_text", side_effect=OSError("ledger unavailable")):
            with self.assertRaisesRegex(OSError, "ledger unavailable"):
                self.host._launch(self.cfg)
        self.assertIsNotNone(self.host.process.poll())
        self.assertFalse(self.ledger.exists())


if __name__ == "__main__":
    unittest.main()
