"""Verify scoped MCP startup and cleanup using real, small HTTP children."""
from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from tsad_harness import agent


class StartupCheckTests(unittest.TestCase):
    def test_worker_sessions_are_isolated_and_legacy_history_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(dsh_home=Path(directory))
            original = cfg.dsh_home / "sessions/workspace/prior"
            original.mkdir(parents=True)
            (original / "session.v4.jsonl").write_text("history")
            interactive = cfg.dsh_home / "sessions/workspace/chat"
            interactive.mkdir()
            (interactive / "session.v4.jsonl.zstd").write_bytes(b"compressed")
            root, compression = agent.worker_session_store(cfg, "prior")
            self.assertEqual(compression, "none")
            self.assertEqual((root / "workspace/prior/session.v4.jsonl").read_text(), "history")
            self.assertFalse((root / "workspace/chat").exists())
            self.assertTrue((original / "session.v4.jsonl").is_file())
            compressed_root, compression = agent.worker_session_store(cfg, "chat")
            self.assertEqual(compression, "zstd")
            self.assertNotEqual(root, compressed_root)
            self.assertTrue((compressed_root / "workspace/chat/session.v4.jsonl.zstd").is_file())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        script = self.root / "python-check"
        python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
        script.write_text(f"#!{python}\n" + '''import http.server,os,sys,time
mode=os.environ['CHECK_MODE']
if mode=='fail':
    print('fixture startup failure',flush=True)
    sys.exit(7)
if mode=='sleep':time.sleep(120)
endpoint=os.environ['TSAD_WORKER_MCP_HTTP_PATH']
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(405 if self.path==endpoint else 404)
        self.end_headers()
    def log_message(self,*args):pass
port=int(sys.argv[sys.argv.index('--http-port')+1])
http.server.HTTPServer(('127.0.0.1',port),Handler).serve_forever()
''')
        script.chmod(0o755)
        self.cfg = type("Config", (dict,), {})(dsh={"mcp_startup_timeout_ms": 5000})
        self.cfg.python_bin = script
        self.task = SimpleNamespace(method_dir=self.root / "methods/Example")
        self.env = {"PATH": os.environ.get("PATH", ""), "CHECK_MODE": "ok"}
        self.patches = [patch.object(agent, "ROOT", self.root),
                        patch.object(agent, "HARNESS_DIR", self.root / "harness"),
                        patch.object(agent, "relative", side_effect=lambda p: p.relative_to(self.root).as_posix()),
                        patch.object(agent, "_tool_environment", return_value=self.env)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.events = []

    def server(self, **kwargs):
        return agent._worker_mcp(self.cfg, self.task, "interface", write_enabled=False,
                                 deadline=time.monotonic() + 10, stage_token="test-stage",
                                 on_process=self.events.append, **kwargs)

    @property
    def directory(self):
        return self.task.method_dir / "reproduction/agents/interface"

    def assert_reaped(self):
        self.assertTrue(self.events)
        self.assertFalse(Path(f"/proc/{self.events[0]['pid']}").exists(), "MCP child was not reaped")
        ledger = json.loads((self.directory / "mcp-process.json").read_text())
        self.assertIn(ledger["status"], ("failed", "stopped"))

    def test_ready_endpoint_is_unique_and_server_lifetime_is_owned(self):
        with self.server() as process:
            self.assertTrue(Path(f"/proc/{process['pid']}").exists())
            self.assertEqual(process["role"], "interface")
            self.assertEqual(process["stageToken"], "test-stage")
            self.assertEqual(process["argv"][-2], "--http-port")
            parsed = urlsplit(process["url"])
            self.assertRegex(parsed.path, r"^/mcp/[a-f0-9]{32}$")
            connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=1)
            try:
                connection.request("GET", "/mcp")
                self.assertEqual(connection.getresponse().status, 404)
            finally:
                connection.close()
            ledger = json.loads((self.directory / "process.json").read_text())
            self.assertEqual(ledger["ownedProcesses"][0]["pid"], process["pid"])
            self.assertEqual(ledger["cwd"], ".")
        self.assertEqual(process["status"], "stopped")
        self.assert_reaped()

    def test_server_failure_reaps_child_and_records_failure(self):
        self.env["CHECK_MODE"] = "fail"
        with self.assertRaisesRegex(Exception, "server exited before readiness"):
            with self.server():
                self.fail("Failed server entered the CLI scope")
        self.assertEqual(self.events[-1]["status"], "failed")
        self.assert_reaped()

    def test_timeout_terminates_and_reaps_actual_child(self):
        self.env["CHECK_MODE"] = "sleep"
        self.cfg["dsh"]["mcp_startup_timeout_ms"] = 200
        with self.assertRaisesRegex(Exception, "startup or run budget"):
            with self.server():
                self.fail("Unready server entered the CLI scope")
        self.assert_reaped()

    def test_cancellation_terminates_and_reaps_actual_child(self):
        self.env["CHECK_MODE"] = "sleep"
        cancel = threading.Event()
        timer = threading.Timer(.2, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaisesRegex(Exception, "initialization cancelled"):
            with self.server(cancel=cancel):
                self.fail("Cancelled server entered the CLI scope")
        self.assert_reaped()

    def test_exception_before_cli_launch_stops_server(self):
        with self.assertRaisesRegex(RuntimeError, "launch preparation failed"):
            with self.server():
                raise RuntimeError("launch preparation failed")
        self.assert_reaped()

    def test_live_server_ledger_refuses_duplicate_start(self):
        with self.server() as process:
            with self.assertRaisesRegex(Exception, "already running"):
                with self.server():
                    self.fail("Launched a duplicate server")
            self.assertTrue(Path(f"/proc/{process['pid']}").exists())
        self.assert_reaped()

    def test_loopback_proxy_exclusions_are_explicit_and_preserve_direct_hosts(self):
        env = agent._loopback_env({"HTTPS_PROXY": "http://proxy.example:8080",
                                   "NO_PROXY": "packages.example", "no_proxy": "internal.example"})
        self.assertEqual(env["HTTPS_PROXY"], "http://proxy.example:8080")
        self.assertEqual(env["NO_PROXY"], env["no_proxy"])
        self.assertEqual(set(env["NO_PROXY"].split(",")),
                         {"packages.example", "internal.example", "127.0.0.1", "localhost", "::1"})


if __name__ == "__main__":
    unittest.main()
