"""Real PTY acceptance without a model or an experiment. Run against prepared UI."""
from __future__ import annotations

import fcntl
import http.server
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import tempfile
import termios
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
calls = []
messages = []
cold = bool(os.environ.get("TSAD_PTY_COLD"))


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.removeprefix("/api/harness/v1")
        calls.append(("GET", path, {}))
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            return
        if path == "/health":
            result = {"ok": True, "connected": not cold, "runtimeError": "fixture host starting" if cold else ""}
        elif path == "/sessions":
            result = {"sessions": [{"sessionId": "pty-root"}]}
        elif "/events?" in path:
            result = {"events": [], "cursor": 0, "epoch": "fixture"}
        elif path == "/sessions/pty-root":
            result = {"status": "idle", "running": False, "history": list(messages)}
        elif path.startswith("/commands?"):
            result = {"commands": []}
        else:
            result = {}
        self.send_json(result)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        path = self.path.removeprefix("/api/harness/v1")
        calls.append(("POST", path, body))
        if path.endswith("/messages"):
            messages.append({"id": str(len(messages)), "role": "user", "content": body["text"]})
        self.send_json({"accepted": True})

    def send_json(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 110, 0, 0))
    output = bytearray()
    env = {**os.environ, "TERM": "xterm-256color", "TSAD_HARNESS_TOKEN": "fixture-token",
           "TSAD_HARNESS_URL": f"http://127.0.0.1:{server.server_port}"}
    started = time.monotonic()
    process = subprocess.Popen([str(ROOT / "scripts/harness"), "tui", "--resume", "latest", "--refresh", ".1"],
                               cwd=ROOT, env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)

    def pump(seconds=.2):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], min(.05, max(0, deadline - time.monotonic())))
            if ready:
                try:
                    chunk = os.read(master, 65536)
                    if not chunk:
                        return
                    output.extend(chunk)
                except OSError:
                    return

    def until(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline and process.poll() is None:
            pump(.05)
        assert predicate(), output.decode(errors="replace")[-3000:]

    try:
        until(lambda: b"HARNESS" in output)
        first_frame = time.monotonic() - started
        if cold:
            until(lambda: any(path == "/health" for _, path, _ in calls))
            os.write(master, "后台启动时也可编辑".encode())
            until(lambda: "后台启动时也可编辑".encode() in output)
            os.write(master, b"\x11")
            until(lambda: process.poll() is not None, 3)
            assert process.returncode == 0
            assert not any(method == "POST" for method, _, _ in calls)
            assert first_frame <= 2, first_frame
            print(json.dumps({"first_frame_seconds": round(first_frame, 3), "cold_host_editable": True, "exit_code": process.returncode}))
            return
        until(lambda: any(path == "/sessions/pty-root" for _, path, _ in calls))
        # Fragmented bracketed paste, explicit newline and Chinese input.
        os.write(master, "\x1b[200~你好，原版输入框\n第二行\x1b[201~".encode())
        pump()
        os.write(master, b"\x1b[13;2u")
        os.write(master, "换行完成".encode())
        os.write(master, b"\r")
        until(lambda: bool(messages))
        assert messages[0]["content"] == "你好，原版输入框\n第二行\n换行完成", messages
        # Focus events and a resize must not become message contents or crash.
        os.write(master, b"\x1b[O\x1b[I")
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 82, 0, 0))
        os.write(master, "焦点正常\r".encode())
        until(lambda: len(messages) == 2)
        assert messages[1]["content"] == "焦点正常", messages
        os.write(master, b"\x0b")
        until(lambda: any(path.endswith("/cancel") for _, path, _ in calls))
        assert not any("/stop" in path for _, path, _ in calls)
        quit_at = time.monotonic()
        os.write(master, b"\x11")
        until(lambda: process.poll() is not None, 3)
        assert process.returncode == 0
        assert b"Traceback" not in output and b"TypeError" not in output
        result = {"first_frame_seconds": round(first_frame, 3), "quit_seconds": round(time.monotonic()-quit_at, 3),
                  "messages": len(messages), "focus_resize": True, "exit_code": process.returncode}
        print(json.dumps(result))
        assert first_frame <= 2, result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    run()
