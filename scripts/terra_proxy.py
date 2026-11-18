#!/usr/bin/env python3
"""Small authenticated YYT proxy, bound to the configured private interface."""

import hmac
import json
import os
import secrets
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS = ("gpt-6.1-terra", "deepseek-v4-pro", "deepseek-v4.1-flash", "deepseek-v4.1-flash-1017")
UPSTREAM = "https://yyt.s-flag.com/v1"
MAX_BODY = 16 * 1024 * 1024
KEY_PATH = Path.home() / ".config" / "terra-proxy" / "api-key"
UPSTREAM_KEY = os.environ.get("yyt_api") or os.environ.get("YYT_API_KEY")
if not UPSTREAM_KEY:
    sys.exit("Set yyt_api or YYT_API_KEY before starting the proxy")
KEY_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
try:
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(fd, "w") as f:
        f.write("sk-terra-" + secrets.token_urlsafe(32) + "\n")
API_KEY = KEY_PATH.read_text().strip()
if not API_KEY:
    sys.exit("Proxy API key is empty")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, status, data):
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def authorized(self):
        expected = "Bearer " + API_KEY
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self.close_connection = True
            self.send_json(401, {"error": {"message": "Invalid API key"}})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "yyt"} for model in MODELS]})
        else:
            self.send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path not in ("/v1/responses", "/v1/chat/completions"):
            self.close_connection = True
            self.send_json(404, {"error": {"message": "Not found"}})
            return
        started = False
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0 or length > MAX_BODY:
                raise ValueError()
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or data.get("model") not in MODELS:
                raise ValueError()
        except (ValueError, json.JSONDecodeError):
            self.close_connection = True
            self.send_json(400, {"error": {"message": "Invalid request or unsupported model"}})
            return
        request = urllib.request.Request(UPSTREAM + self.path[3:], data=json.dumps(data).encode(), headers={
            "Authorization": "Bearer " + UPSTREAM_KEY,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if data.get("stream") else "application/json",
        })
        try:
            try:
                response = urllib.request.urlopen(request, timeout=300)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                self.close_connection = True
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            self.close_connection = True
            if not started:
                self.send_json(502, {"error": {"message": "Upstream connection failed"}})
            return

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    host = os.environ.get("TERRA_PROXY_HOST", "192.168.125.151")
    port = int(os.environ.get("TERRA_PROXY_PORT", "18765"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
