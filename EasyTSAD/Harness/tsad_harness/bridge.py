"""Private HTTP client for the pinned, persistent DSH host."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import secrets
import signal
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from .config import HARNESS_DIR, ROOT, HarnessError, load
from .process_identity import process_identity as _process_identity


def _runtime_dir() -> Path:
    """Locate controller state without importing the API and scheduler into TUI."""
    configured = load().raw.get("interactive", {}).get("runtime_dir", "runtime/harness-workbench")
    path = Path(os.environ.get("TSAD_HARNESS_RUNTIME", configured)).expanduser()
    return path if path.is_absolute() else ROOT / path


class Bridge:
    def __init__(self, runtime: Path, control_url: str, control_token: str):
        self.runtime = runtime
        self.port = int(load().raw.get("interactive", {}).get("bridge_port", 8768))
        self.token = secrets.token_urlsafe(32)
        self.control_url, self.control_token = control_url, control_token
        self.process = None
        self.log = None
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=45, verify=False,
            trust_env=False, headers={"Authorization": "Bearer " + self.token})
        self.error = "DSH host has not started"
        self._start_lock = asyncio.Lock()

    async def start(self):
        async with self._start_lock:
            if self.process is not None and self.process.poll() is None and not self.error:
                return
            await self._start()

    async def _start(self):
        cfg = load()
        self.error = "DSH host is starting"
        deadline = time.monotonic() + float(cfg.raw.get("interactive", {}).get("startup_timeout_seconds", 180))
        try:
            if self.process is None or self.process.poll() is not None:
                launching = asyncio.create_task(asyncio.to_thread(self._launch, cfg))
                try:
                    await asyncio.shield(launching)
                except asyncio.CancelledError:
                    # A filesystem operation cannot be cancelled in its thread.
                    # Finish ownership transfer before close() reaps the host.
                    await launching
                    raise
            while time.monotonic() < deadline:
                if self.process is None:
                    return
                if self.process.poll() is not None:
                    raise HarnessError(f"DSH host exited ({self.process.returncode}); see {self.runtime / 'dsh-host.log'}")
                try:
                    health = await self.call("GET", "/health", timeout=min(.75, max(.01, deadline - time.monotonic())))
                    if health.get("ready", True):
                        self.error = ""
                        return
                except (httpx.HTTPError, HarnessError):
                    pass
                await asyncio.sleep(min(.2, max(0, deadline - time.monotonic())))
            raise HarnessError("DSH host startup timed out; see dsh-host.log")
        except Exception as error:
            self.error = str(error)

    def _launch(self, cfg):
        host = HARNESS_DIR / "dsh-host"
        if not (host / "node_modules" / "@deepseek-ai" / "dsh").exists():
            self.error = "Install the pinned host first: cd EasyTSAD/Harness/dsh-host && npm ci"
            return
        env = cfg.agent_env()
        env.update(TSAD_BRIDGE_PORT=str(self.port), TSAD_BRIDGE_TOKEN=self.token,
                       TSAD_CONTROL_URL=self.control_url, TSAD_CONTROL_TOKEN=self.control_token,
                       TSAD_MODEL_PROVIDER=cfg["llm"]["provider"], TSAD_MODEL=cfg["llm"]["model"],
                       TSAD_MODEL_BASE_URL=cfg["llm"]["base_url"], TSAD_MODEL_API_KEY=cfg.api_key(),
                       TSAD_MODEL_API_KEY_ENV=cfg.dsh_key_env,
                       TSAD_MODEL_CONTEXT_WINDOW=str(cfg["llm"]["context_window"]),
                       TSAD_MODEL_MAX_TOKENS=str(cfg["llm"]["max_output_tokens"]),
                       TSAD_MODEL_API="openai-responses" if cfg["llm"].get("wire_api", "responses") == "responses" else "openai-completions",
                       DSH_HOME=str(self.runtime / "dsh"), TSAD_ROOT=str(ROOT),
                       TSAD_MODEL_PERMITS_DIR=str(self.runtime / "model-permits"),
                       TSAD_MAX_MODEL_CONCURRENCY=str(cfg.raw.get("interactive", {}).get("max_model_requests", 4)),
                       TSAD_MAX_STEPS=str(cfg["budget"]["max_steps_per_run"]),
                       TSAD_MAX_WALL_SECONDS=str(int(cfg["budget"]["max_wall_minutes_per_run"]) * 60))
        search = cfg.raw.get("search", {})
        if search.get("base_url"):
            env["TSAD_SEARCH_URL"] = search["base_url"]
        self.runtime.mkdir(parents=True, exist_ok=True)
        ledger = self.runtime / "dsh-host-launch.json"
        self._recover_host(ledger)
        if self.log:
            self.log.close()
        self.log = open(self.runtime / "dsh-host.log", "a")
        pinned = host / "node_modules/node/bin/node"
        node = str(pinned) if pinned.exists() else shutil.which("node") or "node"
        self.process = subprocess.Popen([node, str(host / "launch.mjs")],
            cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            identity = _process_identity(self.process.pid)
            if identity is None:
                raise HarnessError("DSH host exited before ownership could be recorded")
            data = {"pid": self.process.pid, "processIdentity": identity,
                    "ownerPid": os.getpid(), "ownerIdentity": _process_identity(os.getpid()),
                    "runtime": str(self.runtime.resolve())}
            temporary = ledger.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data) + "\n")
            temporary.replace(ledger)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            raise

    def _recover_host(self, ledger: Path):
        try:
            previous = json.loads(ledger.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(previous, dict) or previous.get("runtime") != str(self.runtime.resolve()):
            return
        pid, owner = previous.get("pid"), previous.get("ownerPid")
        if (type(pid) is not int or pid <= 1 or type(owner) is not int or owner <= 1
                or not isinstance(previous.get("processIdentity"), str)
                or not isinstance(previous.get("ownerIdentity"), str)):
            return
        if _process_identity(pid) != previous["processIdentity"]:
            return
        if _process_identity(owner) == previous["ownerIdentity"]:
            raise HarnessError("This runtime directory already owns a DSH host with a live controller")
        try:
            if os.getpgid(pid) != pid or _process_identity(pid) != previous["processIdentity"]:
                return
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + 2
            while _process_identity(pid) == previous["processIdentity"] and time.monotonic() < deadline:
                time.sleep(.05)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def call(self, method: str, path: str, body: dict | None = None, *, timeout=None):
        try:
            if timeout is None and method.upper() == "GET":
                timeout = 8
            options = {"timeout": timeout} if timeout is not None else {}
            r = await self.client.request(method, path, json=body, **options)
            if r.status_code >= 400:
                raise HarnessError(f"DSH HTTP {r.status_code}: {r.text[:1200]}")
            return r.json() if r.content else {}
        except httpx.ConnectError:
            raise HarnessError(self.error or "DSH host unavailable") from None
        except httpx.TimeoutException:
            raise HarnessError(f"DSH {method.upper()} {path.split('?', 1)[0]} timed out; check DSH host health and dsh-host.log") from None
        except httpx.HTTPError as error:
            raise HarnessError(f"DSH {method.upper()} {path.split('?', 1)[0]} failed ({type(error).__name__}); check DSH host health and dsh-host.log") from None

    async def close(self):
        await self.client.aclose()
        if self.process:
            # The launcher and native host share the session we created. Reap
            # the entire group even if the launcher exits before its child.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self.process.wait, 10)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self.process.wait, 5)
        if self.log:
            self.log.close()


class RemoteClient:
    """Synchronous CLI/TUI client; launching it never owns experiment processes."""
    def __init__(self, url: str | None = None, token: str | None = None):
        cfg = load().raw.get("interactive", {})
        self.url = (url or os.environ.get("TSAD_HARNESS_URL") or f"http://127.0.0.1:{cfg.get('port', 8767)}").rstrip("/")
        path = _runtime_dir() / "access-token"
        self._explicit_token = bool(token or os.environ.get("TSAD_HARNESS_TOKEN"))
        self.token = token or os.environ.get("TSAD_HARNESS_TOKEN") or (path.read_text().strip() if path.exists() else "")

    def call(self, method, path, body=None):
        timeout = 8 if path == "/health" else 10 if method.upper() == "GET" else 120
        # Plain loopback HTTP never uses TLS. Loading CA files for every TUI
        # poll was expensive on network-mounted homes and serves no purpose.
        with httpx.Client(base_url=self.url + "/api/harness/v1", trust_env=False, timeout=timeout,
                          verify=self.url.startswith("https://"),
                          headers={"Authorization": "Bearer " + self.token}) as client:
            r = client.request(method, path, json=body)
            if r.status_code >= 400:
                raise HarnessError(f"HTTP {r.status_code}: {r.text[:1500]}")
            return r.json()

    @staticmethod
    def _check_cancel(cancel):
        if cancel is not None and cancel.is_set():
            raise HarnessError("Connection cancelled; the background service keeps running")

    def _probe(self):
        try:
            self.call("GET", "/health")
        except httpx.ConnectError:
            return False
        except httpx.TimeoutException:
            # A slow live server must never be mistaken for an absent server.
            raise HarnessError("The service is responding slowly; retrying will use the same background service") from None
        if not self._explicit_token:
            path = _runtime_dir() / "access-token"
            if path.exists():
                self.token = path.read_text().strip()
        return True

    def ensure(self, cancel=None):
        self._check_cancel(cancel)
        if self._probe():
            return self
        from urllib.parse import urlparse
        address = urlparse(self.url)
        if address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost"):
            raise HarnessError("Remote harness is not connected")
        runtime = _runtime_dir()
        runtime.mkdir(parents=True, exist_ok=True)
        cfg = load()
        deadline = time.monotonic() + float(cfg.raw.get("interactive", {}).get("startup_timeout_seconds", 180))
        ledger = runtime / "service-launch.json"

        def pause():
            self._check_cancel(cancel)
            if time.monotonic() >= deadline:
                raise HarnessError(f"Service is still starting; see {runtime / 'service.log'}")
            delay = min(.2, max(0, deadline - time.monotonic()))
            cancel.wait(delay) if cancel is not None else time.sleep(delay)

        with open(runtime / "service-launch.lock", "a") as lock:
            while True:
                self._check_cancel(cancel)
                if self._probe():
                    return self
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pause()
            try:
                if self._probe():
                    return self
                previous = {}
                with contextlib.suppress(OSError, ValueError):
                    previous = json.loads(ledger.read_text())
                pid = previous.get("pid", 0)
                identity = _process_identity(pid)
                running = identity is not None and identity == previous.get("processIdentity")
                if running and previous.get("url") != self.url:
                    raise HarnessError("This runtime directory already owns a service at a different URL")
                child = None
                if not running:
                    self._check_cancel(cancel)
                    child_env = {**cfg.network_env(), "PYTHONPATH": str(HARNESS_DIR)}
                    with contextlib.suppress(HarnessError):
                        child_env[cfg.dsh_key_env] = cfg.api_key()
                    service_port = address.port or cfg.raw.get("interactive", {}).get("port", 8767)
                    with open(runtime / "service.log", "a") as out:
                        child = subprocess.Popen([str(cfg.python_bin), "-m", "tsad_harness", "serve", "--port", str(service_port)],
                            cwd=ROOT, env=child_env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
                    data = {"pid": child.pid, "processIdentity": _process_identity(child.pid),
                            "url": self.url, "startedAt": time.time()}
                    temporary = ledger.with_suffix(".json.tmp")
                    try:
                        temporary.write_text(json.dumps(data) + "\n")
                        temporary.replace(ledger)
                    except BaseException:
                        # Ownership must be durable before another client may
                        # acquire the launch lock and consider starting a host.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise
                while True:
                    self._check_cancel(cancel)
                    if self._probe():
                        return self
                    if child is not None and child.poll() is not None:
                        raise HarnessError(f"Service exited ({child.returncode}); see {runtime / 'service.log'}")
                    pause()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
