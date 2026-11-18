#!/usr/bin/env python3
"""Manage the private YYT proxy in a dedicated window of tmux session 0."""

import json
import os
from pathlib import Path
import subprocess
import sys

CONFIG = Path.home() / ".config" / "terra-proxy" / "upstream.json"
SERVER = Path(__file__).resolve().with_name("terra_proxy.py")
SESSION = "0"
WINDOW = "terra-proxy"
ENV_NAMES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "TERRA_PROXY_HOST", "TERRA_PROXY_PORT")


def tmux(*args, check=True):
    return subprocess.run(["tmux", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def window_id():
    result = tmux("list-windows", "-t", SESSION, "-F", "#{window_id} #{window_name}", check=False)
    for line in result.stdout.splitlines():
        window, name = line.split(" ", 1)
        if name == WINDOW:
            return window
    return None


def configure():
    key = os.environ.get("yyt_api") or os.environ.get("YYT_API_KEY")
    if not key:
        sys.exit("Set yyt_api or YYT_API_KEY before configure")
    CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    config["yyt_api"] = key
    try:
        fd = os.open(CONFIG, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        sys.exit("Configuration already exists; remove it explicitly before reconfiguring")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    print("Saved private upstream configuration to", CONFIG)


def start():
    if not CONFIG.is_file():
        sys.exit("Run configure with the upstream credentials first")
    if window_id():
        sys.exit("Proxy tmux window already exists")
    if tmux("has-session", "-t", SESSION, check=False).returncode:
        tmux("new-session", "-d", "-s", SESSION, "-n", WINDOW, sys.executable, str(Path(__file__).resolve()), "serve")
    else:
        tmux("new-window", "-d", "-t", SESSION + ":", "-n", WINDOW, sys.executable, str(Path(__file__).resolve()), "serve")
    print("Started tmux window 0:terra-proxy")


def stop():
    window = window_id()
    if not window:
        print("Proxy tmux window is not running")
        return
    tmux("kill-window", "-t", window)
    print("Stopped tmux window 0:terra-proxy")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("configure", "start", "stop", "restart", "status", "serve"):
        sys.exit("Usage: terra_proxy_ctl.py {configure|start|stop|restart|status}")
    command = sys.argv[1]
    if command == "configure":
        configure()
    elif command == "start":
        start()
    elif command == "stop":
        stop()
    elif command == "restart":
        stop()
        start()
    elif command == "status":
        print("tmux window:", window_id() or "not running")
    else:
        config = json.loads(CONFIG.read_text())
        os.environ.update(config)
        os.execv(sys.executable, [sys.executable, "-u", str(SERVER)])


if __name__ == "__main__":
    main()
