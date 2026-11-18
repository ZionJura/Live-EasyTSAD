"""Method-local UV projects; the scheduler owns when installation may run."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

import tomlkit

from .config import ROOT, HarnessError, load
from .task import now


class EnvironmentStorageError(HarnessError):
    pass


class EnvironmentInstallError(HarnessError):
    pass


def _distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def plan_recovery(task, error, previous=None) -> dict:
    """Plan one repair without importing code or changing the method environment.

    Returns mode, packages (distribution names), missing_import (top level or
    None), and reason. Pass the previous plan, including any controller fields,
    to escalate any failed targeted repair to one full reinstall, then research.
    Error may be an exception, traceback, or JSON-serialized finding evidence.
    """
    missing = error.name if isinstance(error, ModuleNotFoundError) else None
    if not missing:
        matches = re.findall(r"No module named \\?[\"']([\w.]+)\\?[\"']", str(error))
        missing = matches[-1] if matches else None
    missing = missing.split(".")[0] if missing else None
    action = {"mode": "research", "packages": [], "missing_import": missing, "reason": ""}

    def choose(mode, reason, packages=None):
        return {**action, "mode": mode, "packages": packages or [], "reason": reason}

    previous_mode = (previous or {}).get("mode")
    if previous_mode == "reinstall":
        return choose("reinstall_all", "An import/environment failure persisted after targeted repair; "
                      "reinstall all packages once, even if the missing import changed.")
    if previous_mode in {"reinstall_all", "rebuild"}:
        return choose("research", "An import/environment failure persisted after a full reinstall; "
                      "environment research is required instead of repeating it.")
    if not missing or not missing.isidentifier():
        return choose("research", "No concrete ModuleNotFoundError import was found in the evidence.")

    project = task.method_dir / "environment"
    try:
        data = tomllib.loads((project / "pyproject.toml").read_text())
        requirements = list(data.get("project", {}).get("dependencies", []))
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            requirements.extend(group)
        for group in data.get("dependency-groups", {}).values():
            requirements.extend(item for item in group if isinstance(item, str))
        requirements.extend(data.get("tool", {}).get("uv", {}).get("dev-dependencies", []))
        declared = {_distribution_name(match[0]) for requirement in requirements
                    if (match := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement))}
        lock = project / "uv.lock"
        if lock.is_file():
            declared.update(_distribution_name(package["name"])
                            for package in tomllib.loads(lock.read_text()).get("package", []))
        declared.discard(_distribution_name(data.get("project", {}).get("name", "")))
    except (OSError, ValueError, KeyError, TypeError):
        return choose("reinstall_all", "Project/lock declarations could not be read reliably; "
                      "reinstall all packages without guessing a distribution.")

    owners = set()
    incomplete = False
    venv = project / ".venv"
    metadata_dirs = set()
    for pattern in ("lib/python*/site-packages/*.dist-info", "lib64/python*/site-packages/*.dist-info",
                    "Lib/site-packages/*.dist-info"):
        metadata_dirs.update(path.resolve() for path in venv.glob(pattern)
                             if path.is_dir() and path.resolve().is_relative_to(venv.resolve()))
    for directory in sorted(metadata_dirs):
        name = None
        try:
            with (directory / "METADATA").open(errors="replace") as stream:
                for line in stream:
                    if not line.strip():
                        break
                    if line.lower().startswith("name:"):
                        name = line.partition(":")[2].strip()
                        break
            if not name or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
                incomplete = True
                continue
            name = _distribution_name(name)
            top_level = directory / "top_level.txt"
            record = directory / "RECORD"
            if top_level.is_file() and missing in top_level.read_text(errors="replace").split():
                owners.add(name)
            elif record.is_file():
                with record.open(errors="replace", newline="") as stream:
                    for row in csv.reader(stream):
                        first = row[0].split("/", 1)[0] if row else ""
                        if first in {missing, missing + ".py"} or (
                                first.split(".", 1)[0] == missing and first.endswith((".so", ".pyd"))):
                            owners.add(name)
                            break
            elif not top_level.is_file():
                incomplete = True
        except (OSError, csv.Error):
            incomplete = True
    if len(owners) == 1:
        name = next(iter(owners))
        if name in declared:
            return choose("reinstall", f"Installed metadata maps missing import {missing!r} to declared/locked "
                          f"distribution {name!r}; reinstall its package files.", [name])
        return choose("research", f"Installed metadata maps {missing!r} to undeclared distribution {name!r}; "
                      "research its dependency declaration.")
    if owners or incomplete or _distribution_name(missing) in declared:
        return choose("reinstall_all", f"The distribution for {missing!r} cannot be resolved unambiguously "
                      "from local metadata and declarations; reinstall all packages without guessing.")
    return choose("research", f"No declared/locked distribution or installed metadata supplies {missing!r}; "
                  "research the missing dependency.")


def _install_action(recovery) -> dict:
    if recovery is not None and not isinstance(recovery, dict):
        raise HarnessError("Environment recovery must be an action dict")
    recovery = recovery if recovery is not None else {"mode": "sync"}
    requested = recovery.get("mode")
    if not isinstance(requested, str) or requested not in {"sync", "research", "reinstall", "reinstall_all", "rebuild"}:
        raise HarnessError("Unknown environment recovery mode")
    packages = recovery.get("packages", [])
    if not isinstance(packages, list) or any(
            not isinstance(package, str) or not re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", package) for package in packages):
        raise HarnessError("Recovery packages must be distribution names, without versions or options")
    if (requested == "reinstall") != bool(packages):
        raise HarnessError("Only targeted reinstall requires a nonempty packages list")
    mode = "sync" if requested == "research" else "reinstall_all" if requested == "rebuild" else requested
    return {"mode": mode, "packages": sorted({_distribution_name(package) for package in packages}),
            "missing_import": recovery.get("missing_import"),
            "reason": recovery.get("reason", "Synchronize the method-local UV project."),
            "requested_mode": requested}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def register(task) -> dict:
    project = task.method_dir / "environment"
    config = project / "pyproject.toml"
    if not config.is_file():
        raise HarnessError("RESEARCH must produce " + relative(config))
    data = tomllib.loads(config.read_text())
    if not data.get("project", {}).get("requires-python"):
        raise HarnessError("The method UV project must declare project.requires-python")
    if "dependencies" not in data.get("project", {}):
        raise HarnessError("The method UV project must declare project.dependencies")
    registry = ROOT / "envs/registry.toml"
    name = "method_" + task.method
    with open(registry.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = tomlkit.parse(registry.read_text())
        envs = doc.setdefault("envs", tomlkit.table(is_super_table=True))
        for env_name, spec in envs.items():
            if task.method in spec.get("methods", []) and env_name != name:
                # Explicitly move this one method, preserving every other environment assignment.
                spec["methods"] = [method for method in spec.get("methods", []) if method != task.method]
        envs[name] = {"project": relative(project), "python": data["project"]["requires-python"],
                      "description": f"Method-local UV environment for {task.method}", "methods": [task.method]}
        fd, temporary = tempfile.mkstemp(prefix=".registry-", dir=registry.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(tomlkit.dumps(doc))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, registry)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
    return {"environment": name, "project": relative(project), "venv": relative(project / ".venv"),
            "lock": relative(project / "uv.lock"), "python": data["project"]["requires-python"]}


def install(task, *, on_process=None, cancel=None, recovery=None) -> dict:
    """Run one UV action under the controller's method/environment write lease.

    Research means normal sync after the environment worker. Rebuild is accepted
    as an alias for reinstall_all: no virtual environment directory is deleted.
    Success records UV completion only; the controller reruns the original import.
    """
    action = _install_action(recovery)
    project = task.method_dir / "environment"
    if project.is_symlink() or (project / ".venv").is_symlink():
        raise HarnessError("Environment installation requires the exact method-local project/.venv, not a symlink")
    result = register(task)
    result["action"] = action
    project = ROOT / result["project"]
    logs = task.method_dir / "reproduction"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "environment-install.log"
    cfg = load()
    env = cfg.network_env()
    cache = Path(cfg.raw.get("paths", {}).get("uv_cache_dir", "runtime/uv-cache")).expanduser()
    if not cache.is_absolute():
        cache = ROOT / cache
    env.update(UV_PROJECT_ENVIRONMENT=str(project / ".venv"), UV_PYTHON_INSTALL_DIR=str(ROOT / "runtime/python"),
               UV_CACHE_DIR=env.get("UV_CACHE_DIR") or str(cache), UV_MANAGED_PYTHON="1")
    argv = ["uv", "sync", "--project", str(project)]
    if action["mode"] == "reinstall_all":
        argv.append("--reinstall")
    for package in action["packages"]:
        argv.extend(["--reinstall-package", package])
    metadata_path = logs / "environment-install.json"
    metadata_path.unlink(missing_ok=True)
    with open(log_path, "a+", errors="replace") as output:
        output.write("Environment install action: " + json.dumps(action, ensure_ascii=False) + "\n")
        output.flush()
        log_offset = output.tell()
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        from .scheduler import process_identity
        record = {"id": "environment:" + str(process.pid), "kind": "environment", "pid": process.pid,
                  "processIdentity": process_identity(process.pid), "argv": argv, "cwd": ".",
                  "log": relative(log_path), "startedAt": now(), "status": "running", "action": action}
        try:
            if on_process:
                on_process(record)
            while process.poll() is None:
                cancelled = cancel.is_set() if hasattr(cancel, "is_set") else cancel() if callable(cancel) else False
                if cancelled:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise HarnessError("Environment installation cancelled")
                time.sleep(.25)
            if process.returncode:
                output.seek(log_offset)
                failure = f"UV installation failed ({process.returncode}): {output.read()[-4000:]}"
                if "disk quota exceeded" in failure.lower() or "no space left on device" in failure.lower():
                    raise EnvironmentStorageError(failure)
                raise EnvironmentInstallError(failure)
        finally:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if on_process:
                on_process({**record, "status": "finished", "returncode": process.returncode, "finishedAt": now()})
    if not (project / "uv.lock").is_file():
        raise HarnessError("UV succeeded without producing the method uv.lock")
    fd, temporary = tempfile.mkstemp(prefix=".environment-install-", dir=logs)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    return result
