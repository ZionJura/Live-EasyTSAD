"""EasyTSAD environment manager (invoked through `scripts/tsad`).

Reads envs/registry.toml and drives uv so that every method runs inside the
environment it is registered to. Stdlib only; requires Python >= 3.11.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get("TSAD_ROOT", Path(__file__).resolve().parent.parent))
ENVS_DIR = ROOT / "envs"
REGISTRY = ENVS_DIR / "registry.toml"
DOC = ENVS_DIR / "ENVIRONMENTS.md"
METHODS_DIR = ROOT / "EasyTSAD" / "Methods"

ENV_TEMPLATE = """\
# Environment: {name}
# {description}

[project]
name = "tsad-env-{name}"
version = "0.0.0"
requires-python = "=={python}.*"
dependencies = [
    "easytsad",
    # add the method-specific dependencies here
]

[tool.uv]
package = false

[tool.uv.sources]
easytsad = {{ path = "../..", editable = true }}
"""


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_registry() -> dict:
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    reg.setdefault("envs", {})
    reg.setdefault("notes", {})
    seen: dict[str, str] = {}
    for env, spec in reg["envs"].items():
        if not (ENVS_DIR / env / "pyproject.toml").exists():
            die(f"env '{env}' is registered but envs/{env}/pyproject.toml is missing")
        for m in spec.get("methods", []):
            if m in seen:
                die(f"method '{m}' is registered in both '{seen[m]}' and '{env}'")
            seen[m] = env
    reg["method_env"] = seen
    return reg


def env_vars(env: str) -> dict:
    e = dict(os.environ)
    e["UV_PROJECT_ENVIRONMENT"] = str(ENVS_DIR / env / "venv")
    return e


def uv(env: str, *args: str, check: bool = True) -> int:
    cmd = ["uv", *args, "--project", str(ENVS_DIR / env)]
    print("+ " + " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, env=env_vars(env), cwd=ROOT)
    if check and rc != 0:
        die(f"uv failed for env '{env}' (exit {rc})")
    return rc


def uv_run(env: str, command: list[str]) -> None:
    """Run `command` inside env, with the repository root as cwd."""
    if not command:
        die("missing command after '--'")
    cmd = ["uv", "run", "--project", str(ENVS_DIR / env), "--", *command]
    os.chdir(ROOT)
    os.execvpe(cmd[0], cmd, env_vars(env))


def select_envs(reg: dict, names: list[str]) -> list[str]:
    for n in names:
        if n not in reg["envs"]:
            die(f"unknown env '{n}'. Known: {', '.join(reg['envs'])}")
    return names or list(reg["envs"])


def method_dirs() -> list[str]:
    return sorted(p.name for p in METHODS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")))


# ---------------------------------------------------------------- commands
def cmd_sync(reg, a):
    for env in select_envs(reg, a.envs):
        uv(env, "sync", *(["--locked"] if a.locked else []))


def cmd_lock(reg, a):
    for env in select_envs(reg, a.envs):
        uv(env, "lock", *(["--upgrade"] if a.upgrade else []))


def cmd_add(reg, a):
    select_envs(reg, [a.env])
    uv(a.env, "add", *a.packages)


def cmd_run(reg, a):
    env = reg["method_env"].get(a.method)
    if env is None:
        die(f"method '{a.method}' is not registered in envs/registry.toml")
    uv_run(env, a.command)


def cmd_env(reg, a):
    select_envs(reg, [a.env])
    uv_run(a.env, a.command)


def cmd_which(reg, a):
    env = reg["method_env"].get(a.method)
    if env is None:
        die(f"method '{a.method}' is not registered")
    print(env)


def cmd_list(reg, a):
    for env, spec in reg["envs"].items():
        venv = ENVS_DIR / env / "venv"
        state = "synced" if venv.exists() else "not synced"
        print(f"{env:<12} python {spec.get('python', '?'):<6} [{state}]  {spec.get('description', '')}")
        print("    " + ", ".join(spec.get("methods", [])))
    unregistered = [m for m in method_dirs()
                    if m not in reg["method_env"] and m not in reg["notes"]]
    if unregistered:
        print("\nUnregistered method folders: " + ", ".join(unregistered))


def cmd_check(reg, a):
    """Import every registered method inside its environment."""
    failed = []
    for env in select_envs(reg, a.envs):
        methods = reg["envs"][env].get("methods", [])
        if not methods:
            continue
        code = (
            "import sys, traceback\n"
            "import EasyTSAD.Controller  # must precede EasyTSAD.Methods (circular import)\n"
            f"for m in {methods!r}:\n"
            "    try:\n"
            "        __import__('EasyTSAD.Methods.' + m + '.' + m)\n"
            "        print('  ok    ', m)\n"
            "    except Exception as e:\n"
            "        print('  FAIL  ', m, '->', type(e).__name__ + ':', str(e).splitlines()[0] if str(e) else '')\n"
        )
        print(f"[{env}]")
        out = subprocess.run(
            ["uv", "run", "--project", str(ENVS_DIR / env), "--", "python", "-c", code],
            env=env_vars(env), cwd=ROOT, capture_output=True, text=True,
        )
        sys.stdout.write(out.stdout)
        if out.returncode != 0:
            sys.stdout.write(out.stderr)
            failed.append(env)
        failed += [f"{env}:{l.split()[1]}" for l in out.stdout.splitlines() if l.strip().startswith("FAIL")]
    if failed:
        print("\nfailures: " + ", ".join(failed))
        sys.exit(1)


def cmd_new(reg, a):
    d = ENVS_DIR / a.name
    if d.exists():
        die(f"envs/{a.name} already exists")
    d.mkdir(parents=True)
    desc = a.description or f"Environment for {', '.join(a.methods) or 'new methods'}."
    (d / "pyproject.toml").write_text(ENV_TEMPLATE.format(name=a.name, python=a.python, description=desc))
    for m in a.methods:
        if m in reg["method_env"]:
            print(f"warning: '{m}' is still registered in '{reg['method_env'][m]}'; "
                  f"remove it there by hand", file=sys.stderr)
    with open(REGISTRY, "a") as f:
        f.write(f'\n[envs.{a.name}]\npython = "{a.python}"\ndescription = "{desc}"\n'
                f"methods = [{', '.join(repr(m).replace(chr(39), chr(34)) for m in a.methods)}]\n")
    print(f"created envs/{a.name}/pyproject.toml and registered it in envs/registry.toml")
    print(f"next: scripts/tsad add {a.name} <packages...> && scripts/tsad check {a.name}")


def cmd_remove(reg, a):
    select_envs(reg, [a.env])
    venv = ENVS_DIR / a.env / "venv"
    if venv.exists():
        shutil.rmtree(venv)
        print(f"removed {venv.relative_to(ROOT)} (definition and lock file kept)")


def cmd_doc(reg, a):
    lines = [
        "# Environments",
        "",
        "<!-- Generated by `scripts/tsad doc` from envs/registry.toml. Do not edit by hand. -->",
        "",
        "| Environment | Python | Description | Methods |",
        "| --- | --- | --- | --- |",
    ]
    for env, spec in reg["envs"].items():
        lines.append(f"| `{env}` | {spec.get('python', '?')} | {spec.get('description', '')} | "
                     f"{', '.join(spec.get('methods', []))} |")
    lines += ["", "## Method lookup", "", "| Method | Environment | Notes |", "| --- | --- | --- |"]
    names = sorted(set(reg["method_env"]) | set(reg["notes"]), key=str.lower)
    for m in names:
        env = reg["method_env"].get(m)
        lines.append(f"| {m} | {f'`{env}`' if env else '-'} | {reg['notes'].get(m, '')} |")
    unregistered = [m for m in method_dirs() if m not in names]
    if unregistered:
        lines += ["", "Unregistered method folders: " + ", ".join(unregistered)]
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(prog="scripts/tsad", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="create/update env venvs from their lock files")
    s.add_argument("envs", nargs="*"); s.add_argument("--locked", action="store_true",
                                                      help="fail if uv.lock is out of date")
    s.set_defaults(fn=cmd_sync)
    s = sub.add_parser("lock", help="(re)resolve env lock files")
    s.add_argument("envs", nargs="*"); s.add_argument("--upgrade", action="store_true")
    s.set_defaults(fn=cmd_lock)
    s = sub.add_parser("add", help="add dependencies to an env (updates pyproject + lock + venv)")
    s.add_argument("env"); s.add_argument("packages", nargs="+"); s.set_defaults(fn=cmd_add)
    s = sub.add_parser("run", help="run a command in the env registered for METHOD")
    s.add_argument("method"); s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("env", help="run a command in a named env")
    s.add_argument("env"); s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_env)
    s = sub.add_parser("which", help="print the env a method is registered to")
    s.add_argument("method"); s.set_defaults(fn=cmd_which)
    s = sub.add_parser("list", help="list envs, their methods and sync state")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("check", help="import every registered method inside its env")
    s.add_argument("envs", nargs="*"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("new", help="scaffold and register a new env")
    s.add_argument("name"); s.add_argument("--python", required=True)
    s.add_argument("--methods", nargs="*", default=[]); s.add_argument("--description")
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("remove", help="delete an env's venv (keeps its definition)")
    s.add_argument("env"); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("doc", help="regenerate envs/ENVIRONMENTS.md")
    s.set_defaults(fn=cmd_doc)

    a = p.parse_args()
    if getattr(a, "command", None) and a.command[:1] == ["--"]:
        a.command = a.command[1:]
    a.fn(load_registry(), a)


if __name__ == "__main__":
    main()
