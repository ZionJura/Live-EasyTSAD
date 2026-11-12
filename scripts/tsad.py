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
ENV_PROJECTS: dict[str, Path] = {}

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
easytsad = {{ path = "{root_path}", editable = true }}
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
    ENV_PROJECTS.clear()
    for env, spec in reg["envs"].items():
        project = (ROOT / spec.get("path", f"envs/{env}")).resolve()
        if not project.is_relative_to(ROOT.resolve()):
            die(f"env '{env}' project path must be inside the repository")
        if not (project / "pyproject.toml").exists():
            die(f"env '{env}' is registered but {project / 'pyproject.toml'} is missing")
        ENV_PROJECTS[env] = project
        for m in spec.get("methods", []):
            if m in seen:
                die(f"method '{m}' is registered in both '{seen[m]}' and '{env}'")
            seen[m] = env
    reg["method_env"] = seen
    return reg


def env_vars(env: str) -> dict:
    e = dict(os.environ)
    e["UV_PROJECT_ENVIRONMENT"] = str(ENV_PROJECTS[env] / "venv")
    return e


def uv(env: str, *args: str, check: bool = True) -> int:
    cmd = ["uv", *args, "--project", str(ENV_PROJECTS[env])]
    print("+ " + " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, env=env_vars(env), cwd=ROOT)
    if check and rc != 0:
        die(f"uv failed for env '{env}' (exit {rc})")
    return rc


def uv_run(env: str, command: list[str]) -> None:
    """Run `command` inside env, with the repository root as cwd."""
    if not command:
        die("missing command after '--'")
    cmd = ["uv", "run", "--project", str(ENV_PROJECTS[env]), "--", *command]
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
        venv = ENV_PROJECTS[env] / "venv"
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
            ["uv", "run", "--project", str(ENV_PROJECTS[env]), "--", "python", "-c", code],
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
    if a.name in reg["envs"]:
        die(f"env '{a.name}' is already registered")
    if len(a.methods) == 1:
        method = a.methods[0]
        if Path(method).name != method or method in (".", ".."):
            die("method must be a directory name")
        d = METHODS_DIR / method
    else:
        d = ENVS_DIR / a.name
    if (d / "pyproject.toml").exists():
        die(f"{d.relative_to(ROOT)}/pyproject.toml already exists")
    d.mkdir(parents=True, exist_ok=True)
    desc = a.description or f"Environment for {', '.join(a.methods) or 'new methods'}."
    (d / "pyproject.toml").write_text(ENV_TEMPLATE.format(
        name=a.name, python=a.python, description=desc, root_path=os.path.relpath(ROOT, d)))
    for m in a.methods:
        if m in reg["method_env"]:
            print(f"warning: '{m}' is still registered in '{reg['method_env'][m]}'; "
                  f"remove it there by hand", file=sys.stderr)
    with open(REGISTRY, "a") as f:
        path_entry = f'path = "{d.relative_to(ROOT).as_posix()}"\n' if len(a.methods) == 1 else ""
        f.write(f'\n[envs.{a.name}]\npython = "{a.python}"\ndescription = "{desc}"\n{path_entry}'
                f"methods = [{', '.join(repr(m).replace(chr(39), chr(34)) for m in a.methods)}]\n")
    print(f"created {d.relative_to(ROOT)}/pyproject.toml and registered it in envs/registry.toml")
    print(f"next: scripts/tsad add {a.name} <packages...> && scripts/tsad check {a.name}")


def cmd_remove(reg, a):
    select_envs(reg, [a.env])
    venv = ENV_PROJECTS[a.env] / "venv"
    if venv.exists():
        shutil.rmtree(venv)
        print(f"removed {venv.relative_to(ROOT)} (definition and lock file kept)")


def cmd_exp(reg, a):
    """Run an experiment spec. The unit of work is one (method, dataset) pair, run
    in the method's env and pinned to one GPU slot; pairs are spread over all
    slots. Results go to LiveLeaderBoard/results/runs."""
    import datetime as dt
    import signal
    import time

    spec_path = Path(a.spec).resolve()
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)
    methods = a.methods or spec.get("methods", [])
    datasets = a.datasets or spec.get("datasets", [])
    if not methods:
        die("no methods given (spec 'methods' or --methods)")
    missing = [m for m in methods if m not in reg["method_env"]]
    if missing:
        die(f"methods not registered in envs/registry.toml: {', '.join(missing)}")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = spec.get("name", spec_path.stem)
    logs = ROOT / "LiveLeaderBoard" / "results" / "logs" / f"{run_id}_{name}"
    logs.mkdir(parents=True, exist_ok=True)
    slots = [g.strip() for g in a.gpus.split(",")] if a.gpus else [None]
    slots = slots * max(1, a.per_gpu)
    # Datasets in spec order, methods interleaved, so every method makes progress early.
    threads = max(1, (os.cpu_count() or 1) // (len(slots) * max(1, a.concurrent_exps)))
    queue = [(m, d) for d in datasets for m in methods]
    running: dict[int, tuple[subprocess.Popen, str, str, Path]] = {}  # slot index -> job
    failed = []
    print(f"experiment '{name}' run_id={run_id} jobs={len(queue)} slots={slots}", flush=True)

    def stop(signum, frame):
        for p, *_ in running.values():
            p.send_signal(signal.SIGTERM)
        for p, *_ in running.values():
            p.wait()
        print("interrupted; running jobs were marked 'aborted'", flush=True)
        sys.exit(130)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while queue or running:
        for i, gpu in enumerate(slots):
            if i in running or not queue:
                continue
            m, d = queue.pop(0)
            env = reg["method_env"][m]
            e = env_vars(env)
            e["TSAD_ENV"] = env
            e["TSAD_ENV_PROJECT"] = str(ENV_PROJECTS[env])
            e.setdefault("TQDM_DISABLE", "1")  # progress bars would bloat the logs
            # Share the CPU cores between concurrent jobs instead of oversubscribing them.
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                e.setdefault(var, str(threads))
            if gpu is not None:
                e["CUDA_VISIBLE_DEVICES"] = gpu
            out = logs / f"{m}_{d}.out"
            cmd = ["uv", "run", "--project", str(ENV_PROJECTS[env]), "--", "python",
                   str(ROOT / "LiveLeaderBoard" / "pipeline" / "run.py"),
                   "--spec", str(spec_path), "--method", m, "--run-id", run_id,
                   "--datasets", d, *(["--force"] if a.force else [])]
            fh = open(out, "w")
            running[i] = (subprocess.Popen(cmd, env=e, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT), m, d, out)
            print(f"  start  {m:<14} {d:<8} env={env:<8} gpu={gpu}", flush=True)
        time.sleep(2)
        for i, (p, m, d, out) in list(running.items()):
            if p.poll() is None:
                continue
            del running[i]
            if p.returncode != 0:
                failed.append(f"{m}/{d}")
            res = [l for l in out.read_text(errors="replace").splitlines()
                   if l.startswith(("[success]", "[failed]", "[skip]", "[aborted]"))]
            print(f"  {'done' if p.returncode == 0 else 'FAILED':<6} {m:<14} {d:<8} "
                  f"{res[-1].split(' ')[0] if res else f'(exit {p.returncode})'}  "
                  f"[{len(queue)} queued, {len(running)} running]", flush=True)
            try:
                cmd_export(reg, a)  # keep leaderboard files current while the experiment runs
            except Exception as ex:  # never let export problems kill the scheduler
                print(f"  export failed: {ex}", flush=True)
    cmd_export(reg, a)
    if failed:
        print("failed: " + ", ".join(failed) + " (see manifest.json 'error' in their run dirs)")
        sys.exit(1)


def _lb_export():
    sys.path.insert(0, str(ROOT / "LiveLeaderBoard" / "pipeline"))
    import export  # noqa: E402
    return export


def cmd_export(reg, a):
    lb, bm = _lb_export().export()
    print(f"wrote {lb.relative_to(ROOT)} and {bm.relative_to(ROOT)}")


def cmd_table(reg, a):
    ex = _lb_export()
    ex.export()
    print(ex.table(a.schema, a.metric))


def cmd_status(reg, a):
    """Show runs that are still in progress, with curve-level progress."""
    runs = ROOT / "LiveLeaderBoard" / "results" / "runs"
    found = False
    for man_path in sorted(runs.glob("*/*/*/*/manifest.json")):
        import json
        man = json.loads(man_path.read_text())
        if man.get("status") != "running":
            continue
        found = True
        d = man_path.parent
        total = len(man["config"].get("curves") or
                    [p for p in (ROOT / "datasets" / man["dataset_type"] / man["dataset"]).iterdir() if p.is_dir()])
        done = len(list((d / "scores").glob("*.npy")))
        print(f"{man['method']:<14} {man['schema']:<10} {man['dataset']:<8} {done:>4}/{total:<4} "
              f"started {man['started_at']}  ({d.relative_to(ROOT)})")
    if not found:
        print("no runs in progress")


def cmd_doc(reg, a):
    lines = [
        "# Environments",
        "",
        "<!-- Generated by `scripts/tsad doc` from envs/registry.toml. Do not edit by hand. -->",
        "",
        "| Environment | Project | Python | Description | Methods |",
        "| --- | --- | --- | --- | --- |",
    ]
    for env, spec in reg["envs"].items():
        lines.append(f"| `{env}` | `{ENV_PROJECTS[env].relative_to(ROOT.resolve())}` | {spec.get('python', '?')} | {spec.get('description', '')} | "
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
    s = sub.add_parser("new", help="scaffold and register a new env (single-method envs live with the method)")
    s.add_argument("name"); s.add_argument("--python", required=True)
    s.add_argument("--methods", nargs="*", default=[]); s.add_argument("--description")
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("remove", help="delete an env's venv (keeps its definition)")
    s.add_argument("env"); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("doc", help="regenerate envs/ENVIRONMENTS.md")
    s.set_defaults(fn=cmd_doc)
    s = sub.add_parser("exp", help="run an experiment spec into LiveLeaderBoard/results")
    s.add_argument("spec"); s.add_argument("--methods", nargs="*")
    s.add_argument("--datasets", nargs="*", help="subset of the spec's datasets")
    s.add_argument("--gpus", help="comma-separated GPU ids, one method process per slot, e.g. 0,1,2")
    s.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU (default 1)")
    s.add_argument("--concurrent-exps", type=int, default=1,
                   help="number of experiments running side by side, used to split CPU threads (default 1)")
    s.add_argument("--force", action="store_true", help="re-run even if an identical successful run exists")
    s.set_defaults(fn=cmd_exp)
    s = sub.add_parser("export", help="rebuild results/leaderboard.json and results/benchmark.json")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("status", help="show in-progress runs and their curve progress")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("table", help="print the current leaderboard as a text table")
    s.add_argument("--schema"); s.add_argument("--metric", default="event_f1")
    s.set_defaults(fn=cmd_table)

    a = p.parse_args()
    if getattr(a, "command", None) and a.command[:1] == ["--"]:
        a.command = a.command[1:]
    a.fn(load_registry(), a)


if __name__ == "__main__":
    main()
