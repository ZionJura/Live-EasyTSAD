"""File guard: detect agent writes outside the task's own method folder and env.

`baseline(task)` snapshots every protected file (path, size, mtime_ns and sha1
for files < 2 MB) into `<task>/guard/baseline.json` and keeps a copy of each
such file under `<task>/guard/files/<repo-relative path>`. The baseline is taken
once per task; only `accept(task, note)` (an explicit, logged human action)
re-takes it, so a violation is never silently absorbed by a later round.
`check(task)` re-scans the same roots and reports added / modified / deleted
files; `restore(task)` undoes them from the stored copies.

Protected roots are `[protect].paths`, every `EasyTSAD/Methods/*` folder except
the task's own, and every `envs/<env>/` folder (without `venv/`) except a
dedicated env of the method (an env whose only registered method is the task's
method). Method folders and dedicated envs of other unfinished tasks
(`Task.all()`, phase != DONE) are not protected either, so tasks can run side
by side. Shared envs stay protected even when the method is registered there:
packages must never be added to them. Byte-code caches are ignored.

The guard is only checked after agent runs (and in the IMPLEMENT/SMOKE exit
checks), never in deterministic phases: REPORT legitimately writes
`LiveLeaderBoard/catalog.toml` and `Methods/<M>/REPRODUCTION.md` after the
last agent phase, so `validate.evaluation` must not call `check`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from .config import ROOT, load
from .task import Phase, Task, now

HASH_LIMIT = 2 * 1024 * 1024
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = (".pyc", ".pyo")
REGISTRY = ROOT / "envs" / "registry.toml"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_registry() -> tuple[dict, str | None]:
    """`[envs]` of envs/registry.toml and a parse error message (({}, msg) when unreadable)."""
    try:
        with open(REGISTRY, "rb") as f:
            return tomllib.load(f).get("envs", {}), None
    except FileNotFoundError:
        return {}, None
    except (OSError, tomllib.TOMLDecodeError) as e:
        return {}, f"envs/registry.toml does not parse: {e}"


def _own_envs(methods: set[str], reg: dict) -> set[str]:
    """Envs dedicated to one of `methods` (writable)."""
    return {e for e, s in reg.items() if isinstance(s, dict)
            and len(s.get("methods") or []) == 1 and s["methods"][0] in methods}


def _active_methods(task: Task) -> set[str]:
    """The task's method plus the methods of other unfinished tasks."""
    out = {task.method}
    try:
        out |= {t.method for t in Task.all() if t.phase is not Phase.DONE}
    except Exception:   # a broken sibling task must not break the guard
        pass
    return out


def writable(task: Task, reg: dict | None = None) -> list[str]:
    """Repo-relative method folders and dedicated envs that are not protected for this task."""
    active = _active_methods(task)
    reg = load_registry()[0] if reg is None else reg
    return sorted([_rel(load().methods_dir / m) for m in active]
                  + [f"envs/{e}" for e in _own_envs(active, reg)])


def roots(task: Task, *, skip: list[str] | None = None) -> list[Path]:
    """Protected root paths (files or directories) for this task; `skip` overrides `writable()`."""
    cfg = load()
    skip = set(writable(task) if skip is None else skip)
    out = [ROOT / p for p in cfg["protect"].get("paths", [])]
    out += [p for p in sorted(cfg.methods_dir.iterdir())
            if p.is_dir() and _rel(p) not in skip and not p.name.startswith((".", "__"))]
    out += [p for p in sorted((ROOT / "envs").iterdir()) if p.is_dir() and _rel(p) not in skip]
    return out


def _files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_symlink() or e.is_file():
                if not e.name.endswith(SKIP_SUFFIXES):
                    yield e
            elif e.is_dir() and e.name not in SKIP_DIRS:
                stack.append(e)


def _entry(p: Path) -> dict:
    st = p.lstat()
    rec = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if p.is_file() and not p.is_symlink() and st.st_size < HASH_LIMIT:
        rec["sha1"] = hashlib.sha1(p.read_bytes()).hexdigest()
    return rec


def snapshot(task: Task, rts: list[Path] | None = None) -> dict[str, dict]:
    """Current state of all protected files, keyed by repo-relative path."""
    snap: dict[str, dict] = {}
    for r in rts if rts is not None else roots(task):
        for f in _files(r):
            try:
                snap[_rel(f)] = _entry(f)
            except OSError:
                pass
    return snap


def baseline(task: Task, *, force: bool = False) -> Path:
    """Write `<task>/guard/baseline.json` (+ file copies) unless it exists; `force` re-takes it."""
    gdir = task.dir / "guard"
    out = gdir / "baseline.json"
    if out.exists() and not force:
        return out
    skip = writable(task)
    rts = roots(task, skip=skip)
    snap = snapshot(task, rts)
    shutil.rmtree(gdir / "files", ignore_errors=True)
    for k, v in snap.items():
        if "sha1" in v:
            dst = gdir / "files" / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(ROOT / k, dst)
            except OSError:
                pass
    gdir.mkdir(parents=True, exist_ok=True)
    tmp = gdir / "baseline.json.tmp"
    tmp.write_text(json.dumps({"at": now(), "roots": [_rel(r) for r in rts], "writable": skip,
                               "files": snap}))
    tmp.replace(out)
    task.log("guard/baseline", files=len(snap), force=force)
    return out


def _changed(a: dict, b: dict) -> bool:
    if "sha1" in a and "sha1" in b:
        return a["sha1"] != b["sha1"]
    return a["size"] != b["size"] or a["mtime_ns"] != b["mtime_ns"]


def _under(k: str, rts: list[str]) -> bool:
    return any(k == r or k.startswith(r.rstrip("/") + "/") for r in rts)


def check(task: Task) -> list[str]:
    """Return `"<added|modified|deleted> <path>"` for every protected change since the baseline.

    Never raises: problems are returned as `"error <message>"` entries (e.g. an
    unparsable envs/registry.toml, which then keeps the baseline's env roots).
    Paths under roots that became writable since the baseline (e.g. a new
    dedicated env, another task's new method) are ignored. A missing baseline
    is reported as a problem.
    """
    try:
        p = task.dir / "guard" / "baseline.json"
        if not p.exists():
            return ["error missing guard baseline (guard.baseline was never run for this task)"]
        data = json.loads(p.read_text())
        reg, rerr = load_registry()
        errs = [f"error {rerr}"] if rerr else []
        # writable now or at baseline time (a task that finished since keeps its folder
        # writable); on a registry parse error only the baseline's dedicated envs stay writable
        skip = set(writable(task, {} if rerr else reg)) | set(data.get("writable", []))
        rts = roots(task, skip=list(skip))
        cur_roots = [_rel(r) for r in rts]
        base = {k: v for k, v in data["files"].items() if _under(k, cur_roots)}
        cur = snapshot(task, rts)
        out = [f"added {k}" for k in sorted(cur.keys() - base.keys())]
        out += [f"deleted {k}" for k in sorted(base.keys() - cur.keys())]
        out += [f"modified {k}" for k in sorted(cur.keys() & base.keys()) if _changed(base[k], cur[k])]
        return errs + sorted(out, key=lambda s: s.split(" ", 1)[1])
    except Exception as e:  # the guard must never crash the pipeline
        return [f"error guard check failed: {type(e).__name__}: {e}"]


def accept(task: Task, note: str) -> list[str]:
    """Human action: accept the current protected state as the new baseline (logged)."""
    accepted = check(task)
    baseline(task, force=True)
    task.log("guard/accept", note=note, accepted=accepted)
    return accepted


def restore(task: Task, paths: list[str] | None = None) -> list[str]:
    """Undo protected changes from `guard/files` (only paths in the current violation list).

    Modified/deleted files are copied back from the baseline copy, added files are
    deleted. `paths` restricts the action to these repo-relative paths. Returns the
    actions taken (and the files that could not be restored).
    """
    want = {p.split(" ", 1)[1] if p.startswith(("added ", "modified ", "deleted ")) else p
            for p in paths} if paths else None
    store = task.dir / "guard" / "files"
    actions = []
    for v in check(task):
        kind, path = v.split(" ", 1)
        if kind == "error" or (want is not None and path not in want):
            continue
        target = ROOT / path
        try:
            if kind == "added":
                if target.is_symlink() or target.is_file():
                    target.unlink()
                actions.append(f"deleted {path}")
            elif (store / path).is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    target.unlink()
                shutil.copy2(store / path, target)
                actions.append(f"restored {path}")
            else:
                actions.append(f"cannot restore {path} (no stored copy; file >= 2 MB or a symlink)")
        except OSError as e:
            actions.append(f"failed {path}: {e}")
    task.log("guard/restore", paths=sorted(want) if want else None, actions=actions)
    return actions


def restore_hint(paths: list[str], task_id: str = "<task>") -> str:
    """Non-destructive instructions for a human; never suggests git commands."""
    if not paths:
        return "file guard clean"
    lines = ["Protected files changed (only the method folder, its dedicated env and the "
             "task dir are writable):", *(f"  - {p}" for p in paths)]
    if any(not p.startswith("error ") for p in paths):
        lines.append(f"A human must either undo them with `scripts/harness guard {task_id} --restore` "
                     f"(restores the baseline copies, deletes added files) or keep them with "
                     f"`scripts/harness guard {task_id} --accept -m <reason>`.")
    if any(p.startswith("error ") for p in paths):
        lines.append("Resolve the errors listed above (e.g. make envs/registry.toml parse again, "
                     "or take the baseline) before continuing.")
    return "\n".join(lines)


__all__ = ["baseline", "check", "accept", "restore", "restore_hint", "snapshot", "roots", "writable",
           "load_registry"]
