"""Machine-checked exit conditions of the harness phases.

Every function returns a list of human/agent readable error strings; an empty
list means the phase is finished. The agent never decides that a phase is done.
`CONTRACT_SCHEMA` is a small JSON-schema subset (type, required, properties,
items, enum, minLength, minItems) validated by `check_schema` without external
libraries.

Claim `train.epochs` must carry a `config_key` naming a [Model_Params.Default]
key (the smoke run caps the epoch count through it), unless the method has no
epoch loop: then its value is null / 0 / "n/a" or the claim sets `"iterative": false`.

The guard is checked by `implementation` and `smoke` (agent phases) but never by
`evaluation`: REPORT legitimately writes LiveLeaderBoard/catalog.toml afterwards.
"""
from __future__ import annotations

import json
import re
import tomllib

from . import guard, runner
from .config import ROOT
from .task import Task

CLAIM_STATUSES = ["explicit", "inferred", "unspecified"]
REQUIRED_CLAIMS = ["data.preprocess", "data.window_size", "model.architecture", "train.optimizer",
                   "train.lr", "train.epochs", "train.batch_size", "score.function",
                   "threshold.protocol"]
_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

CONTRACT_SCHEMA: dict = {
    "type": "object",
    "required": ["method", "paper", "task_type", "summary", "claims", "score", "dependencies"],
    "properties": {
        "method": {"type": "string", "minLength": 1},
        "paper": {"type": "object", "required": ["title"], "properties": {
            "title": {"type": "string", "minLength": 1}, "arxiv_id": {"type": ["string", "null"]},
            "venue": {"type": ["string", "null"]}, "year": {"type": ["integer", "null"]}}},
        "task_type": {"type": "string", "enum": ["UTS", "MTS", "both"]},
        "summary": {"type": "string", "minLength": 1},
        "claims": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["id", "value", "status", "source"], "properties": {
                "id": {"type": "string", "minLength": 1},
                "value": {},
                "status": {"type": "string", "enum": CLAIM_STATUSES},
                "source": _STR,
                "mutable": {"type": "boolean"},
                "iterative": {"type": "boolean"},   # train.epochs only: false = no epoch loop
                # key in config.toml [Model_Params.Default] (or "Data_Params.<key>") holding the value
                "config_key": {"type": "string"}}}},
        "score": {"type": "object", "required": ["definition", "alignment"], "properties": {
            "definition": {"type": "string", "minLength": 1},
            "alignment": {"type": "string", "minLength": 1}}},
        "dependencies": _STRS,
        "official_code": {"type": ["object", "null"], "properties": {
            "url": _STR, "commit": _STR, "used_files": _STRS}},
        "open_questions": _STRS,
    },
}

CONTRACT_EXAMPLE: dict = {
    "method": "Name",
    "paper": {"title": "...", "arxiv_id": "...", "venue": "...", "year": 2022},
    "task_type": "UTS|MTS|both",
    "summary": "two-sentence description of the algorithm",
    "claims": [
        {"id": "data.window_size", "value": 100, "status": "explicit|inferred|unspecified",
         "source": "Sec. 4.1 / Table 5 / code: solver.py:L42", "mutable": False,
         "config_key": "window_size"},
        {"id": "train.epochs", "value": 50, "status": "explicit", "source": "Sec. 4.2",
         "mutable": False, "config_key": "epochs",
         "note": "config_key is required for train.epochs (the smoke run caps epochs through it); "
                 "for methods without an epoch loop use value null and \"iterative\": false"},
    ],
    "score": {"definition": "how the anomaly score is computed",
              "alignment": "how scores map to timestamps"},
    "dependencies": ["torch>=2", "..."],
    "official_code": {"url": "...", "commit": "...", "used_files": ["model/AnomalyTransformer.py"]},
    "open_questions": ["..."],
}

_TYPES = {"object": dict, "array": list, "string": str, "number": (int, float),
          "integer": int, "boolean": bool, "null": type(None)}


def _is(v, t: str) -> bool:
    if t in ("integer", "number") and isinstance(v, bool):
        return False
    return isinstance(v, _TYPES[t])


def check_schema(obj, schema: dict, path: str = "$") -> list[str]:
    """Validate `obj` against the JSON-schema subset used by CONTRACT_SCHEMA."""
    errs: list[str] = []
    types = schema.get("type")
    if types:
        types = [types] if isinstance(types, str) else types
        if not any(_is(obj, t) for t in types):
            return [f"{path}: expected {'|'.join(types)}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, str) and len(obj.strip()) < schema.get("minLength", 0):
        errs.append(f"{path}: must be a non-empty string")
    if isinstance(obj, dict):
        errs += [f"{path}.{k}: required" for k in schema.get("required", []) if k not in obj]
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs += check_schema(obj[k], sub, f"{path}.{k}")
    if isinstance(obj, list):
        if len(obj) < schema.get("minItems", 0):
            errs.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, x in enumerate(obj):
                errs += check_schema(x, schema["items"], f"{path}[{i}]")
    return errs


def contract_errors(c, method: str) -> list[str]:
    """Schema + semantic checks of a contract object (used by the save_contract tool)."""
    errs = check_schema(c, CONTRACT_SCHEMA)
    if not isinstance(c, dict):
        return errs
    if isinstance(c.get("method"), str) and c["method"] != method:
        errs.append(f"$.method: must be {method!r}, got {c['method']!r}")
    claims = [x for x in c.get("claims", []) if isinstance(x, dict)] \
        if isinstance(c.get("claims"), list) else []
    ids = [x.get("id") for x in claims]
    dup = sorted({i for i in ids if isinstance(i, str) and ids.count(i) > 1})
    if dup:
        errs.append(f"$.claims: duplicate ids {dup}")
    missing = [i for i in REQUIRED_CLAIMS if i not in ids]
    if missing:
        errs.append(f"$.claims: missing required claim ids {missing}")
    for x in claims:
        if x.get("status") in ("explicit", "inferred") and not str(x.get("source") or "").strip():
            errs.append(f"claim {x.get('id')!r}: status {x['status']!r} needs a non-empty source")
        key = x.get("config_key")
        if isinstance(key, str) and key and not re.fullmatch(r"(Data_Params\.)?[A-Za-z_]\w*", key):
            errs.append(f"claim {x.get('id')!r}: config_key {key!r} must be a [Model_Params.Default] "
                        f"key or 'Data_Params.<key>'")
        val = x.get("value")
        if isinstance(key, str) and key and (isinstance(val, (dict, list)) or
                                             (isinstance(val, str) and (len(val) > 40 or " " in val.strip()))):
            errs.append(f"claim {x.get('id')!r}: config_key is only for scalar hyper-parameters "
                        f"(numbers, booleans, short identifiers); drop config_key for descriptive values")
        if x.get("id") == "train.epochs" and not runner.non_iterative(x) and (
                not isinstance(key, str) or not key or key.startswith("Data_Params.")):
            errs.append("claim 'train.epochs': needs a config_key naming the [Model_Params.Default] "
                        "epoch key (the smoke run caps epochs through it); for a method without an "
                        "epoch loop set value null or \"iterative\": false")
    return errs


def _load_contract(task: Task) -> tuple[dict | None, list[str]]:
    p = task.dir / "contract.json"
    if not p.exists():
        return None, ["contract.json missing (call mcp__tsad__save_contract)"]
    try:
        return json.loads(p.read_text()), []
    except json.JSONDecodeError as e:
        return None, [f"contract.json is not valid JSON: {e}"]


def contract(task: Task) -> list[str]:
    """EVIDENCE exit check: contract.json exists and is valid."""
    c, errs = _load_contract(task)
    return errs or contract_errors(c, task.method)


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return a == b


def config_vs_contract(task: Task, cfg: dict) -> list[str]:
    """Claims with a config_key must exist in config.toml with the same scalar value."""
    c, errs = _load_contract(task)
    if errs or not isinstance(c, dict):
        return errs
    default = cfg.get("Model_Params", {}).get("Default", {})
    data = cfg.get("Data_Params", {})
    out = []
    for x in c.get("claims", []):
        key = x.get("config_key") if isinstance(x, dict) else None
        if not key:
            continue
        table, k = (data, key.split(".", 1)[1]) if key.startswith("Data_Params.") else (default, key)
        where = "[Data_Params]" if table is data else "[Model_Params.Default]"
        if k not in table:
            out.append(f"config.toml {where} lacks {k!r} (contract claim {x['id']!r})")
        elif isinstance(x.get("value"), (int, float, str, bool)) and not _same(x["value"], table[k]):
            out.append(f"config.toml {where}.{k} = {table[k]!r} but contract claim {x['id']!r} "
                       f"says {x['value']!r} (fix config.toml, or record a deviation and update the contract)")
    return out


def implementation(task: Task, *, import_timeout: float = 900) -> list[str]:
    """IMPLEMENT exit check: layout, registry, src/ isolation, config, import check, guard."""
    m, d = task.method, task.method_dir
    errs: list[str] = []
    init, main, cfg_p = d / "__init__.py", d / f"{m}.py", d / "config.toml"
    for p in (init, main, cfg_p):
        if not p.is_file():
            errs.append(f"missing {p.relative_to(ROOT)}")
    if init.is_file() and not re.search(rf"from\s+\.{m}\s+import\s+.*\b{m}\b", init.read_text()):
        errs.append(f"__init__.py must contain 'from .{m} import {m}'")
    if main.is_file():
        src = main.read_text()
        if not re.search(rf"^class\s+{m}\s*\(([^)]*\bBaseMethod\b[^)]*)\)", src, re.M):
            errs.append(f"{m}.py must define 'class {m}(BaseMethod)'")
    cfg: dict = {}
    if cfg_p.is_file():
        try:
            cfg = tomllib.loads(cfg_p.read_text())
        except tomllib.TOMLDecodeError as e:
            errs.append(f"config.toml does not parse: {e}")
        if cfg and not isinstance(cfg.get("Model_Params", {}).get("Default"), dict):
            errs.append("config.toml needs a [Model_Params.Default] table")
        if cfg:
            errs += config_vs_contract(task, cfg)
            ec = runner.epochs_claim(task)   # a missing key is reported by config_vs_contract
            if ec and not runner.non_iterative(ec) and not str(ec.get("config_key") or "").strip():
                errs.append("contract claim 'train.epochs' needs a config_key naming the [Model_Params.Default] "
                            "epoch key (the smoke run caps epochs through it), or \"iterative\": false")
    reg, rerr = guard.load_registry()
    if rerr:
        errs.append(rerr)
    envs = [e for e, s in reg.items() if isinstance(s, dict) and m in s.get("methods", [])]
    if rerr is None and len(envs) != 1:
        errs.append(f"{m} must be registered in exactly one env in envs/registry.toml (found {envs})")
    allowed = {"__init__.py", f"{m}.py", "config.toml", "REPRODUCTION.md", "src", "__pycache__"}
    extra = sorted(p.name for p in d.iterdir() if p.name not in allowed) if d.is_dir() else []
    if extra:
        errs.append(f"unexpected files in {d.relative_to(ROOT)}: {extra}; keep only __init__.py, {m}.py, "
                    f"config.toml and src/ (put scratch/self-test scripts in the task dir, not the method)")
    srcdir = d / "src"
    if srcdir.is_dir():
        imp = re.compile(r"^\s*(?:from|import)\s+(EasyTSAD\b|\.+)", re.M)
        for f in sorted(srcdir.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            depth = len(f.relative_to(srcdir).parts) - 1   # packages below src/
            for tgt in imp.findall(f.read_text(errors="replace")):
                if tgt == "EasyTSAD" or len(tgt) - 1 > depth:
                    errs.append(f"src/{f.relative_to(srcdir)} imports outside src/ ({tgt!r}); "
                                f"src/ must not depend on EasyTSAD")
                    break
    if not errs:  # the import check is slow; only run it on a structurally valid method
        ok, out = runner.import_check(m, timeout=import_timeout)
        if not ok:
            errs.append("import check failed:\n" + out[-2500:])
    changes = guard.check(task)
    if changes:
        errs.append(guard.restore_hint(changes, task.id))
    return errs


def smoke(task: Task) -> list[str]:
    """SMOKE exit check: latest smoke run succeeded with finite 1-D scores, method unchanged since.

    "Unchanged" compares `runner.source_hash` (*.py + config.toml) with the hash the
    smoke run recorded, so artifacts the run writes into the method folder do not count.
    """
    res = runner.latest_smoke(task)
    if not res:
        return ["no smoke run yet (call mcp__tsad__run_smoke)"]
    errs = []
    if res.get("status") != "success":
        errs.append(f"latest smoke run #{res.get('n')} status={res.get('status')} "
                    f"[{res.get('failure_class')}]: {(res.get('error') or '')[-1200:]}")
    if not res.get("score_check", {}).get("ok"):
        errs.append("score check failed: " + "; ".join(res.get("score_check", {}).get("problems", [])))
    if res.get("run_dir"):
        run_dir = ROOT / res["run_dir"]
        try:
            man = json.loads((run_dir / "manifest.json").read_text())
            if man.get("status") != "success":
                errs.append(f"run manifest {res['run_dir']} has status {man.get('status')}")
        except (OSError, json.JSONDecodeError):
            errs.append(f"run manifest missing for {res['run_dir']}")
    if res.get("source_hash") != runner.source_hash(task.method_dir):
        errs.append("method source files (*.py, config.toml) changed since the latest smoke run; run it again")
    changes = guard.check(task)
    if changes:
        errs.append(guard.restore_hint(changes, task.id))
    return errs


def evaluation(task: Task) -> list[str]:
    """EVALUATE exit check: every (method, dataset) run of the launched evaluation succeeded.

    No guard check here: guard runs only after agent phases (REPORT writes the catalog).
    """
    st = runner.eval_status(task)
    if not st["launched"]:
        return ["evaluation not launched"]
    if st["running"]:
        return ["evaluation still running"]
    return [f"{d}: {v['status']}" + (f" ({v['error'].splitlines()[0]})" if v.get("error") else "")
            for d, v in st["datasets"].items() if v["status"] != "success"]


__all__ = ["CONTRACT_SCHEMA", "CONTRACT_EXAMPLE", "REQUIRED_CLAIMS", "check_schema", "contract_errors",
           "contract", "implementation", "smoke", "evaluation", "config_vs_contract"]
