"""Artifact contracts for the short lived published DSH CLI workers.

The controller owns phase changes and execution. Workers share the artifact
and method-local write permissions defined here with the MCP server.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from .config import HARNESS_DIR, ROOT, HarnessError


class WorkerHandoffError(HarnessError):
    pass


ROLES = {
    "fetcher": "Identify the paper and official implementation from author/paper evidence. Record repository_url, supported, evidence and paper={title,url,pdf_url,arxiv_id} when found. If initial paper material is unavailable, search or fetch author/project pages and supply its PDF URL for the controller to download. If no official repository is available or the method is not time-series anomaly detection, set supported=false and explain why in compatibility.reason; do not invent or reimplement a method.",
    "hyperparameters": "Read the paper and official source. Write config.toml with actual training defaults and reproduction/hyperparameters.json recording settings, their source file/line or paper section, and uncertainty. Never tune against benchmark test labels or scores.",
    "interface": "Read the official source and EasyTSAD BaseMethod/TSData contract. Specify input shape, preprocessing, train/test entrypoints, score direction, output shape, timestamps and alignment, with source evidence. Fill compatibility.dataset_types and compatibility.schemas with the actual supported data types and training schemas from official code. Do not force single-channel inputs or UTS/naive when the method supports MTS or other schemas. The method score must be timepoint anomaly scores; document any window/channel aggregation and temporal alignment needed. The controller selects the evaluation plan separately. Do not change the algorithm.",
    "environment": "Read official dependency and import files. Write a dedicated uv project pyproject.toml with Python requirements and the dependencies needed by the official implementation and EasyTSAD. Record dependency evidence. The controller installs the project; do not run pip or uv.",
    "implementer": "Implement the EasyTSAD adapter from the official upstream code and all three research artifacts. When context has legacy_integration=true, reuse the existing flat source at context.source (the method directory), adapter, config and registered environment named by context.environment_name (such as autotsad), repair only the supplied exception, and do not redownload, research or reconstruct missing research artifacts merely to satisfy the new layout. For a supplied repair handoff, use its resolved evidence, inspect the failing file and necessary callees, publish the minimal method-local repair, then stop. Do not inspect Harness internals or repeat resolved searches to infer extra acceptance requirements. Deliver the method-named Python module, __init__.py and config.toml; existing unchanged deliverables already satisfy this requirement. Perform any necessary score alignment in the adapter; a separate align.py is optional and has no prescribed framework API. Preserve the official algorithm and training defaults. You may create or repair necessary files anywhere within your own method directory, including upstream compatibility fixes and the method-local environment. No source fidelity or hash gate is required. Never change shared evaluators, metrics, thresholds, labels, shared environments or any other directory. Do not install environments or launch experiments; the controller handles execution.",
    "diagnoser": "Diagnose only the supplied actual exception or abnormal score evidence. The deliverable is one executable repair handoff: identify the failing call, its cause and the method-local file to repair, then publish the diagnosis immediately. Do not audit scientific fidelity, investigate unrelated historical runs or repeat resolved source searches. Write category, action, message and evidence to reproduction/diagnosis.json, then stop. Evidence is a list of objects with file (repository-relative path) and line (one-based integer) or error (exact exception text), not citation strings. Request action=repair only for category execution/environment/interface/dimension/alignment/preprocessing/oom backed by that evidence. Low scores, scientific concerns and leakage concerns alone use action=report_only. Never tune on test performance or edit implementation files.",
}


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise HarnessError(f"Artifact path is outside the repository: {path}") from error


def layout(task) -> dict[str, Path]:
    method = task.method_dir
    return {"source": method / "reproduction/source.json",
            "hyperparameters": method / "reproduction/hyperparameters.json",
            "interface": method / "reproduction/adapter.yaml",
            "environment": method / "environment/pyproject.toml",
            "environment_evidence": method / "reproduction/environment.json",
            "diagnosis": method / "reproduction/diagnosis.json"}


def write_paths(task, role: str) -> list[Path]:
    paths = layout(task)
    if role == "fetcher":
        return [paths["source"]]
    if role == "hyperparameters":
        return [paths["hyperparameters"], task.method_dir / "config.toml"]
    if role == "interface":
        return [paths["interface"]]
    if role == "environment":
        return [paths["environment"], paths["environment_evidence"]]
    if role == "diagnoser":
        return [paths["diagnosis"]]
    if role == "implementer":
        return [task.method_dir / name for name in
                (f"{task.method}.py", "__init__.py", "config.toml")]
    raise HarnessError("Unknown CLI worker role " + role)


def is_write_allowed(task, role: str, path: Path) -> bool:
    """Check both the lexical assignment and its resolved destination."""
    candidate = Path(path)
    if ".." in candidate.parts:
        return False
    root = ROOT.resolve()
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve()
    if not candidate.is_relative_to(root) or not resolved.is_relative_to(root):
        return False
    for checked in (candidate, resolved):
        if any(part.startswith(".") or part in ("venv", "node_modules", "__pycache__", "access-token")
               for part in checked.relative_to(root).parts):
            return False
    assigned = {item.absolute() for item in write_paths(task, role)}
    if role == "implementer":
        method = task.method_dir.absolute()
        return (method.resolve() == method and candidate != method and resolved != method
                and candidate.is_relative_to(method) and resolved.is_relative_to(method))
    return candidate in assigned and resolved == candidate


def prompt(task, role: str, context: dict | None = None) -> str:
    skill_name = {"fetcher": "fetch-repository", "hyperparameters": "research-hyperparameters",
                  "interface": "research-interface", "environment": "research-environment"}.get(role)
    guidance = ""
    if skill_name:
        skill = HARNESS_DIR / "dsh-host/skills" / skill_name / "SKILL.md"
        guidance = f"Domain skill (EasyTSAD/Harness/dsh-host/skills/{skill_name}/SKILL.md):\n{skill.read_text()}\n\n"
    if role in ("hyperparameters", "interface", "environment"):
        guidance += (
            "Research only your assigned deliverable. The output contract is supplied here; do not "
            "search Harness internals, validation code, other workers' outputs or other methods to infer "
            "additional requirements. Read the assigned official entrypoint and its necessary callees; "
            "the interface worker may also read BaseMethod and TSData. Once the effective behavior and "
            "source evidence are established, publish your assigned files and stop. Record unresolved "
            "details in your artifact rather than expanding the investigation. The controller owns "
            "installation, evaluation planning and execution; IMPLEMENT owns executable adaptation.\n")
        if role == "hyperparameters":
            guidance += (
                "config.toml uses [Model_Params.Default] for official parameter values; optional "
                "[Data_Params] declares preprocessing only when required by the official method. "
                "hyperparameters.json records parameter values, origins and uncertainty; provenance "
                "may be nested with each parameter and has no fixed layout. Do not copy another method's "
                "values or add dataset-specific tuning.\n")
        elif role == "interface":
            guidance += (
                "adapter.yaml is a mapping with nonempty source, target and align sections "
                "and the compatibility object described below. Provenance may accompany the relevant "
                "entries. The sections may be descriptive mappings "
                "or text; no undocumented callable, adapter base class or additional format is required.\n")
        else:
            guidance += (
                "environment/pyproject.toml must declare [project] name, requires-python and dependencies. "
                "reproduction/environment.json is a nonempty object recording dependency evidence and "
                "compatibility adjustments; there is no additional fixed schema. Select dependencies "
                "from the official execution imports, including the EasyTSAD runtime requirements; "
                "do not inspect installer or validator internals.\n")
    compatibility = ""
    if role in ("fetcher", "interface"):
        requirement = ("The interface artifact must include a compatibility object: " if role == "interface"
                       else "The discovery artifact may include a compatibility object: ")
        compatibility = (
            requirement
            + "task_type=nonempty string describing the actual task, supported=true|false, "
            "dataset_types=[UTS and/or MTS], schemas=[naive, mts, all_in_one, zero_shot, "
            "zero_shot_cross_ds as supported], reason=string. When supported=true, task_type must be "
            "'time_series_anomaly_detection'. When supported=false, record the actual task type "
            "(e.g. 'time_series_classification') and explain the incompatibility in reason. Use empty lists when unsupported "
            "or not yet established during discovery. State evidence and uncertainty; "
            "compatibility describes capabilities, not a selected evaluation plan. For the interface, also record "
            "compute_device=cpu|cuda|auto based on the actual execution entrypoint and enabled algorithms, with "
            "source evidence: cpu for a CPU-only implementation, cuda when CUDA is required, auto when both paths "
            "are supported. One CPU-only component does not establish that the whole method is CPU-only.\n")
    return (guidance + f"You are the TSAD {role} worker. {ROLES[role]}\n" + compatibility
            + "Use the scoped tsad worker tools. All file paths and evidence paths must be repository-relative. "
            "Do not launch experiments or mutate any files outside your write_paths or write_roots. "
            "Use read_file/list_files/search_files to inspect source and publish_file to publish artifacts.\n"
            + json.dumps({"method": task.method, "task": relative(task.dir),
                          "method_dir": relative(task.method_dir),
                          "paper_dir": relative(task.method_dir / "paper"),
                          "paper": task.spec.get("paper"), "code_url": task.spec.get("code_url"),
                          "artifacts": {k: relative(v) for k, v in layout(task).items()},
                          "write_paths": [relative(p) for p in write_paths(task, role)],
                          "write_roots": [relative(task.method_dir)] if role == "implementer" else [],
                          "context": context or {}}, ensure_ascii=False, indent=2))


def repair_evidence(evidence) -> bool:
    """Only concrete, inspectable source or execution evidence permits repair."""
    if not isinstance(evidence, list) or not evidence:
        return False
    valid = False
    for item in evidence:
        if not isinstance(item, dict):
            return False
        name = item.get("file") or item.get("path")
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            return False
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size > 10_000_000:
            return False
        text = path.read_text(errors="replace")
        line = item.get("line")
        location = item.get("location")
        finding = item.get("error") or item.get("finding")
        source_location = (isinstance(line, int) and not isinstance(line, bool) and 1 <= line <= len(text.splitlines()))
        source_location = source_location or (isinstance(location, str) and bool(location.strip()) and location in text)
        actual_error = isinstance(finding, str) and bool(finding.strip()) and finding in text
        if not source_location and not actual_error:
            return False
        valid = True
    return valid


def validate(task, role: str) -> dict:
    """Require parseable structured handoffs before the dependent phase runs."""
    paths = layout(task)
    if role == "implementer":
        required = write_paths(task, role)
        for path in required:
            if not path.is_file() or not path.read_text().strip():
                raise HarnessError(f"Implementer did not produce {relative(path)}")
        return {"files": [relative(p) for p in required]}
    key = {"fetcher": "source", "diagnoser": "diagnosis"}.get(role, role)
    path = paths[key]
    if not path.is_file():
        raise HarnessError(f"{role} did not produce {relative(path)}")
    text = path.read_text()
    try:
        data = (tomllib.loads(text) if path.suffix == ".toml" else
                yaml.safe_load(text) if path.suffix == ".yaml" else json.loads(text))
    except (ValueError, yaml.YAMLError) as error:
        raise HarnessError(f"Invalid {role} artifact: {error}") from error
    if not isinstance(data, dict) or not data:
        raise HarnessError(f"{role} artifact must be a nonempty object")
    if role in ("fetcher", "interface") and "compatibility" in data:
        compatibility = data["compatibility"]
        if (not isinstance(compatibility, dict)
                or not isinstance(compatibility.get("task_type"), str)
                or not compatibility["task_type"].strip()
                or not isinstance(compatibility.get("supported"), bool)
                or not isinstance(compatibility.get("reason"), str)):
            raise HarnessError("Compatibility requires a nonempty task_type string, supported=true|false and reason string")
        if compatibility["supported"] and compatibility["task_type"] != "time_series_anomaly_detection":
            raise HarnessError("Supported compatibility requires task_type=time_series_anomaly_detection")
        if compatibility.get("compute_device", "auto") not in ("cpu", "cuda", "auto"):
            raise HarnessError("Compatibility compute_device must be cpu, cuda or auto")
        for key, choices in (("dataset_types", {"UTS", "MTS"}),
                             ("schemas", {"naive", "mts", "all_in_one", "zero_shot", "zero_shot_cross_ds"})):
            values = compatibility.get(key)
            if not isinstance(values, list) or any(not isinstance(value, str) or value not in choices for value in values):
                raise HarnessError(f"Compatibility {key} must be a list drawn from {sorted(choices)}")
            if role == "interface" and compatibility["supported"] and not values:
                raise HarnessError(f"Supported interface compatibility requires nonempty {key}")
    if role == "fetcher" and data.get("supported", True) and not data.get("repository_url"):
        raise HarnessError("Source discovery requires repository_url or supported=false")
    if role == "fetcher" and (not isinstance(data.get("supported"), bool) or not data.get("evidence")):
        raise HarnessError("Source discovery requires supported=true|false and official-source/search evidence")
    if role == "environment":
        project = data.get("project") or {}
        if not project.get("name") or not project.get("requires-python") or "dependencies" not in project:
            raise HarnessError("Environment project needs name, requires-python and dependencies")
        evidence = paths["environment_evidence"]
        if not evidence.is_file() or not isinstance(json.loads(evidence.read_text()), dict) or not json.loads(evidence.read_text()):
            raise HarnessError("Environment research requires reproduction/environment.json evidence")
    if role == "interface" and (data.get("compatibility") or {}).get("supported") is False:
        return data
    if role == "hyperparameters":
        config = task.method_dir / "config.toml"
        if not config.is_file() or not tomllib.loads(config.read_text()):
            raise HarnessError("Hyperparameter research must produce a nonempty config.toml")
    if role == "interface" and any(not data.get(key) for key in ("source", "target", "align")):
        raise HarnessError("Interface research requires source, target and align sections")
    if role == "diagnoser":
        if data.get("action") not in ("repair", "report_only") or not data.get("category"):
            raise HarnessError("Diagnosis requires category and action=repair|report_only")
        allowed = {"execution", "environment", "oom", "interface", "dimension", "alignment", "preprocessing"}
        if data.get("action") == "repair" and (data["category"] not in allowed or not repair_evidence(data.get("evidence"))):
            data["action"] = "report_only"
    return data
