"""Fixed TSAD workflow; model turns are limited to artifact-producing workers."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import papers, report, runner, workers
from .agent import run_agent
from .config import HarnessError, ROOT, load
from .task import Phase, Status, Task, now


class UnsupportedPaper(HarnessError):
    """No verified official repository is available for reproduction."""


def evaluation_plan(task: Task) -> list[dict]:
    saved = task.state.get("evaluation_plan")
    source = _read_json(workers.layout(task)["source"])
    interface_path = workers.layout(task)["interface"]
    interface = {}
    if interface_path.is_file():
        import yaml
        interface = yaml.safe_load(interface_path.read_text()) or {}
    compatibility = {**(source.get("compatibility") or {}), **(interface.get("compatibility") or {})}
    if compatibility.get("supported") is False or compatibility.get("task_type", "time_series_anomaly_detection") != "time_series_anomaly_detection":
        raise UnsupportedPaper(compatibility.get("reason") or "Not compatible with EasyTSAD time-series anomaly detection")
    if saved:
        return saved
    defaults = dict(load()["eval"])
    if task.spec.get("evaluation"):
        plans = [{**defaults, **task.spec["evaluation"]}]
    elif not compatibility.get("dataset_types"):
        previous = task.dir / "runs/eval.toml"
        if previous.is_file():
            import tomllib
            defaults.update(tomllib.loads(previous.read_text()))
        plans = [defaults]
    else:
        plans = []
        for dtype in compatibility["dataset_types"]:
            schemas = compatibility.get("schemas") or ["mts" if dtype == "MTS" else "naive"]
            schema = "mts" if dtype == "MTS" and "mts" in schemas else "naive" if "naive" in schemas else schemas[0]
            directory = ROOT / "datasets" / dtype
            datasets = list(defaults["datasets"]) if dtype == defaults.get("dataset_type", "UTS") else sorted(
                path.name for path in directory.iterdir() if path.is_dir()) if directory.is_dir() else []
            if not datasets:
                raise HarnessError(f"No prepared {dtype} datasets are available")
            plans.append({**defaults, "dataset_type": dtype, "schema": schema, "datasets": datasets})
    for plan in plans:
        plan["compute_device"] = compatibility.get("compute_device", "auto")
    task.set(evaluation_plan=plans)
    return plans


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if any(part in (".git", "__pycache__", ".venv", "venv") for part in path.relative_to(directory).parts):
            continue
        if path.is_file() and path.suffix not in (".pyc", ".pyo"):
            yield path


def fetch_paper(task: Task) -> dict:
    """Download text for discovery; cloning waits for the fetcher's decision."""
    task.method_dir.mkdir(parents=True, exist_ok=True)
    canonical = task.method_dir / "paper"
    compatibility = task.dir / "paper"
    if compatibility.exists() and not compatibility.is_symlink():
        if not canonical.exists():
            compatibility.rename(canonical)
        elif not any(compatibility.iterdir()):
            compatibility.rmdir()
        else:
            raise HarnessError("Conflicting paper directories require inspection before migration")
    canonical.mkdir(parents=True, exist_ok=True)
    if not compatibility.exists():
        compatibility.symlink_to(os.path.relpath(canonical, compatibility.parent), target_is_directory=True)
    meta_path = canonical / "meta.json"
    if meta_path.exists():
        return _read_json(meta_path)
    rec = dict(task.spec.get("paper") or {})
    try:
        meta = papers.fetch_to(task.method_dir, rec, code_url=task.spec.get("code_url", ""), clone_code=False)
    except Exception as error:
        # A title or project URL may not name a PDF yet. Discovery belongs to
        # the FETCH worker, so missing material must not prevent its first turn.
        meta = {"paper": rec, "errors": [str(error)],
                "code": {"url": task.spec.get("code_url", "")}, "discovery_pending": True}
    def relative_paths(value):
        if isinstance(value, dict):
            return {key: relative_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relative_paths(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            candidate = Path(value)
            if candidate.resolve().is_relative_to(ROOT.resolve()):
                return workers.relative(candidate)
        return value
    meta = relative_paths(meta)
    _write_json(meta_path, meta)
    empty_code = task.method_dir / "code"
    if empty_code.is_dir() and not any(empty_code.iterdir()):
        empty_code.rmdir()
    return meta


def prepare_source(task: Task) -> dict:
    """Clone/copy official source into method/upstream, preserving the adapter."""
    record = workers.layout(task)["source"]
    discovery = _read_json(record)
    upstream = task.method_dir / "upstream"
    if discovery.get("supported") is False:
        raise UnsupportedPaper(discovery.get("reason") or "No official repository is available")
    compatibility = discovery.get("compatibility") or {}
    if compatibility.get("supported") is False or compatibility.get("task_type", "time_series_anomaly_detection") != "time_series_anomaly_detection":
        raise UnsupportedPaper(compatibility.get("reason") or "Repository does not implement compatible time-series anomaly detection")
    selected_paper = discovery.get("paper")
    if selected_paper:
        if not isinstance(selected_paper, dict):
            raise HarnessError("FETCH source.paper must be a structured paper record")
        # The Agent selects URLs; only this script downloads and extracts them.
        meta_path = task.method_dir / "paper/meta.json"
        meta = _read_json(meta_path)
        rec = papers.record(**{**task.spec.get("paper", {}), **selected_paper})
        pdf_source = rec.get("pdf_url") or rec.get("pdf_path")
        if not meta.get("pdf") and meta.get("paper_download_attempted") != pdf_source:
            if rec.get("pdf_url") or rec.get("pdf_path"):
                try:
                    fetched = papers.fetch_to(task.method_dir, rec,
                        code_url=discovery.get("repository_url", ""), clone_code=False)
                except HarnessError as error:
                    meta.setdefault("errors", []).append(str(error))
                    meta["paper_download_attempted"] = rec.get("pdf_url") or rec.get("pdf_path")
                    _write_json(meta_path, meta)
                    task.log("paper/download_failed", error=str(error))
                    fetched = meta
                for section in ("pdf", "markdown"):
                    if fetched.get(section, {}).get("path"):
                        fetched[section]["path"] = workers.relative(Path(fetched[section]["path"]))
                if fetched.get("paper", {}).get("pdf_path"):
                    fetched["paper"]["pdf_path"] = workers.relative(Path(fetched["paper"]["pdf_path"]))
                _write_json(meta_path, fetched)
    if discovery.get("commit") and upstream.is_dir():
        return discovery
    meta = _read_json(task.dir / "paper/meta.json")
    code = meta.get("code") or {}
    url = discovery.get("repository_url") or task.spec.get("code_url") or code.get("url")
    if not url:
        raise UnsupportedPaper("No verified official repository is available")
    if not discovery.get("repository_url"):
        raise HarnessError("Official source must be identified by the FETCH worker before cloning")
    source = Path(code["dir"]) if code.get("dir") else None
    if source and not source.is_absolute():
        source = ROOT / source
    if upstream.is_dir() and (upstream / ".git").exists():
        official = subprocess.run(["git", "-C", str(upstream), "remote", "get-url", "origin"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        if official.removesuffix(".git").rstrip("/") != url.removesuffix(".git").rstrip("/"):
            raise HarnessError("Existing upstream belongs to a different repository")
        code = {"url": url, "commit": subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True).stdout.strip()}
        source = upstream
    elif source is None or not source.is_dir() or code.get("url") != url:
        code = papers.clone(url, task.dir / "code", revision=discovery.get("commit"))
        source = Path(code["dir"])
    if upstream.exists() and not any(upstream.iterdir()):
        upstream.rmdir()
    if not upstream.exists():
        shutil.move(str(source), str(upstream))
    # Refuse to treat an unrelated populated tree as this repository.
    elif not discovery.get("commit") and source != upstream:
        raise HarnessError("Unrecorded upstream directory exists; inspect before replacing official source")
    files = sum(1 for path in _files(upstream))
    result = {**discovery, "at": now(), "supported": True, "repository_url": url,
              "commit": code.get("commit"), "upstream": workers.relative(upstream),
              "file_count": files}
    _write_json(record, result)
    task.log("source/staged", url=url, commit=result["commit"], files=files)
    return result


def snapshot_implementation(task: Task) -> dict:
    path = task.method_dir / "config.toml"
    result = {"at": now(), "config_text": path.read_text() if path.exists() else "",
              "environment": runner.method_env(task.method), "source_hash": runner.source_hash(task.method_dir)}
    _write_json(task.dir / "runs/implementation.json", result)
    return result


def handoff(task: Task) -> dict:
    paths = workers.layout(task)
    upstream = task.method_dir / "upstream"
    result = {"source": workers.relative(upstream if upstream.is_dir() else task.method_dir),
              "artifacts": {key: workers.relative(value) for key, value in paths.items()},
              "environment": workers.relative(task.method_dir / "environment"),
              "environment_name": runner.method_env(task.method),
              "legacy_integration": not upstream.is_dir(),
              "human_corrections": task.read_jsonl(task.dir / "corrections.jsonl"),
              "errors": task.state.get("implementation_errors", [])}
    _write_json(task.dir / "handoff.json", result)
    return result


def phase_prompt(task: Task, phase: Phase) -> str:
    if phase not in (Phase.IMPLEMENT, Phase.DIAGNOSE):
        raise ValueError(f"{phase} is not an implementation/diagnosis worker phase")
    return workers.prompt(task, "implementer" if phase is Phase.IMPLEMENT else "diagnoser", handoff(task))


def install_environment(task: Task, **kwargs) -> dict:
    from .environments import install
    return install(task, **kwargs)


def failure_category(error: str, method: str = "") -> str:
    if runner.publication_failure({"error": error}):
        return "publication"
    kind = runner.classify_failure(error, method)
    return {"ENVIRONMENT": "environment", "RESOURCE_EXECUTION": "resource"}.get(kind, "execution")


def evaluation_findings(state: dict) -> list[dict]:
    """Deterministic trigger: actual failures and explicitly abnormal scores only."""
    findings = []
    if not state.get("launched", True) or not state.get("datasets"):
        findings.append({"category": "execution", "message": "Evaluation produced no dataset runs", "evidence": [{"file": state.get("log", ""), "error": state.get("log_tail", "")} ]})
    for dataset, result in state.get("datasets", {}).items():
        if result.get("status") != "success":
            category = "publication" if runner.publication_failure(result) else failure_category(result.get("error", ""))
            findings.append({"category": category, "message": f"{dataset}: {result.get('status')}",
                             "evidence": [{"file": result.get("manifest") or result.get("run_dir", ""), "error": result.get("error", "")} ]})
        diagnostics = result.get("diagnostics") or {}
        # The runner reads actual arrays/manifests. Never infer failure from a
        # merely low metric or the scientific validity of a method.
        for key in ("nonfinite", "empty", "wrongshape", "constant", "all_zero", "core_f1_zero"):
            value = diagnostics.get(key)
            if value:
                findings.append({"category": "score_anomaly", "message": f"{dataset}: {key}",
                                 "evidence": [{"file": result.get("manifest") or result.get("run_dir", ""), "diagnostic": key, "details": value}]})
    return findings


def _call(task: Task, role: str, context=None):
    retries = int(load()["budget"].get("max_transient_resumes", 3))
    for attempt in range(retries + 1):
        result = run_agent(task, workers.prompt(task, role, context), label=role, role=role)
        if result.ok:
            return workers.validate(task, role)
        if attempt >= retries or not any(marker in result.error for marker in ("TRANSPORT", "TIMEOUT", "RATE_LIMIT", "SERVER", "PI_AI_ERROR", "ECONNRESET")):
            raise HarnessError(f"{role}: {result.reason}: {result.error[-2000:]}")


def _agent_phase(task: Task, phase: Phase):
    """Compatibility entrypoint for callers using the local pipeline."""
    _call(task, "implementer", handoff(task))
    workers.validate(task, "implementer")
    check = runner.check_method(task)
    if not check.get("ok"):
        task.set(phase=Phase.DIAGNOSE, status=Status.READY,
                 implementation_errors=[json.dumps(check, ensure_ascii=False)])
        return
    snapshot_implementation(task)
    task.set(phase=Phase.EVALUATE, status=Status.READY, eval=None)


def step(task: Task, *, gpus: str | None = None) -> None:
    if task.status in (Status.AWAITING_HUMAN, Status.FAILED, Status.DONE):
        return
    phase = task.phase
    if phase is Phase.FETCH:
        meta = fetch_paper(task)
        _call(task, "fetcher", {"paper_metadata": meta})
        prepare_source(task)
        task.set(phase=Phase.RESEARCH, status=Status.READY)
    elif phase in (Phase.RESEARCH, Phase.EVIDENCE):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_call, task, role) for role in ("hyperparameters", "interface", "environment")]
            for future in futures:
                future.result()
        install_environment(task)
        task.set(phase=Phase.IMPLEMENT, status=Status.READY)
    elif phase is Phase.IMPLEMENT:
        _agent_phase(task, phase)
    elif phase is Phase.SMOKE:
        task.set(phase=Phase.EVALUATE, status=Status.READY)
    elif phase is Phase.EVALUATE:
        if not task.state.get("eval"):
            try:
                info = runner.launch_eval(task, gpus)
            except HarnessError as error:
                if "RESOURCE_EXECUTION (retryable)" not in str(error):
                    raise
                task.set(waiting_resources=True, reason=str(error))
                return
            task.set(status=Status.RUNNING, eval=info)
            return
        state = runner.eval_status(task)
        if state.get("running"):
            return
        findings = evaluation_findings(state)
        task.set(phase=Phase.DIAGNOSE if findings else Phase.REPORT, status=Status.READY,
                 evaluation_findings=findings, evaluation_failed=not state.get("done"), eval=None)
    elif phase is Phase.DIAGNOSE:
        diagnosis = _call(task, "diagnoser", {"findings": task.state.get("evaluation_findings"),
                                             "implementation_errors": task.state.get("implementation_errors")})
        diagnostic_path = task.method_dir / "reproduction/diagnostics.jsonl"
        with open(diagnostic_path, "a") as stream:
            stream.write(json.dumps({"at": now(), **diagnosis}, ensure_ascii=False) + "\n")
        repairs = task.counter("eval_repairs")
        if diagnosis["action"] == "repair" and repairs < min(4, int(load()["budget"]["max_repair_rounds"])):
            task.bump("eval_repairs")
            task.set(phase=Phase.IMPLEMENT, status=Status.READY, implementation_errors=[diagnosis])
        elif task.state.get("evaluation_failed") or task.state.get("implementation_errors"):
            task.set(status=Status.FAILED, reason="No evidenced repair or repair budget exhausted")
        else:
            task.set(phase=Phase.REPORT, status=Status.READY)
    elif phase is Phase.REPORT:
        benchmark = report.finalize(task)
        task.set(phase=Phase.DONE, status=Status.DONE, report=workers.relative(task.method_dir / "README.md"),
                 benchmark=workers.relative(Path(benchmark)))


def run(task: Task, *, until: Phase | None = None, wait: bool = True, gpus: str | None = None,
        poll_s: int = 60) -> Task:
    with task.lock():
        task.reload()
        if task.phase in (Phase.EVIDENCE, Phase.SMOKE):
            task.set(phase=task.phase.next, status=Status.READY)
        while True:
            task.reload()
            if until is not None and task.phase is until:
                return task
            before = task.phase, task.status
            try:
                step(task, gpus=gpus)
            except Exception as error:
                with contextlib.suppress(Exception):
                    task.reload()
                task.log("pipeline/error", step=before[0].value, error=repr(error), traceback=traceback.format_exc())
                task.set(status=Status.FAILED, reason=f"{before[0].value}: {error}"[:1200])
                return task
            if task.status in (Status.AWAITING_HUMAN, Status.FAILED, Status.DONE):
                return task
            if task.phase is Phase.EVALUATE and task.status is Status.RUNNING:
                if not wait:
                    return task
                time.sleep(poll_s)
            elif (task.phase, task.status) == before:
                if wait and task.state.get("waiting_resources"):
                    time.sleep(poll_s)
                else:
                    return task
