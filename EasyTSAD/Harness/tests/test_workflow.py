"""Fixed phase, scoped worker, and process recovery regressions (no model calls)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tsad_harness import agent, pipeline, scheduler, task as task_module, worker_tools, workers
from tsad_harness.config import ROOT, HarnessError
from tsad_harness.store import Store
from tsad_harness.task import Phase, Status, Task


class FakeConfig:
    def __init__(self, root):
        self.methods_dir = root / "methods"
        self.tasks_dir = root / "tasks"
        self.raw = {"interactive": {"max_papers": 2, "max_model_requests": 4}}
        self.sections = {"report": {"default_family": "deep"},
                         "budget": {"max_repair_rounds": 4},
                         "eval": {"datasets": ["A"], "schema": "naive", "gpus": "cpu"}}
    def __getitem__(self, key):
        return self.sections[key]


class Fixture:
    def setup_fixture(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsad-workflow-")
        self.root = Path(self.tmp.name)
        self.cfg = FakeConfig(self.root)
        self.patches = [patch.object(module, "load", return_value=self.cfg)
                        for module in (pipeline, scheduler, task_module)]
        self.patches.extend(patch.object(module, "ROOT", self.root)
                            for module in (pipeline, scheduler, task_module, workers, worker_tools, agent))
        for item in self.patches:
            item.start()
        self.task = Task.create("Example", paper={"title": "Example"})
        self.task.method_dir.mkdir(parents=True)
        self.store = Store(self.root / "state.sqlite3")
        self.bridge = SimpleNamespace(error="", call=AsyncMock(return_value={"running": False}))
        self.scheduler = scheduler.Scheduler(self.store, self.bridge)
    def teardown_fixture(self):
        self.store.close()
        for item in self.patches:
            item.stop()
        self.tmp.cleanup()
    def job(self, stage="EVALUATE", **fields):
        return self.store.put("jobs", {"id": "job", "method": self.task.method, "taskId": self.task.id,
            "stage": stage, "status": "queued", "authorized": True, "repairRounds": 0,
            "datasetResults": {}, **fields})


class ArtifactTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_fixture()
    def tearDown(self):
        self.teardown_fixture()
    def test_missing_dependency_classification_accepts_empty_method_and_exact_prefix(self):
        error = "ModuleNotFoundError: No module named 'matplotlib'"
        self.assertEqual(scheduler.runner.classify_failure(error, ""), "ENVIRONMENT")
        self.assertEqual(scheduler.runner.classify_failure(error, "mat"), "ENVIRONMENT")
        self.assertEqual(pipeline.failure_category(error), "environment")
        self.assertEqual(scheduler.runner.classify_failure("ModuleNotFoundError: No module named 'Example.model'", "Example"), "ADAPTER_INTERFACE")
    def test_result_identity_tracks_only_relevant_dataset_configuration(self):
        config = self.task.method_dir / "config.toml"
        plan = {"schema": "naive", "dataset_type": "UTS", "datasets": ["A", "B"], "seed": 1}
        config.write_text("[Model_Params.Default]\nwindow = 10\n[Model_Params.B]\nwindow = 20\n")
        first = scheduler.runner.evaluation_identity(self.task, plan, "A")
        config.write_text("[Model_Params.Default]\nwindow = 10\n[Model_Params.B]\nwindow = 30\n")
        self.assertEqual(first, scheduler.runner.evaluation_identity(self.task, plan, "A"))
        config.write_text("[Model_Params.Default]\nwindow = 11\n")
        self.assertNotEqual(first, scheduler.runner.evaluation_identity(self.task, plan, "A"))
    def test_metric_identity_separates_protocols(self):
        job = self.job(evaluation={"evaluation": {"schema": "naive", "dataset_type": "UTS"}})
        self.scheduler.publish_metric(job, "A", "same-run", {}, False)
        job["evaluation"]["evaluation"]["schema"] = "zero_shot"
        self.scheduler.publish_metric(job, "A", "same-run", {}, False)
        self.assertEqual(len(self.store.all("metrics")), 2)
    def test_multivariate_interface_selects_prepared_multivariate_data(self):
        dataset = self.root / "datasets/MTS/SMD"
        dataset.mkdir(parents=True)
        interface = workers.layout(self.task)["interface"]
        interface.parent.mkdir(parents=True)
        interface.write_text("compatibility:\n  supported: true\n  task_type: time_series_anomaly_detection\n  dataset_types: [MTS]\n  schemas: [mts]\n  compute_device: cuda\n")
        plans = pipeline.evaluation_plan(self.task)
        self.assertEqual(plans[0]["dataset_type"], "MTS")
        self.assertEqual(plans[0]["schema"], "mts")
        self.assertEqual(plans[0]["datasets"], ["SMD"])
        self.assertEqual(plans[0]["compute_device"], "cuda")
    def test_incompatible_interface_cannot_be_overridden_by_saved_plan(self):
        interface = workers.layout(self.task)["interface"]
        interface.parent.mkdir(parents=True)
        interface.write_text("compatibility:\n  supported: false\n  task_type: time_series_classification\n  reason: classification only\n")
        self.task.set(evaluation_plan=[{"schema": "naive", "datasets": ["A"]}])
        with self.assertRaisesRegex(pipeline.UnsupportedPaper, "classification only"):
            pipeline.evaluation_plan(self.task)
    def test_role_writer_rejects_cross_role_and_upstream_mutations(self):
        path = workers.relative(self.task.method_dir / "reproduction/hyperparameters.json")
        worker_tools.write_file(self.task, "hyperparameters", path, '{"evidence": ["paper section 4"]}')
        for denied in (self.task.method_dir / "upstream/model.py", self.task.method_dir / "align.py",
                       self.task.method_dir / "environment/pyproject.toml"):
            with self.assertRaises(HarnessError):
                worker_tools.write_file(self.task, "hyperparameters", workers.relative(denied), "x")
        with self.assertRaises(HarnessError):
            worker_tools.resolve_path("../outside")
        with self.assertRaises(HarnessError):
            worker_tools.resolve_path("/etc/passwd")
        protected = self.root / "protected.py"
        protected.write_text("original")
        assigned = self.task.method_dir / "align.py"
        assigned.symlink_to(protected)
        with self.assertRaises(HarnessError):
            worker_tools.write_file(self.task, "implementer", workers.relative(assigned.parent) + "/align.py", "changed")
        self.assertEqual(protected.read_text(), "original")
    def test_hyperparameters_require_config_without_fixed_provenance_layout(self):
        path = self.task.method_dir / "reproduction/hyperparameters.json"
        path.parent.mkdir()
        path.write_text('{"parameters": [{"value": 100, "evidence": "upstream/train.py:12"}]}')
        with self.assertRaises(HarnessError):
            workers.validate(self.task, "hyperparameters")
        (self.task.method_dir / "config.toml").write_text('[Model_Params.Default]\nwindow = 100\n')
        self.assertEqual(workers.validate(self.task, "hyperparameters")["parameters"][0]["value"], 100)
    def test_unsupported_repository_stops_instead_of_reimplementation(self):
        path = workers.layout(self.task)["source"]
        path.parent.mkdir()
        path.write_text('{"supported": false, "reason": "No official repository"}')
        with self.assertRaises(pipeline.UnsupportedPaper):
            pipeline.prepare_source(self.task)
    def test_paper_has_one_canonical_copy_and_relative_metadata(self):
        def fetch(root, record, **kwargs):
            self.assertEqual(root, self.task.method_dir)
            directory = root / "paper"
            directory.mkdir(exist_ok=True)
            (directory / "paper.md").write_text("Paper content")
            (root / "code").mkdir(exist_ok=True)
            return {"markdown": {"path": str(directory / "paper.md")}, "paper": record}
        with patch.object(pipeline.papers, "fetch_to", side_effect=fetch):
            meta = pipeline.fetch_paper(self.task)
        self.assertTrue((self.task.dir / "paper").is_symlink())
        self.assertEqual((self.task.method_dir / "paper/paper.md").read_text(), "Paper content")
        self.assertEqual(meta["markdown"]["path"], workers.relative(self.task.method_dir / "paper/paper.md"))
        self.assertFalse((self.task.method_dir / "code").exists())
    def test_clone_staging_moves_into_upstream_without_duplicate_source(self):
        source_record = workers.layout(self.task)["source"]
        source_record.parent.mkdir()
        source_record.write_text('{"repository_url":"https://github.com/author/official","supported":true,"evidence":["paper"]}')
        staged = self.task.dir / "code/official"
        def clone(url, root, *, revision=None):
            staged.mkdir()
            (staged / "model.py").write_text("official algorithm")
            return {"url": url, "dir": str(staged), "commit": "actual-commit"}
        with patch.object(pipeline.papers, "clone", side_effect=clone):
            provenance = pipeline.prepare_source(self.task)
        self.assertFalse(staged.exists())
        self.assertEqual((self.task.method_dir / "upstream/model.py").read_text(), "official algorithm")
        self.assertEqual(provenance["commit"], "actual-commit")
        self.assertFalse(Path(provenance["upstream"]).is_absolute())
    def test_scientific_diagnosis_cannot_request_repair(self):
        path = workers.layout(self.task)["diagnosis"]
        path.parent.mkdir()
        path.write_text(json.dumps({"category": "leakage", "action": "repair", "evidence": ["paper:4"]}))
        self.assertEqual(workers.validate(self.task, "diagnoser")["action"], "report_only")
    def test_repair_requires_inspectable_source_or_actual_log_text(self):
        source = self.task.method_dir / "align.py"
        source.write_text("def align(scores):\n    return scores[1:]\n")
        name = workers.relative(source)
        self.assertFalse(workers.repair_evidence("low F1"))
        self.assertFalse(workers.repair_evidence([{"file": name, "line": 999}]))
        self.assertFalse(workers.repair_evidence([{"file": name, "error": "invented traceback"}]))
        self.assertTrue(workers.repair_evidence([{"file": name, "line": 2}]))
        self.assertTrue(workers.repair_evidence([{"file": name, "location": "return scores[1:]"}]))
    def test_shared_gate_reclaims_dead_pid_and_preserves_owner_grace(self):
        runtime = self.root / "runtime"
        slot = runtime / "model-permits/0"
        slot.mkdir(parents=True)
        self.assertFalse(agent._stale_permit(slot))
        (slot / "owner.json").write_text('{"pid":99999999,"identity":"1","leaseId":"stale"}')
        self.cfg.raw["interactive"]["max_model_requests"] = 1
        with patch.object(agent, "load", return_value=self.cfg), patch.dict(os.environ, {"TSAD_HARNESS_RUNTIME": str(runtime)}):
            with agent.model_gate() as permit:
                self.assertEqual(permit, slot)
                owner = json.loads((permit / "owner.json").read_text())
                self.assertEqual(owner["pid"], os.getpid())
                self.assertNotEqual(owner["leaseId"], "stale")
            self.assertFalse(slot.exists())
    def test_low_score_is_not_a_diagnosis_trigger(self):
        state = {"datasets": {"A": {"status": "success", "metrics": {"point_f1": {"f1": .01}},
                                     "diagnostics": {"constant": False, "core_f1_zero": False}}}}
        self.assertEqual(pipeline.evaluation_findings(state), [])
        for diagnostic in ("nonfinite", "empty", "wrongshape", "constant", "all_zero", "core_f1_zero"):
            state["datasets"]["A"]["diagnostics"] = {diagnostic: True}
            self.assertEqual(pipeline.evaluation_findings(state)[0]["category"], "score_anomaly")
    def test_research_waits_for_all_three_before_uv_install(self):
        self.task.set(phase=Phase.RESEARCH)
        started = set()
        barrier = threading.Barrier(3, timeout=5)
        def research(task, role, context=None):
            started.add(role)
            barrier.wait()
            return {"evidence": ["source:1"]}
        def install(task):
            self.assertEqual(started, {"hyperparameters", "interface", "environment"})
            return {}
        with patch.object(pipeline, "_call", side_effect=research), patch.object(pipeline, "install_environment", side_effect=install):
            pipeline.step(self.task)
        self.assertIs(self.task.phase, Phase.IMPLEMENT)


class AgentLaunchTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_fixture()
        self.cfg.sections["budget"].update(max_wall_minutes_per_run=1, max_steps_per_run=4)
        self.cfg.sections["dsh"] = {"profile": "headless"}
        self.cfg.path = lambda *args: self.root / "trim.patch.yml"
        self.cfg.dsh_home = self.root / "runtime/dsh"
        self.cfg.agent_env = lambda: dict(os.environ)
        self.runtime = tempfile.TemporaryDirectory(prefix="tsad-pinned-node-")
        self.addCleanup(self.runtime.cleanup)
        external = Path(self.runtime.name)
        binary = external / "node/bin/node"
        binary.parent.mkdir(parents=True)
        binary.write_text(f"#!{sys.executable}\nimport sys\nsys.stdin.read()\n"
                          "print('{\"type\":\"session\",\"sessionId\":\"real-child\"}',flush=True)\n"
                          "print('{\"type\":\"status\",\"phase\":\"turn_end\",\"reason\":{\"kind\":\"completed\"}}',flush=True)\n")
        binary.chmod(0o755)
        self.harness = self.root / "harness"
        (self.harness / "dsh-host").mkdir(parents=True)
        (self.harness / "dsh-host/node_modules").symlink_to(external, target_is_directory=True)
        self.runtime_patch = self.task.method_dir / "reproduction/agents/implementer/runtime.patch.yml"
        self.runtime_patch.parent.mkdir(parents=True)
        self.runtime_patch.write_text("[]")
        self.patches.extend([patch.object(agent, "load", return_value=self.cfg),
                             patch.object(agent, "HARNESS_DIR", self.harness),
                             patch.object(agent, "render_patch", return_value=self.runtime_patch),
                             patch.object(agent, "_worker_mcp", return_value=contextlib.nullcontext({
                                 "pid": 0, "status": "running", "url": "http://127.0.0.1:1/mcp", "kind": "worker-tools"}))])
        for item in self.patches[-4:]:
            item.start()
    def tearDown(self):
        self.teardown_fixture()
    def test_symlinked_pinned_node_argv_is_lexically_repository_relative(self):
        result = agent._run_agent(self.task, "No model", label="test")
        self.assertTrue(result.ok)
        ledger = json.loads((self.runtime_patch.parent / "process.json").read_text())
        self.assertEqual(ledger["argv"][0], "harness/dsh-host/node_modules/node/bin/node")
        self.assertEqual(ledger["status"], "completed")
        self.assertEqual(ledger["sessionId"], "real-child")
    def test_fresh_session_ignores_saved_role_session_and_records_budget(self):
        session = self.runtime_patch.parent / "session.json"
        session.write_text(json.dumps({"sessionId": "old-session"}))
        result = agent._run_agent(self.task, "Use existing files", label="test", role="implementer",
                                  session_id="old-session", fresh_session=True, max_output_tokens=32768)
        self.assertTrue(result.ok)
        ledger = json.loads((self.runtime_patch.parent / "process.json").read_text())
        self.assertNotIn("--session-id", ledger["argv"])
        self.assertTrue(ledger["freshSession"])
        self.assertEqual(ledger["maxOutputTokens"], 32768)
    def test_ledger_initialization_failure_reaps_the_spawned_worker(self):
        spawned = []
        actual_popen = agent.subprocess.Popen
        def spawn(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            spawned.append(child)
            return child
        try:
            with patch.object(agent.subprocess, "Popen", side_effect=spawn), \
                 patch.object(agent, "_write_json", side_effect=OSError("ledger unavailable")):
                with self.assertRaisesRegex(OSError, "ledger unavailable"):
                    agent._run_agent(self.task, "No model", label="test")
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll(), "Initialization failure left the worker process alive")
            self.assertFalse(Path(f"/proc/{spawned[0].pid}").exists(), "Child process was not reaped")
            self.assertTrue(spawned[0].stdin.closed)
            self.assertTrue(spawned[0].stdout.closed)
        finally:
            for child in spawned:
                if child.poll() is None:
                    agent._terminate(child)
                    child.wait(timeout=5)
    def test_completed_text_does_not_hide_scoped_mcp_startup_failure(self):
        binary = self.harness / "dsh-host/node_modules/node/bin/node"
        binary.write_text(binary.read_text() + "sys.stderr.write('tsad-tools (@deepseek-ai/dsh-mcp-client): Error: mcp-client(tsad): initial connection or tool synchronization failed\\n')\n")
        result = agent._run_agent(self.task, "No model", label="test")
        self.assertFalse(result.ok)
        self.assertIn("tsad-tools", result.error)
        ledger = json.loads((self.runtime_patch.parent / "process.json").read_text())
        self.assertEqual(ledger["status"], "failed")

    def test_output_limit_exit_preserves_structured_reason_and_session(self):
        binary = self.harness / "dsh-host/node_modules/node/bin/node"
        binary.write_text(binary.read_text().replace('"kind":"completed"', '"kind":"max-tokens"')
                          + "sys.exit(1)\n")
        result = agent._run_agent(self.task, "No model", label="test")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "max-tokens")
        ledger = json.loads((self.runtime_patch.parent / "process.json").read_text())
        self.assertEqual(ledger["reason"], "max-tokens")
        self.assertEqual(ledger["sessionId"], "real-child")
        self.assertEqual(ledger["exitCode"], 1)


class SchedulerTests(Fixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.setup_fixture()
        asyncio.get_running_loop().slow_callback_duration = 30
    async def asyncTearDown(self):
        await self.scheduler.close()
        self.teardown_fixture()
    async def test_output_budget_and_session_follow_persisted_progress(self):
        job = self.job(stage="IMPLEMENT")
        result = agent.RunResult("old-session", "max-tokens", reached_model=True)
        with patch.object(scheduler.agent, "run_agent", return_value=result) as launch:
            for index in range(3):
                with self.assertRaises(agent.WorkerOutputLimitError):
                    await self.scheduler.run_worker(self.store.get("jobs", "job"), self.task, "implementer")
                if index == 0:
                    self.scheduler = scheduler.Scheduler(self.store, self.bridge)
        calls = launch.call_args_list
        self.assertEqual([call.kwargs["max_output_tokens"] for call in calls], [8192, 16384, 32768])
        self.assertEqual([call.kwargs["fresh_session"] for call in calls], [False, False, True])
        self.assertEqual(self.store.get("jobs", job["id"])["repairRounds"], 0)

    async def test_successful_output_budget_survives_new_attempt_and_scheduler_restart(self):
        self.job(stage="IMPLEMENT")
        with patch.object(scheduler.agent, "run_agent", return_value=agent.RunResult("session", "max-tokens", reached_model=True)):
            with self.assertRaises(agent.WorkerOutputLimitError):
                await self.scheduler.run_worker(self.store.get("jobs", "job"), self.task, "implementer")
        with patch.object(scheduler.agent, "run_agent", return_value=agent.RunResult("session", "completed")) as launch, \
             patch.object(workers, "validate", return_value={"adapter": "delivered"}):
            await self.scheduler.run_worker(self.store.get("jobs", "job"), self.task, "implementer")
            self.store.update("jobs", "job", attemptId="new-attempt", repairRounds=1)
            self.scheduler = scheduler.Scheduler(self.store, self.bridge)
            await self.scheduler.run_worker(self.store.get("jobs", "job"), self.task, "implementer")
        self.assertEqual([call.kwargs["max_output_tokens"] for call in launch.call_args_list], [16384, 16384])

    async def test_environment_diagnosis_routes_before_exhausted_algorithm_budget(self):
        self.job(stage="DIAGNOSE", repairRounds=4, evaluationFailed=True,
                 diagnosticFindings=[{"category": "execution", "evidence": [{"error": "deferred diagnostic"}]}])
        diagnosis = {"category": "environment", "action": "repair", "message": "broken installed package",
                     "evidence": [{"file": "environment-install.log", "error": "missing external dependency"}]}
        with patch.object(self.scheduler, "run_worker", AsyncMock(return_value=diagnosis)) as worker:
            await self.scheduler.work("job")
        self.assertEqual(worker.await_count, 1)
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "RESEARCH")
        self.assertEqual(current["repairRounds"], 4)
        self.assertEqual(current["environmentRepairs"], 1)
        self.assertTrue(current["authorized"])

    async def test_import_failure_goes_directly_to_environment_recovery(self):
        self.job(stage="IMPLEMENT", repairRounds=0)
        check = {"ok": False, "error": "ModuleNotFoundError: No module named 'matplotlib'"}
        with patch.object(scheduler.runner, "check_method", return_value=check), \
             patch.object(self.scheduler, "run_worker", AsyncMock()) as worker:
            self.assertFalse(await self.scheduler.finish_implementation("job", self.task))
        worker.assert_not_called()
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "RESEARCH")
        self.assertEqual(current["repairRounds"], 0)
        self.assertEqual(current["environmentRepairs"], 1)

    async def test_environment_repair_installs_and_checks_without_implementation_worker(self):
        self.job(stage="RESEARCH", environmentError="missing installed package", environmentRepairs=1,
                 environmentRecovery={"returnStage": "IMPLEMENT"})
        plan = {"mode": "reinstall", "packages": ["matplotlib"], "reason": "installed metadata without import"}
        with patch.object(scheduler.environments, "plan_recovery", return_value=plan, create=True), \
             patch.object(workers, "validate", return_value={"adapter": "existing"}), \
             patch.object(self.scheduler, "install_environment", AsyncMock()) as install, \
             patch.object(self.scheduler, "run_worker", AsyncMock()) as worker, \
             patch.object(scheduler.runner, "check_method", return_value={"ok": True}), \
             patch.object(pipeline, "snapshot_implementation"):
            await self.scheduler.repair_environment("job", self.task)
        install.assert_awaited_once_with("job", self.task, recovery=plan)
        worker.assert_not_called()
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "EVALUATE")
        self.assertTrue(current["environmentRecovery"]["verifiedImport"])
        self.assertEqual(current["environmentRepairs"], 1)

    async def test_repair_counter_migration_preserves_algorithm_failures_and_history(self):
        job = self.job(stage="DIAGNOSE", repairRounds=3)
        for index, category, error in ((1, "execution", "git clone failed: HTTP 407"),
                                       (2, "environment", "missing package"),
                                       (3, "dimension", "shapes cannot be multiplied"),
                                       (4, "environment", "diagnosed but budget already exhausted")):
            self.store.put("diagnostics", {"id": str(index), "jobId": "job", "createdAt": str(index),
                "category": category, "action": "repair", "evidence": [{"error": error}]})
        current = self.scheduler.reconcile_repair_rounds(job)
        self.assertEqual(current["repairRounds"], 1)
        self.assertEqual(current["legacyRepairRounds"]["count"], 3)
        self.assertEqual(current["legacyRepairRounds"]["reclassifiedDiagnostics"], ["1", "2"])
        self.assertEqual(self.scheduler.reconcile_repair_rounds(current)["repairRounds"], 1)

    async def test_output_limit_with_delivered_files_completes_handoff(self):
        self.job(stage="IMPLEMENT", runtimeRetries=3)
        self.store.put("messages", {"id": "correction", "jobId": "job", "role": "implementer", "status": "queued", "text": "Finish adapter"})
        adapter = self.task.method_dir / "Example.py"
        def deliver(*args, **kwargs):
            adapter.write_text("class Example: pass\n")
            return agent.RunResult("session", "max-tokens", reached_model=True, tool_calls=1)
        with patch.object(scheduler.agent, "run_agent", side_effect=deliver), \
             patch.object(workers, "validate", return_value={"adapter": str(adapter)}):
            result = await self.scheduler.run_worker(self.store.get("jobs", "job"), self.task, "implementer")
        self.assertEqual(result["adapter"], str(adapter))
        self.assertEqual(self.store.get("jobs", "job")["runtimeRetries"], 0)
        self.assertEqual(self.store.get("messages", "correction")["status"], "delivered")

    async def test_legacy_publication_failure_routes_without_model_or_algorithm_repair(self):
        manifest = self.task.method_dir / "reproduction/runs/attempt/naive/A/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"schema": "naive", "dataset": "A", "attempt_id": "attempt"}))
        finding = {"category": "publication", "evidence": [{"file": str(manifest.relative_to(self.root))}]}
        self.job(stage="DIAGNOSE", attemptId="attempt", diagnosticFindings=[finding], datasetResults={"B": {"status": "success"}})
        with patch.object(self.scheduler, "run_worker", AsyncMock()) as worker:
            self.scheduler.shutdown = False
            with patch.object(self.scheduler, "evaluate", AsyncMock()):
                await self.scheduler.work("job")
        worker.assert_not_called()
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "EVALUATE")
        self.assertEqual(current["publicationRecovery"]["attemptId"], "attempt")
        self.assertEqual(current["datasetResults"]["B"]["status"], "success")
        self.assertEqual(current["repairRounds"], 0)

    async def test_environment_install_failure_does_not_enter_algorithm_diagnosis(self):
        self.job(stage="IMPLEMENT")
        with patch.object(self.scheduler, "run_worker", AsyncMock()), \
             patch.object(self.scheduler, "install_environment", AsyncMock(side_effect=scheduler.environments.EnvironmentInstallError("missing package"))):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "RESEARCH")
        self.assertEqual(current["environmentRepairs"], 1)
        self.assertEqual(current["repairRounds"], 0)
        self.assertIn("missing package", current["environmentError"])

    async def test_install_concurrency_learns_success_and_backs_off_after_failure(self):
        self.job()
        self.scheduler.environment_ceiling = 3
        with patch.object(pipeline, "install_environment", return_value={"environment": "example"}):
            await self.scheduler.install_environment("job", self.task)
            await self.scheduler.install_environment("job", self.task)
        self.assertEqual(self.scheduler.environment_policy["limit"], 2)
        with patch.object(pipeline, "install_environment", side_effect=scheduler.environments.EnvironmentStorageError("quota")):
            with self.assertRaises(scheduler.environments.EnvironmentStorageError):
                await self.scheduler.install_environment("job", self.task)
        self.assertEqual(self.scheduler.environment_policy["limit"], 1)
        self.assertEqual(self.scheduler.environment_active, 0)
        recovered = scheduler.Scheduler(self.store, self.bridge)
        self.assertEqual(recovered.environment_policy["limit"], 1)

    async def test_cuda_oom_changes_device_requirement_without_changing_method_parameters(self):
        config = self.task.method_dir / "config.toml"
        config.write_text("[Model_Params.Default]\nepochs = 100\nbatch_size = 64\n")
        self.job(stage="DIAGNOSE", lastEvaluationResources={"gpus": "0", "devices": [
            {"index": 0, "total_memory_mib": 24000}]})
        finding = {"category": "execution", "evidence": [{"error": "torch.OutOfMemoryError: CUDA out of memory"}]}
        self.assertTrue(self.scheduler.route_runtime_failure("job", self.task, [finding]))
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "EVALUATE")
        self.assertGreater(current["gpuFeedback"]["min_gpu_memory_mib"], 24000)
        self.assertEqual(current["repairRounds"], 0)
        self.assertIn("epochs = 100\nbatch_size = 64", config.read_text())

    async def test_partial_success_survives_failed_sibling_dataset(self):
        self.cfg.sections["eval"]["datasets"] = ["A", "B"]
        info = {"pid": 99999999, "datasets": ["A", "B"], "log": "runs/eval.out"}
        self.task.set(eval=info)
        job = self.job(evaluation=info)
        state = {"running": False, "done": False, "datasets": {
            "A": {"status": "success"}, "B": {"status": "failed", "error": "execution failure"}}}
        with patch.object(self.scheduler, "poll_progress", AsyncMock()), \
             patch.object(scheduler.runner, "eval_status", return_value=state):
            await self.scheduler.evaluate(job, self.task)
        saved = self.store.get("jobs", "job")
        self.assertEqual(saved["datasetResults"]["A"]["status"], "success")
        self.assertNotIn("B", saved["datasetResults"])
        self.assertEqual(saved["stage"], "DIAGNOSE")

    async def test_legacy_success_is_recovered_from_its_manifest(self):
        manifest = self.task.method_dir / "reproduction/runs/old/naive/A/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"status": "success", "method": "Example", "schema": "naive",
            "dataset_type": "UTS", "dataset": "A", "run_id": "prior", "config_hash": "config"}))
        job = self.job(legacy=True)
        with patch.object(scheduler.runner, "find_run", return_value=manifest.parent), \
             patch.object(scheduler.runner, "read_summary", return_value={"auprc": {"value": .5}}):
            saved = self.scheduler.recover_results(job, self.task)
        self.assertTrue(saved["resultsRecovered"])
        self.assertEqual(saved["datasetResults"]["A"]["run_id"], "prior")
        self.assertEqual(saved["datasetResults"]["A"]["config_hash"], "config")

    async def test_successful_evaluate_and_report_never_call_a_model(self):
        info = {"pid": 99999999, "processIdentity": "1", "datasets": ["A"], "log": "runs/eval.out"}
        self.task.set(phase=Phase.EVALUATE, status=Status.RUNNING, eval=info)
        self.job(evaluation=info)
        state = {"launched": True, "running": False, "done": True, "datasets": {"A": {"status": "success", "diagnostics": {}}}}
        benchmark = self.root / "benchmark.json"
        benchmark.write_text('{"methods":[{"id":"Example"}],"revision":"r1"}')
        model = AsyncMock(side_effect=AssertionError("No model call in evaluate/report"))
        with patch.object(self.scheduler, "run_worker", model), patch.object(self.scheduler, "poll_progress", AsyncMock()), \
             patch.object(scheduler.runner, "eval_status", return_value=state), \
             patch.object(scheduler.runner, "launch_eval", side_effect=AssertionError("Never relaunch persisted evaluation")), \
             patch.object(scheduler.report, "finalize", return_value=benchmark, create=True):
            await self.scheduler.work("job")
            await self.scheduler.work("job")
            await self.scheduler.work("job")
        self.assertEqual(self.store.get("jobs", "job")["status"], "done")
        model.assert_not_called()
        self.bridge.call.assert_not_called()
    async def test_abnormal_success_enters_diagnose_without_model_in_evaluate(self):
        info = {"pid": 99999999, "processIdentity": "1", "datasets": ["A"], "log": "runs/eval.out"}
        self.task.set(phase=Phase.EVALUATE, eval=info)
        job = self.job(evaluation=info)
        state = {"running": False, "done": True, "datasets": {"A": {"status": "success", "diagnostics": {"constant": True}}}}
        with patch.object(self.scheduler, "poll_progress", AsyncMock()), patch.object(scheduler.runner, "eval_status", return_value=state), \
             patch.object(self.scheduler, "run_worker", AsyncMock(side_effect=AssertionError("Evaluate is scripted"))):
            await self.scheduler.evaluate(job, self.task)
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "DIAGNOSE")
        self.assertFalse(current["evaluationFailed"])
    async def test_report_only_diagnosis_continues_remaining_datasets(self):
        self.cfg.sections["eval"]["datasets"] = ["A", "B"]
        self.job(stage="DIAGNOSE", evaluationFailed=False, datasetResults={"A": {"status": "success"}})
        diagnose = AsyncMock(return_value={"category": "scientific", "action": "report_only", "message": "Observed constant outputs", "evidence": []})
        evaluate = AsyncMock()
        with patch.object(self.scheduler, "run_worker", diagnose), patch.object(self.scheduler, "evaluate", evaluate), \
             patch.object(scheduler.report, "finalize", side_effect=AssertionError("B has not run"), create=True):
            await self.scheduler.work("job")
        evaluate.assert_awaited_once()
        passed_job = evaluate.await_args.args[0]
        self.assertEqual(passed_job["stage"], "EVALUATE")
        self.assertIn("A", passed_job["datasetResults"])
        self.assertNotIn("B", passed_job["datasetResults"])
    async def test_completion_transaction_rolls_back_without_losing_process(self):
        info = {"pid": 99999999, "processIdentity": "1", "datasets": ["A"], "log": "runs/eval.out"}
        self.task.set(phase=Phase.EVALUATE, eval=info)
        job = self.job(evaluation=info)
        self.store.acquire(["method:Example"], "job")
        state = {"running": False, "done": True, "datasets": {"A": {"status": "success"}}}
        with patch.object(self.scheduler, "poll_progress", AsyncMock()), patch.object(scheduler.runner, "eval_status", return_value=state), \
             patch.object(self.store, "release", side_effect=RuntimeError("interrupted commit")):
            with self.assertRaisesRegex(RuntimeError, "interrupted commit"):
                await self.scheduler.evaluate(job, self.task)
        persisted = self.store.get("jobs", "job")
        self.assertEqual(persisted["evaluation"], info)
        self.assertEqual(persisted["datasetResults"]["A"]["status"], "success")
        self.assertEqual(len(self.store.leases()), 1)
    async def test_canonical_progress_publishes_scored_count_and_durable_cursor(self):
        self.job(evaluation={"datasets": ["A"]}, attemptId="attempt1")
        directory = self.task.method_dir / "reproduction/runs/attempt1/naive/A"
        directory.mkdir(parents=True)
        event = {"type": "curve_evaluated", "curve": "normal", "data": {
            "summary": {"point_f1": {"f1": .4}}, "n_curves_scored": 3,
            "n_curves_evaluated": 2, "total_curves": 3}}
        progress = directory / "progress.jsonl"
        progress.write_text(json.dumps(event) + "\n")
        (directory / "manifest.json").write_text('{"status":"running"}')
        with patch.object(scheduler.runner, "eval_status", return_value={"run_id": "r1"}), \
             patch.object(scheduler.runner, "find_run", return_value=directory):
            await self.scheduler.poll_progress(self.store.get("jobs", "job"))
            await self.scheduler.poll_progress(self.store.get("jobs", "job"))
        current = self.store.get("jobs", "job")
        self.assertEqual(current["progress"]["completed"], 3)
        self.assertEqual(current["progress"]["total"], 3)
        self.assertEqual(self.store.all("metrics")[0]["data"]["n_curves_evaluated"], 2)
        self.assertEqual(self.store.all("cursors")[0]["offset"], progress.stat().st_size)
        self.assertEqual(len([event for event in self.store.events() if event["type"] == "experiment.curve_evaluated"]), 1)
    async def test_diagnosis_repairs_are_capped_at_four(self):
        self.job(stage="DIAGNOSE", repairRounds=4, evaluationFailed=True)
        diagnosis = {"category": "alignment", "action": "repair", "message": "Output index is shifted", "evidence": [{"file": "method.py", "line": 4}]}
        with patch.object(self.scheduler, "run_worker", AsyncMock(return_value=diagnosis)):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["repairRounds"], 4)
        self.assertEqual(current["status"], "failed")
    async def test_correction_resume_preserves_successful_datasets(self):
        results = {"A": {"status": "success", "run_id": "original"}}
        self.job(datasetResults=results)
        await self.scheduler.control("job", "correct", {"message": "Repair the execution failure", "run": True})
        current = self.store.get("jobs", "job")
        self.assertEqual(current["datasetResults"], results)
        self.assertEqual(current["stage"], "IMPLEMENT")
        self.assertTrue(current["authorized"])

    async def test_stale_worker_write_cannot_undo_pause(self):
        self.job()
        self.store.update("jobs", "job", status="paused", authorized=False)
        with self.assertRaises(asyncio.CancelledError):
            self.scheduler.change("job", stage="IMPLEMENT", status="running")
        self.assertEqual(self.store.get("jobs", "job")["status"], "paused")
    async def test_pause_during_launch_tracks_and_clears_the_exact_process(self):
        self.job()
        started, release = threading.Event(), threading.Event()
        info = {"pid": 99999999, "processIdentity": "1", "datasets": ["A"], "log": "runs/eval.out"}
        def launch(*args, **kwargs):
            started.set()
            release.wait(5)
            self.task.set(eval=info)
            return info
        with patch.object(scheduler.runner, "discover_resources", return_value={"gpus": ""}), \
             patch.object(scheduler.runner, "method_env", return_value=None), \
             patch.object(scheduler.runner, "launch_eval", side_effect=launch):
            turn = asyncio.create_task(self.scheduler.work("job"))
            self.scheduler.workers["job"] = turn
            while not started.is_set():
                await asyncio.sleep(.01)
            pause = asyncio.create_task(self.scheduler.control("job", "pause", {}))
            await asyncio.sleep(.01)
            release.set()
            await pause
        current = self.store.get("jobs", "job")
        self.assertEqual(current["status"], "paused")
        self.assertFalse(current["authorized"])
        self.assertIsNone(current.get("evaluation"))
        self.assertEqual(self.store.leases(), [])
    async def test_restart_preserves_paused_and_cancelled_intent(self):
        self.job(status="paused", authorized=False)
        self.store.put("jobs", {"id": "cancelled", "method": "Other", "status": "cancelled", "authorized": False})
        with patch.object(self.scheduler, "import_legacy"), patch.object(self.scheduler, "loop", AsyncMock()):
            await self.scheduler.start()
        self.assertEqual(self.store.get("jobs", "job")["status"], "paused")
        self.assertEqual(self.store.get("jobs", "cancelled")["status"], "cancelled")
    async def test_completed_role_resume_uses_its_session_and_is_read_only(self):
        job = self.job(stage="REPORT", authorized=False, status="done", workerProcesses={"interface": {"status": "completed", "sessionId": "s1"}})
        captured = {}
        def fake_agent(task, prompt, **kwargs):
            captured.update(kwargs)
            return agent.RunResult("s1", "completed", final_text="Evidence explanation")
        with patch.object(scheduler.agent, "run_agent", side_effect=fake_agent):
            result = await self.scheduler.run_worker(job, self.task, "interface", read_only=True, message="Explain score alignment")
        self.assertEqual(captured["session_id"], "s1")
        self.assertFalse(captured["write_enabled"])
        self.assertEqual(result["finalText"], "Evidence explanation")
    async def test_completed_startup_check_is_not_a_completed_worker_turn(self):
        job = self.job(stage="IMPLEMENT", workerProcesses={"implementer": {
            "status": "completed", "kind": "worker-startup", "stageToken": "IMPLEMENT:initial:0"}})
        with patch.object(scheduler.agent, "run_agent", return_value=agent.RunResult(None, "completed")) as run, \
             patch.object(scheduler.workers, "validate", return_value={"files": []}):
            await self.scheduler.run_worker(job, self.task, "implementer")
        run.assert_called_once()
    async def test_invalid_completed_handoff_resumes_its_original_worker(self):
        job = self.job(stage="RESEARCH", workerProcesses={"interface": {
            "status": "completed", "sessionId": "interface-session", "stageToken": "RESEARCH:initial:0"}})
        artifact = workers.layout(self.task)["interface"]
        artifact.parent.mkdir(parents=True)
        artifact.write_text('evidence:\n  - source\n  unresolved: invalid indentation\n')
        captured = {}
        def repair_handoff(task, prompt, **kwargs):
            captured.update(kwargs, prompt=prompt)
            artifact.write_text('source: official\ntarget: EasyTSAD\nalign: pointwise\n')
            return agent.RunResult("interface-session", "completed", reached_model=True)
        with patch.object(scheduler.agent, "run_agent", side_effect=repair_handoff):
            result = await self.scheduler.run_worker(job, self.task, "interface")
        self.assertEqual(result["align"], "pointwise")
        self.assertEqual(captured["session_id"], "interface-session")
        self.assertIn("Invalid interface artifact", captured["prompt"])

    async def test_invalid_handoff_retries_without_algorithm_diagnosis(self):
        self.job(stage="DIAGNOSE", runtimeRetries=2, runtimeRetryKey="DIAGNOSE:worker_handoff")
        artifact = workers.layout(self.task)["diagnosis"]
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{invalid JSON')
        result = agent.RunResult("diagnoser-session", "completed", reached_model=True)
        with patch.object(scheduler.agent, "run_agent", return_value=result):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "DIAGNOSE")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["runtimeRetryKey"], "DIAGNOSE:worker_handoff")
        self.assertEqual(current["runtimeRetries"], 3)
        self.assertEqual(current["repairRounds"], 0)

    async def test_storage_quota_waits_without_algorithm_diagnosis_or_retry_exhaustion(self):
        self.job(stage="RESEARCH", runtimeRetries=7, runtimeRetryKey="RESEARCH:environment_storage")
        plans = [{"schema": "naive", "dataset_type": "UTS", "datasets": ["A"]}]
        with patch.object(self.scheduler, "run_worker", AsyncMock(return_value={})), \
             patch.object(scheduler.pipeline, "evaluation_plan", return_value=plans), \
             patch.object(scheduler.pipeline, "install_environment",
                          side_effect=scheduler.environments.EnvironmentStorageError("Disk quota exceeded")):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "RESEARCH")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["runtimeRetryKey"], "RESEARCH:environment_storage")
        self.assertEqual(current["runtimeRetries"], 8)
        self.assertEqual(current["repairRounds"], 0)
        self.assertTrue(current["authorized"])

    async def test_proxy_failure_retries_without_algorithm_repair(self):
        self.job(stage="DIAGNOSE", runtimeRetries=4, runtimeRetryKey="FETCH:source_download")
        result = agent.RunResult(None, "crash", reached_model=True,
            error='lan-proxy API error (502): Upstream connection failed')
        with patch.object(scheduler.agent, "run_agent", return_value=result):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "DIAGNOSE")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["repairRounds"], 0)
        self.assertEqual(current["runtimeRetries"], 1)
        self.assertEqual(current["runtimeRetryKey"], "DIAGNOSE:model_transport")
        self.assertGreater(current["nextRunAt"], scheduler.time.time())
        self.assertTrue(current["authorized"])
    async def test_repeated_proxy_failure_stops_with_runtime_reason(self):
        self.job(stage="DIAGNOSE", runtimeRetries=5, runtimeRetryKey="DIAGNOSE:model_transport")
        with patch.object(self.scheduler, "run_worker", AsyncMock(side_effect=agent.WorkerTransportError("HTTP 502"))):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["status"], "failed")
        self.assertEqual(current["repairRounds"], 0)
        self.assertEqual(self.store.all("diagnostics")[-1]["category"], "harness_runtime")

    async def test_output_limit_retries_without_algorithm_repair(self):
        self.job(stage="DIAGNOSE")
        result = agent.RunResult("preserved-session", "max-tokens", reached_model=True,
                                 error="dsh exit code 1")
        with patch.object(scheduler.agent, "run_agent", return_value=result):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "DIAGNOSE")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["repairRounds"], 0)
        self.assertEqual(current["runtimeRetryKey"], "DIAGNOSE:model_output_limit")
        self.assertTrue(current["authorized"])
    async def test_source_download_failure_retries_fetch_without_algorithm_repair(self):
        self.job(stage="FETCH")
        source = workers.layout(self.task)["source"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps({"supported": True, "repository_url": "https://github.com/author/official",
                                      "evidence": ["official repository"]}))
        with patch.object(pipeline, "fetch_paper") as fetch, \
             patch.object(self.scheduler, "run_worker", AsyncMock()) as worker, \
             patch.object(pipeline, "prepare_source", side_effect=scheduler.papers.SourceFetchError("HTTP 407")):
            await self.scheduler.work("job")
        fetch.assert_not_called()
        worker.assert_not_called()
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "FETCH")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["runtimeRetries"], 1)
        self.assertEqual(current["repairRounds"], 0)
        self.assertGreater(current["nextRunAt"], scheduler.time.time())
    async def test_account_capacity_wait_does_not_exhaust_algorithm_or_transport_budget(self):
        self.job(stage="DIAGNOSE", runtimeRetries=20, runtimeRetryKey="DIAGNOSE:model_capacity")
        with patch.object(self.scheduler, "run_worker", AsyncMock(side_effect=agent.WorkerTransportError("HTTP 503 No available accounts"))):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["runtimeRetries"], 21)
        self.assertEqual(current["repairRounds"], 0)
        self.assertTrue(current["authorized"])
    async def test_one_research_transport_failure_keeps_other_branches_running(self):
        self.job(stage="RESEARCH")
        completed = []
        async def worker(job, task, role, context):
            if role == "interface":
                raise agent.WorkerTransportError("HTTP 503 No available accounts")
            await asyncio.sleep(.01)
            completed.append(role)
            return {}
        with patch.object(self.scheduler, "run_worker", side_effect=worker):
            await self.scheduler.work("job")
        self.assertEqual(set(completed), {"hyperparameters", "environment"})
        current = self.store.get("jobs", "job")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["runtimeRetryKey"], "RESEARCH:model_capacity")
    async def check_stage_exception(self, stage, owner, function):
        self.job(stage=stage)
        with patch.object(owner, function, side_effect=HarnessError(f"{stage} fixture error")):
            await self.scheduler.work("job")
        current = self.store.get("jobs", "job")
        self.assertEqual(current["stage"], "DIAGNOSE")
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["repairTarget"], stage)
        self.assertTrue(current["evaluationFailed"])
        self.assertIn(f"{stage} fixture error", current["diagnosticFindings"][0]["message"])
    async def test_fetch_exception_queues_diagnosis_for_fetch(self):
        await self.check_stage_exception("FETCH", scheduler.pipeline, "fetch_paper")
    async def test_report_exception_queues_diagnosis_for_report(self):
        await self.check_stage_exception("REPORT", scheduler.report, "finalize")
    async def test_orphan_tool_server_blocks_cached_turn_and_holds_the_write_lease(self):
        process = {"kind": "worker-tools", "pid": 123, "processIdentity": "stamp", "status": "running"}
        job = self.job(stage="IMPLEMENT", workerProcesses={"implementer": {"kind": "worker", "status": "completed",
            "stageToken": "IMPLEMENT:initial:0", "ownedProcesses": [process]}})
        self.store.acquire(["method:Example"], "job")
        with patch.object(scheduler, "process_identity", side_effect=lambda pid: "stamp" if pid == 123 else None), \
             patch.object(scheduler.agent, "run_agent") as run:
            self.assertFalse(await self.scheduler.release_if_idle("job"))
            with self.assertRaisesRegex(HarnessError, "tool server is still running"):
                await self.scheduler.run_worker(job, self.task, "implementer")
        run.assert_not_called()
        self.assertEqual(len(self.store.leases()), 1)
    async def test_restart_recovers_tool_server_before_a_cli_ledger_exists(self):
        self.job(stage="RESEARCH", status="running")
        directory = self.task.method_dir / "reproduction/agents/interface"
        directory.mkdir(parents=True)
        (directory / "mcp-process.json").write_text(json.dumps({"kind": "worker-tools", "role": "interface",
            "pid": 123, "processIdentity": "stamp", "status": "running"}))
        with patch.object(scheduler, "process_identity", side_effect=lambda pid: "stamp" if pid == 123 else None), \
             patch.object(self.scheduler, "import_legacy"), patch.object(self.scheduler, "loop", AsyncMock()):
            await self.scheduler.start()
        job = self.store.get("jobs", "job")
        self.assertEqual(job["status"], "paused")
        self.assertFalse(job["authorized"])
        self.assertEqual(job["workerProcesses"]["interface"]["pid"], 123)
    async def test_worker_cancel_stops_an_owned_server_after_cli_exit(self):
        self.job(workerProcesses={"interface": {"kind": "worker", "status": "completed", "ownedProcesses": [
            {"kind": "worker-tools", "pid": 123, "processIdentity": "stamp", "status": "running"}]}})
        live = {123: "stamp"}
        def terminate(pid, sig):
            self.assertEqual(pid, 123)
            live.clear()
        with patch.object(scheduler, "process_identity", side_effect=lambda pid: live.get(pid)), \
             patch.object(scheduler.os, "killpg", side_effect=terminate) as stop:
            await self.scheduler.cancel_roles("job")
        stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
