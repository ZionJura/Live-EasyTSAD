"""Runtime credentials and other workers stay outside scoped source tools."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tsad_harness import worker_tools, workers
from tsad_harness.config import HarnessError


class RuntimeReadScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tsad-worker-read-scope-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.method = self.root / "runtime/verification/methods/Example"
        self.task = self.root / "runtime/verification/tasks/Example"
        self.allowed = (self.task, self.method)
        self.sources = {
            "EasyTSAD/Framework/source.py": "SCOPE public framework",
            "EasyTSAD/Methods/Existing/Existing.py": "SCOPE public method",
            "runtime/verification/methods/Example/Example.py": "SCOPE own method",
            "runtime/verification/tasks/Example/handoff.json": "SCOPE own task",
            "runtime/harness-workbench/access-token": "SCOPE HIDDEN_CREDENTIAL",
            "runtime/verification/methods/Other/Other.py": "SCOPE OTHER_PRIVATE",
            "runtime/verification/scoped-tool-probe.txt": "SCOPE OUTSIDE_ARTIFACT",
        }
        for name, content in self.sources.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content + "\n")
        for module in (worker_tools, workers):
            item = patch.object(module, "ROOT", self.root)
            item.start()
            self.addCleanup(item.stop)
        self.worker_task = SimpleNamespace(dir=self.task, method_dir=self.method, method="Example", spec={})

    def read(self, path):
        return worker_tools._read_file(path, allowed_roots=self.allowed)

    def test_access_token_and_other_runtime_files_are_denied(self):
        for name in ("runtime/harness-workbench/access-token",
                     "runtime/verification/methods/Other/Other.py",
                     "runtime/verification/scoped-tool-probe.txt"):
            with self.subTest(path=name), self.assertRaises(HarnessError):
                self.read(name)

    def test_public_sources_and_current_task_method_runtime_artifacts_are_readable(self):
        for name in list(self.sources)[:4]:
            with self.subTest(path=name):
                self.assertIn(self.sources[name], self.read(name))

    def test_dependency_source_is_readable_without_exposing_environment_credentials(self):
        source = self.root / "envs/example/venv/lib/python3.9/site-packages/prts/base.py"
        source.parent.mkdir(parents=True)
        source.write_text("dependency source")
        self.assertIn("dependency source", self.read(source.relative_to(self.root).as_posix()))
        credential = self.root / "envs/example/venv/access-token"
        credential.write_text("credential")
        with self.assertRaises(HarnessError):
            self.read(credential.relative_to(self.root).as_posix())
        files = json.loads(worker_tools._list_files("envs", allowed_roots=self.allowed))
        self.assertFalse(any("/venv/" in name for name in files))

    def test_numpy_labels_are_read_as_array_metadata(self):
        import numpy as np
        source = self.root / "datasets/UTS/WSD/99/test_label.npy"
        source.parent.mkdir(parents=True)
        np.save(source, np.zeros(4, dtype=np.int8))
        metadata = json.loads(self.read(source.relative_to(self.root).as_posix()))
        self.assertEqual(metadata["shape"], [4])
        self.assertEqual(metadata["dtype"], "int8")
        self.assertEqual(metadata["unique_values"], [0])

    def test_symlinks_cannot_escape_to_credentials_or_other_runtime(self):
        targets = ("runtime/harness-workbench/access-token", "runtime/verification/methods/Other/Other.py")
        for index, target in enumerate(targets):
            for directory in (self.method, self.root / "EasyTSAD/Framework"):
                alias = directory / f"alias{index}.py"
                alias.symlink_to(self.root / target)
                with self.subTest(path=alias), self.assertRaises(HarnessError):
                    self.read(alias.relative_to(self.root).as_posix())
        hidden = self.method / ".credential"
        hidden.write_text("hidden")
        (self.method / "visible.py").symlink_to(hidden)
        with self.assertRaises(HarnessError):
            self.read("runtime/verification/methods/Example/visible.py")

    def test_recursive_listing_filters_other_runtime_and_symlink_targets(self):
        alias = self.method / "leak.py"
        alias.symlink_to(self.root / "runtime/harness-workbench/access-token")
        archive = self.task / "guard/old/source.py"
        archive.parent.mkdir(parents=True)
        archive.write_text("archived source")
        files = json.loads(worker_tools._list_files("", allowed_roots=self.allowed))
        self.assertIn("EasyTSAD/Framework/source.py", files)
        self.assertIn("runtime/verification/methods/Example/Example.py", files)
        self.assertFalse(any("access-token" in name or "/Other/" in name for name in files))
        self.assertNotIn(alias.relative_to(self.root).as_posix(), files)
        self.assertNotIn(archive.relative_to(self.root).as_posix(), files)
        self.assertIn(archive.relative_to(self.root).as_posix(), json.loads(
            worker_tools._list_files((self.task / "guard").relative_to(self.root).as_posix(), allowed_roots=self.allowed)))
        with self.assertRaises(HarnessError):
            worker_tools._list_files("runtime/harness-workbench", allowed_roots=self.allowed)

    def test_recursive_listing_timeout_requests_a_narrower_directory(self):
        with patch.object(worker_tools.time, "monotonic", side_effect=[0, 31]):
            with self.assertRaisesRegex(HarnessError, "narrower source directory"):
                worker_tools._list_files("EasyTSAD", allowed_roots=self.allowed)

    def test_root_search_excludes_runtime_and_method_search_allows_own_sources(self):
        result = worker_tools._search_files("", "SCOPE", allowed_roots=self.allowed)
        self.assertIn("SCOPE public framework", result)
        self.assertNotIn("runtime/", result)
        self.assertNotIn("HIDDEN_CREDENTIAL", result)
        self.assertNotIn("OTHER_PRIVATE", result)
        own = worker_tools._search_files("runtime/verification/methods/Example", "SCOPE", allowed_roots=self.allowed)
        self.assertIn("SCOPE own method", own)
        with self.assertRaises(HarnessError):
            worker_tools._search_files("runtime/harness-workbench/access-token", "SCOPE", allowed_roots=self.allowed)

    def test_search_revalidates_returned_paths_before_exposing_lines(self):
        # A path outside runtime may alias a runtime directory. Even an rg
        # implementation that follows it must not return those secret lines.
        alias = self.root / "EasyTSAD/Framework/leak.py"
        alias.symlink_to(self.root / "runtime/harness-workbench/access-token")
        output = json.dumps({"type": "match", "data": {"path": {"text": "EasyTSAD/Framework/leak.py"},
                            "lines": {"text": "SCOPE HIDDEN_CREDENTIAL\n"}, "line_number": 1}}) + "\n"
        with patch.object(worker_tools.subprocess, "run") as run:
            run.return_value.stdout = output
            self.assertEqual(worker_tools._search_files("", "SCOPE", allowed_roots=self.allowed), "")


    def test_implementer_can_repair_upstream_and_method_environment(self):
        for name in ("upstream/package/model.py", "environment/pyproject.toml", "helpers/score.py"):
            path = self.method / name
            self.assertTrue(workers.is_write_allowed(self.worker_task, "implementer", path))
            worker_tools.write_file(self.worker_task, "implementer", path.relative_to(self.root).as_posix(), "repair")
            self.assertEqual(path.read_text(), "repair")

    def test_method_write_scope_denies_shared_other_hidden_and_traversal_paths(self):
        for name in ("EasyTSAD/Evaluations/metric.py", "EasyTSAD/Framework/labels.py",
                     "envs/shared/pyproject.toml", "runtime/verification/methods/Other/Other.py",
                     "runtime/verification/methods/ExampleSibling/helper.py",
                     "runtime/verification/tasks/Example/handoff.json",
                     "runtime/verification/methods/Example/.secret",
                     "runtime/verification/methods/Example/access-token",
                     "runtime/verification/methods/Example/venv/config",
                     "runtime/verification/methods/Example/../Other/new.py"):
            with self.subTest(path=name):
                self.assertFalse(workers.is_write_allowed(self.worker_task, "implementer", Path(name)))
                with self.assertRaises(HarnessError):
                    worker_tools.write_file(self.worker_task, "implementer", name, "changed")

    def test_other_roles_keep_exact_artifact_permissions(self):
        for role in ("fetcher", "hyperparameters", "interface", "environment", "diagnoser"):
            for path in workers.write_paths(self.worker_task, role):
                with self.subTest(role=role, path=path):
                    worker_tools.write_file(self.worker_task, role, path.relative_to(self.root).as_posix(), "artifact")
            with self.subTest(role=role), self.assertRaises(HarnessError):
                worker_tools.write_file(self.worker_task, role,
                                        "runtime/verification/methods/Example/upstream/new.py", "changed")

    def test_write_symlinks_must_stay_inside_lexical_method_scope(self):
        outside = self.root / "EasyTSAD/Framework/source.py"
        (self.method / "align.py").symlink_to(outside)
        (self.method / "upstream").symlink_to(outside.parent, target_is_directory=True)
        alias = self.root / "EasyTSAD/Methods/alias"
        alias.symlink_to(self.method, target_is_directory=True)
        hidden = self.method / ".secret"
        hidden.write_text("secret")
        (self.method / "secret.py").symlink_to(hidden)
        for path in (self.method / "align.py", self.method / "upstream/new.py",
                     alias / "helper.py", self.method / "secret.py"):
            with self.subTest(path=path), self.assertRaises(HarnessError):
                worker_tools.write_file(self.worker_task, "implementer", path.relative_to(self.root).as_posix(), "changed")
        self.assertEqual(outside.read_text(), "SCOPE public framework\n")
        self.assertEqual(hidden.read_text(), "secret")
        internal = self.method / "local.py"
        internal.symlink_to(self.method / "Example.py")
        worker_tools.write_file(self.worker_task, "implementer", internal.relative_to(self.root).as_posix(), "repair")
        self.assertEqual((self.method / "Example.py").read_text(), "repair")

    def test_symlinked_method_directory_does_not_grant_write_access(self):
        alias = self.root / "EasyTSAD/Methods/Alias"
        alias.symlink_to(self.method, target_is_directory=True)
        self.worker_task.method_dir = alias
        with self.assertRaises(HarnessError):
            worker_tools.write_file(self.worker_task, "implementer", "EasyTSAD/Methods/Alias/Example.py", "changed")

    def test_read_only_continuation_rejects_method_local_publish(self):
        registered = {}

        def tool(**kwargs):
            def register(function):
                registered[function.__name__] = function
                return function
            return register

        server = SimpleNamespace(tool=tool)
        module = SimpleNamespace(MCPServer=lambda *args, **kwargs: server)
        (self.task / "task.toml").write_text("")
        (self.task / "state.json").write_text("{}")
        with patch.dict(sys.modules, {"mcp.server.mcpserver": module}), \
                patch.object(worker_tools, "Task", return_value=self.worker_task), \
                patch.dict(os.environ, {"TSAD_TASK_DIR": str(self.task), "TSAD_WORKER_ROLE": "implementer",
                                        "TSAD_WORKER_READ_ONLY": "1"}), \
                contextlib.redirect_stdout(io.StringIO()):
            worker_tools.main(check_only=True)
            with self.assertRaisesRegex(HarnessError, "read-only"):
                registered["publish_file"]("runtime/verification/methods/Example/upstream/new.py", "repair")
        self.assertFalse((self.method / "upstream/new.py").exists())

    def artifact(self, role, compatibility=None, *, include_compatibility=True):
        data = ({"supported": True, "repository_url": "https://example.org/official", "evidence": ["paper"]}
                if role == "fetcher" else {"source": "official", "target": "EasyTSAD", "align": "timepoint", "evidence": ["source"]})
        if include_compatibility:
            data["compatibility"] = compatibility
        path = workers.layout(self.worker_task)["source" if role == "fetcher" else "interface"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return workers.validate(self.worker_task, role)

    def test_compatibility_is_optional_and_preserves_legacy_artifacts(self):
        for role in ("fetcher", "interface"):
            with self.subTest(role=role):
                self.assertNotIn("compatibility", self.artifact(role, include_compatibility=False))

    def test_supported_compatibility_records_actual_multivariate_schemas(self):
        compatibility = {"task_type": "time_series_anomaly_detection", "supported": True,
                         "dataset_types": ["UTS", "MTS"],
                         "schemas": ["naive", "mts", "all_in_one", "zero_shot", "zero_shot_cross_ds"],
                         "reason": "Official timepoint score entrypoints"}
        for role in ("fetcher", "interface"):
            with self.subTest(role=role):
                self.assertEqual(self.artifact(role, compatibility)["compatibility"], compatibility)

    def test_non_tsad_discovery_can_be_unsupported_without_repository(self):
        data = {"supported": False, "evidence": ["paper describes classification"],
                "compatibility": {"task_type": "time_series_classification", "supported": False,
                                  "dataset_types": [], "schemas": [], "reason": "Classification only"}}
        path = workers.layout(self.worker_task)["source"]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data))
        self.assertEqual(workers.validate(self.worker_task, "fetcher"), data)
        interface = workers.layout(self.worker_task)["interface"]
        interface.write_text(json.dumps({"compatibility": data["compatibility"], "evidence": data["evidence"]}))
        self.assertFalse(workers.validate(self.worker_task, "interface")["compatibility"]["supported"])

    def test_discovery_can_leave_capabilities_unknown_until_interface_research(self):
        compatibility = {"task_type": "time_series_anomaly_detection", "supported": True,
                         "dataset_types": [], "schemas": [], "reason": "Needs source inspection"}
        self.assertEqual(self.artifact("fetcher", compatibility)["compatibility"], compatibility)
        with self.assertRaises(HarnessError):
            self.artifact("interface", compatibility)

    def test_malformed_compatibility_is_rejected(self):
        valid = {"task_type": "time_series_anomaly_detection", "supported": True,
                 "dataset_types": ["MTS"], "schemas": ["mts"], "reason": "Official code"}
        invalid = [None, [], "supported"]
        for key, value in (("task_type", "classification"), ("supported", 1), ("reason", None),
                           ("dataset_types", "UTS"), ("dataset_types", ["singlechannel"]),
                           ("schemas", ["unknown"]), ("schemas", [{}])):
            invalid.append({**valid, key: value})
        invalid.extend({key: value for key, value in valid.items() if key != missing} for missing in valid)
        for role in ("fetcher", "interface"):
            for compatibility in invalid:
                with self.subTest(role=role, compatibility=compatibility), self.assertRaises(HarnessError):
                    self.artifact(role, compatibility)


if __name__ == "__main__":
    unittest.main()
