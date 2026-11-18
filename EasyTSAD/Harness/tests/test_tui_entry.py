"""Run the actual Node TUI launch scripts against isolated build fixtures."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[3]


class TerminalEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsad node tui entry-")
        self.root = Path(self.tmp.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("harness", "prepare-harness-tui"):
            shutil.copyfile(REPOSITORY / "scripts" / name, scripts / name)
            (scripts / name).chmod(0o755)
        self.project = self.root / "EasyTSAD/Harness/tui"
        (self.project / "src").mkdir(parents=True)
        (self.project / "package.json").write_text('{"name":"fixture","scripts":{"build":"fixture"}}\n')
        self.lock = self.project / "package-lock.json"
        self.lock.write_text('{"lockfileVersion":3}\n')
        self.source = self.project / "src/main.mjs"
        self.source.write_text('// fixture TUI\n')
        (self.project / "build.mjs").write_text('// fixture build\n')
        self.cache = self.root / "private cache"
        self.trace = self.root / "preparation.log"
        tools = self.root / "bin"
        tools.mkdir()
        self.write_executable(tools / "npm", '''#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$TEST_PREPARATION_LOG"
[ "${TEST_NPM_FAIL:-}" != "$1" ] || exit 97
if [ "$1" = run ]; then
    mkdir -p dist
    cp src/main.mjs dist/tui.mjs
    printf 'asset\\n' > dist/resource.txt
fi
''')
        self.write_executable(scripts / "tsad", '#!/bin/sh\nprintf "sync\\n" >> "$TEST_PREPARATION_LOG"\nexit 98\n')
        self.host_node = self.root / "EasyTSAD/Harness/dsh-host/node_modules/node/bin/node"
        self.write_executable(self.host_node, '''#!/bin/sh
set -eu
if [ "${1:-}" = -p ]; then printf '%s\\n' "${TEST_NODE_VERSION:-24.9.0}"; exit; fi
printf 'node\\n'
printf 'arg:%s\\n' "$@"
printf 'root:%s\\n' "$TSAD_ROOT"
printf 'python:%s\\n' "$TSAD_HARNESS_PYTHON"
printf 'url:%s\\n' "${TSAD_HARNESS_URL:-}"
printf 'pythonpath:%s\\n' "${PYTHONPATH:-}"
''')
        self.full = self.root / "envs/harness/venv/bin/python"
        self.write_executable(self.full, '#!/bin/sh\nprintf "full\\n"\nprintf "arg:%s\\n" "$@"\n')
        self.env = {**os.environ, "TSAD_TUI_CACHE_DIR": str(self.cache),
                    "TEST_PREPARATION_LOG": str(self.trace), "PYTHONPATH": "fixture-existing-path",
                    "TSAD_HARNESS_URL": "http://127.0.0.1:19999",
                    "PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")}
        self.env.pop("TSAD_HARNESS_PYTHON", None)
        self.env.pop("TSAD_TUI_NODE", None)

    def tearDown(self):
        self.tmp.cleanup()

    def write_executable(self, path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        path.chmod(0o755)

    def run_script(self, script, *arguments):
        return subprocess.run(["bash", str(self.root / "scripts" / script), *arguments],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def lookup(self, mode="--print-entry"):
        return self.run_script("prepare-harness-tui", mode)

    def prepare(self):
        result = self.run_script("harness", "prepare-tui")
        self.assertEqual(result.returncode, 0, result.stderr)
        entry = self.lookup()
        self.assertEqual(entry.returncode, 0, entry.stderr)
        self.trace.unlink()
        return Path(entry.stdout.strip())

    def assert_node(self, result, entry, arguments):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["node", "arg:" + str(entry)] +
                         ["arg:" + arg for arg in arguments] + [
                             "root:" + str(self.root),
                             "python:" + self.env.get("TSAD_HARNESS_PYTHON", str(self.full)),
                             "url:" + self.env["TSAD_HARNESS_URL"],
                             "pythonpath:" + str(self.root / "EasyTSAD/Harness") + ":fixture-existing-path"])

    def test_prepared_bundle_uses_standalone_node_and_preserves_arguments_and_adapter_environment(self):
        entry = self.prepare()
        node = Path(self.lookup("--print-node").stdout.strip())
        self.assertNotEqual(node, self.host_node)
        self.assertEqual(node.read_bytes(), self.host_node.read_bytes())
        self.assertEqual((entry.parent / "resource.txt").read_text(), "asset\n")
        arguments = ["Some task", "--gpus", "0,1", "--refresh", "0.75"]
        self.assert_node(self.run_script("harness", "tui", *arguments), entry, arguments)
        self.assertFalse(self.trace.exists(), "Warm launch must not run npm or Python sync")

    def test_missing_cache_builds_once_then_launches_without_running_npm_again(self):
        first = self.run_script("harness", "tui", "--refresh", "1")
        entry = Path(self.lookup().stdout.strip())
        self.assert_node(first, entry, ["--refresh", "1"])
        self.assertEqual(self.trace.read_text().splitlines(), [
            "ci --cache " + str(self.cache / "npm-cache") + " --no-audit --no-fund", "run build"])
        self.trace.unlink()
        self.assert_node(self.run_script("harness", "tui"), entry, [])
        self.assertFalse(self.trace.exists())
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o700)
        self.assertFalse((entry.parent.parent / "source").exists())

    def test_lookup_does_not_prepare_or_create_an_absent_cache(self):
        for mode in ("--print-entry", "--print-node"):
            result = self.lookup(mode)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.trace.exists())

    def test_source_or_lock_changes_require_a_new_build(self):
        old = self.prepare()
        for file in (self.source, self.lock):
            file.write_text(file.read_text() + "\n")
            self.assertEqual(self.lookup().returncode, 1)
            result = self.run_script("harness", "tui")
            current = Path(self.lookup().stdout.strip())
            self.assertNotEqual(current, old)
            self.assert_node(result, current, [])
            self.assertTrue(old.exists(), "Older complete artifacts are retained")
            self.assertIn("run build", self.trace.read_text())
            self.trace.unlink()
            old = current

    def test_build_outputs_and_node_modules_do_not_invalidate_the_source_cache(self):
        entry = self.prepare()
        for directory in ("dist", "node_modules", "test"):
            path = self.project / directory / "ignored.txt"
            path.parent.mkdir()
            path.write_text("not runtime source\n")
        self.assertEqual(self.lookup().stdout.strip(), str(entry))
        self.assertFalse(self.trace.exists())

    def test_archived_upstream_source_is_not_hashed_but_license_and_provenance_are(self):
        vendor = self.project / "vendor"
        (vendor / "deepseek-harness/src").mkdir(parents=True)
        notice = vendor / "PROVENANCE.md"
        notice.write_text("fixed upstream commit\n")
        (vendor / "deepseek-harness/LICENSE").write_text("BSD-3-Clause\n")
        archived = vendor / "deepseek-harness/src/unused.ts"
        archived.write_text("// historical source\n")
        entry = self.prepare()
        archived.write_text("// still not compiled\n")
        self.assertEqual(self.lookup().stdout.strip(), str(entry))
        notice.write_text("updated provenance\n")
        self.assertEqual(self.lookup().returncode, 1)
        self.assertFalse(self.trace.exists())

    def test_vendor_license_changes_additions_and_removals_invalidate_cached_build(self):
        licenses = self.project / "vendor/licenses"
        (licenses / "nested").mkdir(parents=True)
        notice = licenses / "nested/dependency.txt"
        notice.write_text("MIT copyright and license\n")
        entry = self.prepare()
        notice.write_text("updated MIT copyright and license\n")
        self.assertEqual(self.lookup().returncode, 1)
        notice.write_text("MIT copyright and license\n")
        self.assertEqual(self.lookup().stdout.strip(), str(entry))
        added = licenses / "another.txt"
        added.write_text("ISC copyright and license\n")
        self.assertEqual(self.lookup().returncode, 1)
        added.unlink()
        self.assertEqual(self.lookup().stdout.strip(), str(entry))
        notice.unlink()
        self.assertEqual(self.lookup().returncode, 1)
        self.assertFalse(self.trace.exists())

    def test_failed_rebuild_keeps_old_complete_artifact_and_never_publishes_partial_ready(self):
        old = self.prepare()
        self.source.write_text("// changed\n")
        self.env["TEST_NPM_FAIL"] = "run"
        result = self.run_script("harness", "tui")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(old.exists())
        self.assertEqual(self.lookup().returncode, 1)
        self.assertEqual(list((self.cache / "builds").glob("*/.ready")), [old.parent.parent / ".ready"])
        self.assertEqual(list(self.cache.glob("build-*")), [])

    def test_symlink_and_public_cache_are_rejected_without_install_or_python_fallback(self):
        self.prepare()
        self.cache.chmod(0o755)
        self.assertNotEqual(self.run_script("harness", "tui").returncode, 0)
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o755)
        self.cache.chmod(0o700)
        destination = self.root / "cache destination"
        self.cache.rename(destination)
        self.cache.symlink_to(destination, target_is_directory=True)
        self.assertEqual(self.lookup().returncode, 1)
        result = self.run_script("harness", "tui")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("full\n", result.stdout)
        self.assertFalse(self.trace.exists())

    def test_non_tui_commands_keep_using_the_full_harness_environment(self):
        result = self.run_script("harness", "status", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["full", "arg:-m", "arg:tsad_harness", "arg:status", "arg:--help"])
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.trace.exists())

    def test_explicit_full_python_and_fixture_url_are_preserved(self):
        entry = self.prepare()
        self.env["TSAD_HARNESS_PYTHON"] = "/fixture/full python"
        self.assert_node(self.run_script("harness", "tui"), entry, [])

    def test_prepare_rejects_unsupported_node_before_npm(self):
        self.env["TEST_NODE_VERSION"] = "22.23.3"
        result = self.run_script("harness", "prepare-tui")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Node 24 is required", result.stderr)
        self.assertFalse(self.trace.exists())
        self.assertFalse((self.cache / "builds").exists())

    def test_python_cli_delegates_to_the_same_node_entry_with_all_options(self):
        from tsad_harness import cli
        args = SimpleNamespace(task="Some task", gpus="cpu", refresh=.75)
        with mock.patch.object(cli, "ROOT", self.root), mock.patch("os.execv") as execute:
            cli.cmd_tui(args)
        path = str(self.root / "scripts/harness")
        execute.assert_called_once_with(path, [path, "tui", "Some task", "--gpus", "cpu", "--refresh", "0.75"])


if __name__ == "__main__":
    unittest.main()
