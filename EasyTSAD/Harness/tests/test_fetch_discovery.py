"""FETCH must discover missing material before deterministic downloads."""
import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tsad_harness import papers, pipeline, workers
from tsad_harness.config import HarnessError


class FetchDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tsad-fetch-")
        self.root = Path(self.temp.name)
        self.method = self.root / "methods/Example"
        self.directory = self.root / "tasks/Example"
        self.directory.mkdir(parents=True)
        self.task = SimpleNamespace(method_dir=self.method, dir=self.directory,
            spec={"paper": {"title": "A paper without an initial PDF"}}, log=lambda *args, **kwargs: None)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_initial_pdf_does_not_prevent_fetch_worker(self):
        with patch.object(pipeline, "ROOT", self.root), patch.object(workers, "ROOT", self.root), \
             patch.object(papers, "fetch_to", side_effect=HarnessError("No PDF yet")):
            metadata = pipeline.fetch_paper(self.task)
        self.assertTrue(metadata["discovery_pending"])
        self.assertEqual(metadata["errors"], ["No PDF yet"])
        self.assertTrue((self.directory / "paper").is_symlink())
        self.assertFalse((self.directory / "paper").readlink().is_absolute())
        self.assertTrue((self.method / "paper/meta.json").is_file())

    def test_selected_paper_is_downloaded_by_script_and_paths_are_relative(self):
        (self.method / "upstream").mkdir(parents=True)
        (self.method / "paper").mkdir()
        record = self.method / "reproduction/source.json"
        record.parent.mkdir()
        record.write_text(json.dumps({"supported": True, "repository_url": "https://github.com/authors/paper",
            "commit": "verified", "paper": {"title": "Resolved paper", "pdf_url": "https://example.org/paper.pdf"}}))
        fetched = {"paper": {"title": "Resolved paper"},
                   "pdf": {"path": str(self.method / "paper/paper.pdf")},
                   "markdown": {"path": str(self.method / "paper/paper.md")}}
        with patch.object(pipeline, "ROOT", self.root), patch.object(workers, "ROOT", self.root), \
             patch.object(papers, "fetch_to", return_value=fetched) as download:
            result = pipeline.prepare_source(self.task)
        self.assertEqual(result["commit"], "verified")
        self.assertEqual(download.call_args.args[1]["pdf_url"], "https://example.org/paper.pdf")
        self.assertFalse(download.call_args.kwargs["clone_code"])
        metadata = json.loads((self.method / "paper/meta.json").read_text())
        self.assertEqual(metadata["pdf"]["path"], "methods/Example/paper/paper.pdf")

    def test_unresolved_title_is_left_for_fetch_agent_without_hidden_model_call(self):
        with patch.object(papers, "arxiv_search", return_value=[]), patch.object(papers, "llm_search") as model:
            record = papers.resolve(title="A paper title")
        model.assert_not_called()
        self.assertEqual(record["title"], "A paper title")
        self.assertEqual(record["source"], "manual")

    def test_blocked_paper_pdf_does_not_block_official_source(self):
        (self.method / "upstream").mkdir(parents=True)
        (self.method / "paper").mkdir()
        record = self.method / "reproduction/source.json"
        record.parent.mkdir()
        record.write_text(json.dumps({"supported": True, "repository_url": "https://github.com/authors/paper",
            "commit": "verified", "paper": {"title": "Resolved paper", "pdf_url": "https://example.org/paper.pdf"}}))
        with patch.object(pipeline, "ROOT", self.root), patch.object(workers, "ROOT", self.root), \
             patch.object(papers, "fetch_to", side_effect=HarnessError("HTTP 403")) as download:
            self.assertEqual(pipeline.prepare_source(self.task)["commit"], "verified")
            pipeline.prepare_source(self.task)
        self.assertEqual(download.call_count, 1)
        metadata = json.loads((self.method / "paper/meta.json").read_text())
        self.assertIn("HTTP 403", metadata["errors"])

    def test_metadata_service_outage_still_admits_arxiv_discovery(self):
        with patch.object(papers, "_arxiv_query", side_effect=HarnessError("metadata unavailable")):
            record = papers.resolve(arxiv="2401.12345")
        self.assertEqual(record["arxiv_id"], "2401.12345")
        self.assertEqual(record["pdf_url"], "https://arxiv.org/pdf/2401.12345")

    def test_proxy_rejection_uses_official_revision_archive(self):
        revision = "a" * 40
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w:gz") as archive:
            source = b"official implementation\n"
            member = tarfile.TarInfo(f"Example-{revision}/model.py")
            member.size = len(source)
            archive.addfile(member, io.BytesIO(source))
            link = tarfile.TarInfo(f"Example-{revision}/wandb")
            link.type = tarfile.SYMTYPE
            link.linkname = "/original-machine/wandb"
            archive.addfile(link)
        failed = SimpleNamespace(returncode=128, stderr="Received HTTP code 407 from proxy after CONNECT")
        config = SimpleNamespace(network_env=lambda: {})
        with patch.object(papers, "load", return_value=config), \
             patch.object(papers.subprocess, "run", return_value=failed), \
             patch.object(papers, "http", return_value=content.getvalue()) as download:
            result = papers.clone("https://github.com/authors/Example", self.root / "code", revision=revision)
        self.assertEqual((Path(result["dir"]) / "model.py").read_bytes(), source)
        self.assertEqual(str((Path(result["dir"]) / "wandb").readlink()), "/original-machine/wandb")
        self.assertEqual(result["commit"], revision)
        self.assertEqual(result["transport"], "github-archive")
        self.assertEqual(download.call_args.args[0], f"https://codeload.github.com/authors/Example/tar.gz/{revision}")
