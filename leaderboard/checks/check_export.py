import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("export_results", Path(__file__).parents[1] / "scripts" / "export_results.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_result(self, method="TranAD", schema="mts", dataset="SMD", **values):
        path = self.root / method / schema / dataset / "avg.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values), encoding="utf-8")

    def test_actual_protocol_mapping_and_nonfinite_values(self):
        self.write_result(**{
            "event-based f1 under pa with mode log": {"f1": 0.8, "precision": 0.9},
            "best f1 under pa": {"f1": 0.9},
            "point-based auprc": float("nan"),
            "point-based auprc pa": 0.99,
        })
        data = exporter.export_results(self.root)
        self.assertEqual(data["results"][0]["metrics"], {"event_f1": 0.8, "point_f1": 0.9, "auprc": None})
        self.assertFalse(data["demo"])
        self.assertEqual(data["datasets"][0]["task"], "mts")
        json.dumps(data, allow_nan=False)

    def test_legacy_schema_normalization(self):
        self.write_result(method="AE", schema="one_by_one", dataset="Yahoo", **{"point-based auprc": 0.6})
        data = exporter.export_results(self.root)
        self.assertEqual(data["results"][0]["schema"], "naive")
        self.assertEqual(data["datasets"][0]["task"], "uts")

    def test_empty_invalid_scores_and_duplicate_normalized_records(self):
        with self.assertRaises(ValueError):
            exporter.export_results(self.root)
        self.write_result(**{"point-based auprc": 1.1})
        with self.assertRaises(ValueError):
            exporter.export_results(self.root)
        self.write_result(**{"point-based auprc": 0.5})
        self.write_result(schema="naive", dataset="Yahoo", **{"point-based auprc": 0.5})
        self.write_result(schema="one_by_one", dataset="Yahoo", **{"point-based auprc": 0.5})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            exporter.export_results(self.root)


if __name__ == "__main__":
    unittest.main()
