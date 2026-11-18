"""EasyTSAD adapter for the official AutoTSAD implementation.

AutoTSAD is an unsupervised AutoML pipeline for univariate time series
anomaly detection.  This adapter is intentionally thin: it only converts the
EasyTSAD ``TSData`` structure into the official ``autotsad`` programmatic API
(``autotsad.system.main.autotsad``) and returns the official combined score
vector.  All preprocessing, candidate selection, ensembling and scoring are
performed by the untouched official code.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .. import BaseMethod
from ...DataFactory import TSData

_METHOD_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _METHOD_DIR / "autotsad-exp-config.yaml"

# Loaded lazily on the first test_phase call of a process.  AutoTSAD keeps a
# single global nx_config ``config`` singleton (and a single global timer
# object), so the official configuration file is filled only once per process.
_config_loaded: bool = False
_tmp_base: Optional[Path] = None
_result_base: Optional[Path] = None


class AutoTSAD(BaseMethod):
    """Adapter exposing the official AutoTSAD pipeline to EasyTSAD."""

    def __init__(self, params: dict) -> None:
        self.name = "AutoTSAD"
        self.params = params if isinstance(params, dict) else {}
        self.__anomaly_score: Optional[np.ndarray] = None

    def train_valid_phase(self, tsData: TSData) -> None:
        # AutoTSAD is fully unsupervised: the official recipe derives its
        # training/base time series from the test series itself inside
        # test_phase().  Nothing to do here.
        pass

    def test_phase(self, tsData: TSData) -> None:
        data = np.asarray(tsData.test, dtype=np.float64)
        if data.ndim != 1:
            data = data.reshape(-1)

        labels = np.asarray(tsData.test_label, dtype=np.bool_)
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        hexhash = hashlib.md5(data.tobytes()).hexdigest()

        config, autotsad, update_section = self._load_autotsad()
        self._ensure_java_on_path()
        self._ensure_runtime_config(config, update_section, hexhash)

        dataset_path = self._write_dataset_csv(data, labels, hexhash)

        name = self._safe_curve_name(tsData) or hexhash
        from autotsad.dataset import TestDataset

        testdataset = TestDataset(data=data, labels=labels, hexhash=hexhash, name=name)
        # `dataset_path` is only recorded in the official result directory and
        # never re-read; `testdataset` carries the actual data/labels.
        self.__anomaly_score = autotsad(
            dataset_path, testdataset=testdataset, use_gt_for_cleaning=False
        )

    def anomaly_score(self) -> np.ndarray:
        score = self.__anomaly_score
        if score is None:
            raise RuntimeError("anomaly_score() called before test_phase().")
        return score

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_java_on_path() -> None:
        """Put the bundled Temurin JDK on ``PATH`` for the official pipeline.

        The untouched official AutoTSAD implementation invokes the GrammarViz
        Java tool with ``subprocess.run(["java", "-jar", ...])``.  ``jdk4py``
        (a dependency of the ``autotsad`` environment) ships a complete JDK 17
        runtime, so we only expose its ``java`` binary to the subprocesses.
        """
        import jdk4py

        java_bin = str(Path(jdk4py.JAVA).resolve().parent)
        current = os.environ.get("PATH", "")
        if not current.startswith(java_bin + os.pathsep):
            os.environ["PATH"] = java_bin + os.pathsep + current

    @staticmethod
    def _load_autotsad():
        """Insert the official package root and import the lazy entry points.

        The wheel only ships the top-level ``EasyTSAD`` package, so the
        official ``autotsad`` source tree (which lives next to this file) has
        to be made importable explicitly.
        """
        if str(_METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(_METHOD_DIR))

        from autotsad.config import config
        from autotsad.system.main import autotsad
        from nx_config.test_utils import update_section

        return config, autotsad, update_section

    def _ensure_runtime_config(self, config, update_section, hexhash: str) -> None:
        global _config_loaded, _tmp_base, _result_base

        if not _config_loaded:
            from nx_config import fill_config_from_path

            fill_config_from_path(config, path=_CONFIG_PATH, env_prefix="AUTOTSAD")

            _tmp_base = Path(tempfile.mkdtemp(prefix="autotsad-tmp-"))
            _result_base = Path(tempfile.mkdtemp(prefix="autotsad-result-"))
            update_section(config.general, tmp_path=_tmp_base)
            update_section(config.general, result_path=_result_base)
            update_section(config.general, seed=int(self.params.get("seed", 1)))
            update_section(config.general, n_jobs=int(self.params.get("n_jobs", 1)))
            update_section(config.general, progress=False)
            _config_loaded = True

        update_section(config.general, cache_key=hexhash)

    def _write_dataset_csv(self, data: np.ndarray, labels: np.ndarray, hexhash: str) -> Path:
        import pandas as pd

        filename = _result_base / f"{hexhash}.csv"
        df = pd.DataFrame(
            {
                "timestamp": np.arange(len(data)),
                "data": data,
                "is_anomaly": labels,
            }
        )
        df.to_csv(filename, index=False)
        return filename

    @staticmethod
    def _safe_curve_name(tsData: TSData) -> str:
        info = getattr(tsData, "info", None)
        if isinstance(info, dict):
            for key in ("file_name", "name", "dataset", "curve"):
                value = info.get(key)
                if value is not None:
                    return str(value)
        return ""