"""EasyTSAD adapter for the official HYDRA implementation.

HYDRA (TheDatumOrg/HYDRA at
``5f76abee37833eed0365c2da69f1afc2d22ef0ab``) is a training-free,
parameter-light, multi-level hierarchical anomaly detector for time series.
This adapter is intentionally thin: it converts the EasyTSAD ``TSData`` test
curve into the array the official driver expects, delegates all computation to
the untouched upstream implementation
(``upstream/HYDRA/HYDRA_loader.py`` -> ``models/HYDRA_approx.py``), and returns
the official timepoint score vector.

Nothing is trained (paper, "Algorithm 3: HYDRA; (training-free,
parameter-light)"), no labels are consumed, and no score rescaling is
performed here: the official ``MinMaxScaler`` in ``upstream/HYDRA/main.py:61``
is reporting-only, and the EasyTSAD evaluator already handles score shifting.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import numpy as np

from .. import BaseMethod
from ...DataFactory import TSData

# The official source tree (``upstream/HYDRA``) is not an installed
# distribution and its directory name (``HYDRA``) collides with this very
# package, so it is loaded under a synthetic top-level namespace package. Both
# the tree root and its ``models`` subdirectory are registered explicitly (the
# checkout ships no ``__init__.py`` files) so the upstream relative imports
# ``from .models.HYDRA_approx import ...`` and ``from .HNSW_index import ...``
# resolve without touching ``sys.path``.
_UPSTREAM_DIR = Path(__file__).resolve().parent / "upstream" / "HYDRA"
_UPSTREAM_PKG = "_easytsad_hydra_upstream"
_UPSTREAM_MODELS_PKG = f"{_UPSTREAM_PKG}.models"

# Effective defaults reached from the official entry point:
# upstream/HYDRA/main.py:58 ``run_HYDRA(data, 64)`` -> main.py:53
# ``loader.HYDRA(winsize, mode='approx')`` -> HYDRA_loader.py:5-15.
_DEFAULTS: Dict[str, Any] = {
    "win_size": 64,
    "stride": 1,
    "K": 2,
    "include_all_windows": True,
    "debug": False,
    "ann_backend": "auto",
    "hnsw_M": 16,
    "hnsw_ef_construction": 100,
    "hnsw_ef": 40,
    "ann_threads": 1,
    "mode": "approx",
    "random_state": 42,
}

# Keys accepted in config.toml that the upstream factory does not take. They are
# recorded for provenance only (``metric`` is fixed to Euclidean inside the
# ANN index; ``seed`` is the TSB-AD harness RNG seed; ``n_jobs`` is unused
# because HYDRA is single-process).
_NON_FACTORY_KEYS = ("metric", "seed", "n_jobs")


def _register_namespace_package(name: str, path: Path) -> ModuleType:
    """Put a plain (``__init__.py``-free) directory on ``sys.modules``."""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        if spec is None:  # pragma: no cover - defensive
            raise ImportError(f"Cannot create a module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
    module.__path__ = [str(path)]
    return module


def _load_upstream_factory():
    """Import ``upstream/HYDRA/HYDRA_loader.py`` and return its factory."""
    _register_namespace_package(_UPSTREAM_PKG, _UPSTREAM_DIR)
    _register_namespace_package(_UPSTREAM_MODELS_PKG, _UPSTREAM_DIR / "models")
    loader = importlib.import_module(f"{_UPSTREAM_PKG}.HYDRA_loader")
    return loader.HYDRA


class HYDRA(BaseMethod):
    """Expose the official HYDRA detector through the EasyTSAD method API."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.name = "HYDRA"
        self.params: Dict[str, Any] = dict(_DEFAULTS)
        if isinstance(params, dict):
            self.params.update(params)
        self._score: Optional[np.ndarray] = None

        mode = str(self.params["mode"]).lower()
        if mode not in ("approx", "exact"):
            raise ValueError(
                f"Unsupported HYDRA mode {self.params['mode']!r}; "
                "the paper's configuration uses 'approx' "
                "(paper Sec. 5.1.4 footnote 2)."
            )
        self.params["mode"] = mode

        win_size = int(self.params["win_size"])
        if win_size < 1:
            raise ValueError(f"win_size must be >= 1, got {win_size}")
        self.params["win_size"] = int(win_size)
        self.params["stride"] = int(self.params["stride"])
        self.params["K"] = int(self.params["K"])

        self._factory_kwargs = {
            key: value
            for key, value in self.params.items()
            if key not in _NON_FACTORY_KEYS
        }

    # ------------------------------------------------------------------ phases
    def train_valid_phase(self, tsTrain: TSData) -> None:
        """HYDRA is training-free: no parameter is fit on the training split."""
        # The official script fits and scores the same series and never touches
        # data_train (upstream/HYDRA/main.py:48-58), so this is a deliberate
        # no-op rather than a missing implementation.
        self._score = None

    def test_phase(self, tsData: TSData) -> None:
        """Score ``tsData.test`` exactly as the official entry point does."""
        data = np.asarray(tsData.test, dtype=np.float64)
        if data.ndim == 1:
            pass  # [T] -- promoted to [T, 1] inside the detector
        elif data.ndim == 2:
            pass  # [T, C] -- passed unchanged, channels z-scored internally
        else:
            raise ValueError(
                "HYDRA expects a 1-D [T] or 2-D [T, C] test curve, "
                f"got shape {data.shape}"
            )

        length = data.shape[0]
        win_size = self.params["win_size"]
        if length < win_size:
            # Fewer than one full window: the hierarchy is undefined, and
            # padding would fabricate a score for an observation HYDRA never
            # saw (upstream/HYDRA/models/HYDRA_approx.py:270-271).
            raise ValueError(
                f"Series is too short for HYDRA: length={length} < "
                f"win_size={win_size}. Provide a longer curve or a smaller "
                "win_size instead of padding."
            )

        factory = _load_upstream_factory()
        model = factory(**self._factory_kwargs)
        model.compress_and_score_multi(data)
        ens_score, _ = model.ensemble_maxpool_windows()

        score = np.asarray(ens_score, dtype=np.float64).ravel()
        # With stride == 1 reverse_windowing_min already returns exactly T
        # timepoints (upstream/HYDRA/models/HYDRA_approx.py:150-155). Keep the
        # contract explicit anyway: the evaluator right-aligns labels, so a
        # shorter vector must stay aligned to the end of the curve.
        if score.size > length:
            score = score[-length:]
        elif score.size < length:
            pad_value = float(score.min()) if score.size else 0.0
            score = np.concatenate([np.full(length - score.size, pad_value), score])

        self._score = np.ascontiguousarray(score, dtype=np.float64)

    def anomaly_score(self) -> np.ndarray:
        if self._score is None:
            raise RuntimeError("Call test_phase before requesting HYDRA scores.")
        return self._score

    # ------------------------------------------------------------- reporting
    def param_statistic(self, save_file) -> None:
        # Called before any curve is seen; HYDRA has no trainable parameters.
        with open(save_file, "w", encoding="utf-8") as handle:
            handle.write(
                "HYDRA is training-free (paper Algorithm 3); no trainable "
                "parameters.\n"
                f"win_size: {self.params['win_size']}\n"
                f"stride: {self.params['stride']}\n"
                f"K: {self.params['K']}\n"
                f"include_all_windows: {self.params['include_all_windows']}\n"
                f"mode: {self.params['mode']}\n"
                f"ann_backend: {self.params['ann_backend']}\n"
                f"random_state: {self.params['random_state']}\n"
            )
