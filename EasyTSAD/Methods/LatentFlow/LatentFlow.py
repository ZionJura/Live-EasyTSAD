"""EasyTSAD adapter for the official LatentFlow implementation.

LatentFlow (``SunboTax/LatentFlow`` at
``3695f54317fcc1a3fdbfa0a7037d1cdf4369db27``) is a reconstruction-based
multivariate time-series anomaly detector: overlapping patches are embedded by a
gated temporal convolution (``TemporalFeatureExtractor``), advanced by a
``GRUCell``, related by a learned sparse graph whose adjacency evolves with an
inertia-gated ``StructureEvolutionCell``, aggregated by ``GraphAggregator`` and
decoded back to the patch (``upstream/models/LatentFlow.py:115-176``).  The
anomaly score of a sliding window is the channel-mean squared reconstruction
error of its **last** timepoint (``upstream/exp/exp.py:217-224``); larger means
more anomalous.

The adapter re-implements the official train/test loop
(``upstream/exp/exp.py:146-231``) on top of the untouched upstream modules
(``upstream/models/LatentFlow.py``, ``upstream/utils/{tools,earlystop}.py``,
``upstream/data/dataset.py``).  ``upstream/exp/exp.py`` itself is deliberately
**not** imported: it calls ``torch.multiprocessing.set_start_method('spawn',
force=True)`` at import time (``upstream/exp/exp.py:20-21``), a process-wide side
effect that would break the calling harness.  ``aggregate_reconstruction`` is
therefore ported verbatim (``upstream/exp/exp.py:37-56``).  Likewise the SPOT
thresholding / metric computation in ``upstream/utils/vus_evaluate.py`` is not
ported: EasyTSAD owns thresholding and scoring.

Faithfulness notes
------------------
* No re-normalisation: EasyTSAD applies ``Data_Params.preprocess = "z-score"``
  (sklearn ``StandardScaler`` fitted on the train split), which is exactly what
  ``upstream/data/preprocess.py:17-20`` does.
* ``getStable`` detrending (``upstream/data/preprocess.py:6-9``) is skipped.  Its
  centred rolling median both crops ``period // 2`` points off each end and
  produces the ``*_stable`` tensors, which are pure passthroughs: ``Model.forward``
  never consumes them (``upstream/models/LatentFlow.py:140-176``) and
  ``_process_one_batch`` discards ``batch_stable``
  (``upstream/exp/exp.py:127-144``).  Skipping it is therefore a no-op
  algorithmically while avoiding the length-changing crop on short curves.
* ``train_rate`` split is reproduced here (``fit = train[:rate*T]``,
  ``valid = train[rate*T:]``), matching ``upstream/data/preprocess.py:22-32``.
  EasyTSAD's ``MTSData.buildfrom`` sets ``valid = train``, so the split is
  re-derived from ``tsTrain.train`` rather than taken from ``tsTrain.valid``.
* ``random_seed`` seeds one run.  The official ``itr`` simply repeats the same
  experiment for stability reporting (``upstream/main.py:83-84``); it is not part
  of the algorithm, so only ``random_seed`` is used.
* Score alignment: the official score vector has length ``T - window_size + 1``
  (one score per complete sliding window).  EasyTSAD expects one score per
  timepoint, so the leading ``window_size - 1`` positions -- for which no
  complete window exists -- are filled with the first available score.  This is
  recorded as unresolved in ``reproduction/adapter.yaml`` (the official code
  never defines these positions).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from .. import BaseMethod
from ...DataFactory import TSData

# --------------------------------------------------------------------------- #
# Upstream loading
#
# ``upstream/`` is not an installed distribution and its top-level directory
# names (``models``, ``utils``, ``data``, ``layers``) are too generic to claim in
# ``sys.modules``.  The tree is loaded under a synthetic namespace package.
# A single exception is needed: ``upstream/models/LatentFlow.py:5`` uses the
# absolute import ``from layers.RevIN import RevIN``, so a temporary ``layers``
# alias is installed *only* for the duration of that module's exec and any
# pre-existing ``layers`` module is restored afterwards.
# --------------------------------------------------------------------------- #
_UPSTREAM_DIR = Path(__file__).resolve().parent / "upstream"
_UPSTREAM_PKG = "_easytsad_latentflow_upstream"

# Effective defaults of the official entry point ``upstream/main.py`` (argparse
# defaults).  Mirrors ``config.toml``; per-dataset overrides from
# ``upstream/script/run.sh`` are deliberately not applied.
_DEFAULTS: Dict[str, Any] = {
    # data / window
    "train_rate": 0.8,
    "window_size": 64,
    # model structure
    "d_model": 64,
    "dropout": 0.6,
    "patch_size": 32,
    "patch_stride": 2,
    "k": 15,
    "lambda_sparsity": 1e-3,
    # optimisation
    "epochs": 30,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "lradj": "6",
    "optimizer": "Adam",
    # runtime
    "random_seed": 42,
}


def _register_namespace_package(name: str, path: Path) -> types.ModuleType:
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


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Execute ``path`` as the module ``name`` (cached in ``sys.modules``)."""
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_upstream_components() -> Dict[str, Any]:
    """Import the untouched upstream modules and return their symbols."""
    cached = getattr(_load_upstream_components, "_cache", None)
    if cached is not None:
        return cached

    _register_namespace_package(_UPSTREAM_PKG, _UPSTREAM_DIR)
    _register_namespace_package(f"{_UPSTREAM_PKG}.layers", _UPSTREAM_DIR / "layers")
    _register_namespace_package(f"{_UPSTREAM_PKG}.models", _UPSTREAM_DIR / "models")

    revin = _load_module_from_path(
        f"{_UPSTREAM_PKG}.layers.RevIN", _UPSTREAM_DIR / "layers" / "RevIN.py"
    )

    # ``upstream/models/LatentFlow.py:5`` -> ``from layers.RevIN import RevIN``.
    # Install the alias, exec the model module, then restore whatever was there.
    _missing = object()
    previous_layers = sys.modules.get("layers", _missing)
    previous_revin = sys.modules.get("layers.RevIN", _missing)
    layers_pkg = types.ModuleType("layers")
    layers_pkg.__path__ = [str(_UPSTREAM_DIR / "layers")]
    layers_pkg.RevIN = revin
    sys.modules["layers"] = layers_pkg
    sys.modules["layers.RevIN"] = revin
    try:
        model_module = _load_module_from_path(
            f"{_UPSTREAM_PKG}.models.LatentFlow",
            _UPSTREAM_DIR / "models" / "LatentFlow.py",
        )
    finally:
        for alias, previous in (
            ("layers", previous_layers),
            ("layers.RevIN", previous_revin),
        ):
            if previous is _missing:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous

    tools = _load_module_from_path(
        f"{_UPSTREAM_PKG}.utils.tools", _UPSTREAM_DIR / "utils" / "tools.py"
    )
    earlystop = _load_module_from_path(
        f"{_UPSTREAM_PKG}.utils.earlystop", _UPSTREAM_DIR / "utils" / "earlystop.py"
    )
    dataset = _load_module_from_path(
        f"{_UPSTREAM_PKG}.data.dataset", _UPSTREAM_DIR / "data" / "dataset.py"
    )

    components = {
        "Model": model_module.Model,
        "RG_EAD_Loss": tools.RG_EAD_Loss,
        "adjust_learning_rate": tools.adjust_learning_rate,
        "EarlyStop": earlystop.EarlyStop,
        "Dataset": dataset.Dataset,
    }
    _load_upstream_components._cache = components  # type: ignore[attr-defined]
    return components


def _aggregate_reconstruction(torch, recon_seq, original_len: int, stride: int = 1):
    """Verbatim port of ``upstream/exp/exp.py:37-56``.

    Overlap-averages the per-patch predictions back onto the window canvas.
    """
    B, S, N, P = recon_seq.shape
    device = recon_seq.device

    recon_canvas = torch.zeros((B, original_len, N), device=device)
    count_canvas = torch.zeros((B, original_len, N), device=device)

    for t in range(S):
        start_idx = t * stride
        end_idx = start_idx + P
        patch_pred = recon_seq[:, t, :, :].permute(0, 2, 1)
        recon_canvas[:, start_idx:end_idx, :] += patch_pred
        count_canvas[:, start_idx:end_idx, :] += 1.0

    return recon_canvas / (count_canvas + 1e-8)


def _set_seed(torch, seed: int) -> None:
    """Port of ``upstream/utils/seed.py:8-17``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _as_2d(array: Any, name: str) -> np.ndarray:
    """Coerce a curve to ``float32 [T, C]``; ``[T]`` becomes ``[T, 1]``."""
    data = np.asarray(array, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"{name} must be [T] or [T, C], got shape {data.shape}")
    if data.size == 0:
        raise ValueError(f"{name} is empty; LatentFlow needs at least one window")
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class LatentFlow(BaseMethod):
    """Expose the official LatentFlow detector through the EasyTSAD method API."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.name = "LatentFlow"
        self.params: Dict[str, Any] = dict(_DEFAULTS)
        if isinstance(params, dict):
            self.params.update(params)

        # Coerce to the types the upstream modules expect.
        for key in (
            "window_size",
            "d_model",
            "patch_size",
            "patch_stride",
            "k",
            "epochs",
            "patience",
            "batch_size",
            "random_seed",
        ):
            self.params[key] = int(self.params[key])
        for key in (
            "train_rate",
            "dropout",
            "lambda_sparsity",
            "lr",
            "weight_decay",
        ):
            self.params[key] = float(self.params[key])
        self.params["lradj"] = str(self.params["lradj"])

        window_size = self.params["window_size"]
        patch_size = self.params["patch_size"]
        patch_stride = self.params["patch_stride"]
        if window_size < 1 or patch_size < 1 or patch_stride < 1:
            raise ValueError(
                "window_size, patch_size and patch_stride must be >= 1, got "
                f"{window_size}, {patch_size}, {patch_stride}"
            )
        if patch_size > window_size:
            raise ValueError(
                f"patch_size={patch_size} must be <= window_size={window_size}"
            )

        self._model = None
        self._device = None
        self._num_channels: Optional[int] = None
        self._score: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ helpers
    def _resolve_device(self, torch):
        requested = str(self.params.get("device") or "auto").lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, torch, num_channels: int):
        """``upstream/exp/exp.py:108-115`` with an EasyTSAD ``SimpleNamespace``."""
        config = SimpleNamespace(
            enc_in=num_channels,
            window_size=self.params["window_size"],
            d_model=self.params["d_model"],
            patch_size=self.params["patch_size"],
            patch_stride=self.params["patch_stride"],
            k=self.params["k"],
            dropout=self.params["dropout"],
        )
        components = _load_upstream_components()
        return components["Model"](config).float().to(self._device)

    def _criterion(self):
        """``upstream/exp/exp.py:124-125``."""
        components = _load_upstream_components()
        return components["RG_EAD_Loss"](
            lambda_sparsity=self.params["lambda_sparsity"]
        )

    # ------------------------------------------------------------------ phases
    def train_valid_phase(self, tsTrain: TSData) -> None:
        """Train the official model on ``tsTrain.train`` with early stopping."""
        import torch
        from torch.utils.data import DataLoader

        data = _as_2d(tsTrain.train, "tsTrain.train")
        total, num_channels = data.shape
        window_size = self.params["window_size"]
        if total < window_size:
            raise ValueError(
                "LatentFlow needs at least window_size training points: "
                f"got {total} < window_size={window_size}"
            )
        if num_channels < 2:
            # SparseGraphLearner clamps k to N - 1, which is 0 for a single
            # channel (upstream/models/LatentFlow.py:66-72): LatentFlow is an
            # MTS-only model.
            raise ValueError(
                "LatentFlow is a multivariate detector; expected >= 2 channels, "
                f"got {num_channels}"
            )

        # upstream/data/preprocess.py:22-32 -- contiguous train_rate split of the
        # pre-split train array (EasyTSAD's .valid mirrors .train, so it is not
        # used for early stopping).
        split = int(total * self.params["train_rate"])
        split = min(max(split, window_size), total)
        fit, valid = data[:split], data[split:]
        if valid.shape[0] < window_size:
            # Too short to form a single validation window; fall back to the
            # fitting split so early stopping still has a well-defined loss.
            valid = fit
        self._num_channels = num_channels

        _set_seed(torch, self.params["random_seed"])
        self._device = self._resolve_device(torch)
        self._model = self._build_model(torch, num_channels)

        components = _load_upstream_components()
        Dataset = components["Dataset"]
        EarlyStop = components["EarlyStop"]
        adjust_learning_rate = components["adjust_learning_rate"]

        batch_size = self.params["batch_size"]
        train_loader = DataLoader(
            Dataset(fit, fit, fit, window_size),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            Dataset(valid, valid, valid, window_size),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = self._criterion()
        lr_args = SimpleNamespace(lradj=self.params["lradj"], lr=self.params["lr"])

        with tempfile.TemporaryDirectory(prefix="easytsad_latentflow_") as tmp:
            checkpoint_path = os.path.join(tmp, "latentflow.pth")
            early_stopping = EarlyStop(
                patience=self.params["patience"], path=checkpoint_path, verbose=False
            )

            for epoch in range(self.params["epochs"]):
                # ------------------------------------------------ train epoch
                self._model.train()
                train_losses = []
                for batch_data, _batch_stable, _batch_label in train_loader:
                    batch_data = batch_data.float().to(self._device)
                    optimizer.zero_grad()
                    recon, truth, adjs = self._model(
                        batch_data, inference_inertia=False
                    )
                    loss, _l_rec, _l_spar = criterion(recon, truth, adjs)
                    train_losses.append(float(loss.item()))
                    loss.backward()
                    optimizer.step()

                # --------------------------------------------- validation epoch
                with torch.no_grad():
                    self._model.eval()
                    valid_losses = []
                    for batch_data, _batch_stable, _batch_label in valid_loader:
                        batch_data = batch_data.float().to(self._device)
                        # upstream/exp/exp.py:170-173 -- validation reuses the
                        # training branch (no patch aggregation).
                        recon, truth, adjs = self._model(
                            batch_data, inference_inertia=False
                        )
                        loss, _l_rec, _l_spar = criterion(recon, truth, adjs)
                        valid_losses.append(float(loss.item()))

                # upstream/exp/exp.py:175-194
                early_stopping(float(np.average(valid_losses)), self._model)
                if early_stopping.early_stop:
                    break
                adjust_learning_rate(optimizer, epoch + 1, lr_args, printout=False)

            # upstream/exp/exp.py:196-197 -- restore the best checkpoint.
            self._model.load_state_dict(
                torch.load(checkpoint_path, map_location=self._device)
            )

        self._model.eval()

    def test_phase(self, tsData: TSData) -> None:
        """Score ``tsData.test`` exactly as the official entry point does."""
        import torch
        from torch.utils.data import DataLoader

        if self._model is None:
            raise RuntimeError(
                "Call train_valid_phase before test_phase: LatentFlow must be "
                "trained before it can score a curve."
            )

        data = _as_2d(tsData.test, "tsData.test")
        length = data.shape[0]
        window_size = self.params["window_size"]
        if length < window_size:
            raise ValueError(
                "LatentFlow needs at least window_size test points: "
                f"got {length} < window_size={window_size}"
            )

        components = _load_upstream_components()
        Dataset = components["Dataset"]
        loader = DataLoader(
            Dataset(data, data, data, window_size),
            batch_size=self.params["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        scores = []
        with torch.no_grad():
            self._model.eval()
            for batch_data, _batch_stable, _batch_label in loader:
                x = batch_data.float().to(self._device)
                recon, _truth, _adj = self._model(
                    x, inference_inertia=True, return_adj=False
                )
                recon = _aggregate_reconstruction(
                    torch,
                    recon,
                    original_len=window_size,
                    stride=self.params["patch_stride"],
                )
                # upstream/exp/exp.py:217-224
                batch_score = torch.mean(
                    (x[:, -1, :] - recon[:, -1, :]) ** 2, dim=-1, keepdim=True
                )
                scores.append(
                    batch_score.detach().cpu().numpy().astype(np.float32, copy=False)
                )

        score = np.concatenate(scores, axis=0).reshape(-1).astype(np.float64)

        # EasyTSAD wants one score per timepoint; the official vector covers only
        # the windows that fit (T - W + 1).  Keep right alignment and repeat the
        # first available score over the leading W - 1 positions (undefined
        # upstream -- recorded as unresolved in reproduction/adapter.yaml).
        if score.size > length:
            score = score[-length:]
        elif score.size < length:
            pad = length - score.size
            fill = float(score[0]) if score.size else 0.0
            score = np.concatenate([np.full(pad, fill, dtype=np.float64), score])

        self._score = np.ascontiguousarray(score, dtype=np.float64)

    def anomaly_score(self) -> np.ndarray:
        if self._score is None:
            raise RuntimeError("Call test_phase before requesting LatentFlow scores.")
        return self._score

    # --------------------------------------------------------------- reporting
    def param_statistic(self, save_file) -> None:
        p = self.params
        with open(save_file, "w", encoding="utf-8") as handle:
            handle.write(
                "LatentFlow (SunboTax/LatentFlow @ "
                "3695f54317fcc1a3fdbfa0a7037d1cdf4369db27)\n"
                "Reconstruction-based MTS detector: gated temporal conv patches -> "
                "GRUCell -> evolving sparse graph -> decoder; score = channel-mean "
                "squared error of the last window timepoint.\n"
                f"window_size: {p['window_size']}\n"
                f"patch_size: {p['patch_size']}\n"
                f"patch_stride: {p['patch_stride']}\n"
                f"d_model: {p['d_model']}\n"
                f"dropout: {p['dropout']}\n"
                f"k: {p['k']}\n"
                f"lambda_sparsity: {p['lambda_sparsity']}\n"
                f"train_rate: {p['train_rate']}\n"
                f"epochs: {p['epochs']}\n"
                f"patience: {p['patience']}\n"
                f"batch_size: {p['batch_size']}\n"
                f"lr: {p['lr']}\n"
                f"weight_decay: {p['weight_decay']}\n"
                f"lradj: {p['lradj']}\n"
                f"optimizer: {p['optimizer']}\n"
                f"random_seed: {p['random_seed']}\n"
                f"num_channels_seen: {self._num_channels}\n"
                "Score alignment: leading window_size - 1 positions repeat the "
                "first window score (undefined in the official code).\n"
            )
