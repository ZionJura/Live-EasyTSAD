"""EasyTSAD adapter for CANDI.

CANDI (AAAI 2026, "CANDI: Curated Test-Time Adaptation for Multivariate
Time-Series Anomaly Detection Under Distribution Shift") = frozen MLP
autoencoder detector + plug-and-play SANA residual module (``TCN_iTrans``)
that is adapted at test time with SGD.  Sample curation is driven by frequency
percentile mining (FPM): an ``alpha``-percentile validation threshold plus a
Mahalanobis / chi-square reference set.

The official implementation is vendored verbatim at
``EasyTSAD/Methods/CANDI/upstream`` (commit 28c9679e503832f59e351208cde63657fcb51cad).
This module only provides the EasyTSAD plumbing:

  * preprocessing: ``StandardScaler`` fit on the 80% train split only, then
    applied to the 20% validation split and to the test split (upstream
    ``datasets/build.py::_normalize_data`` + ``_split_train_val``);
  * fixed-length sliding windows of ``window_size`` with ``train_stride`` /
    ``test_stride`` (upstream ``datasets/build.py::__len__`` / ``__getitem__``);
  * MLP autoencoder pretraining with the official solver defaults
    (Adam, lr 1e-4, weight decay 1e-4, cosine schedule with warmup, gradient
    clipping) on reconstruction MSE;
  * the CANDI test-time adaptation loop (upstream
    ``predictor.py::Predictor.predict`` + ``tta/candi/adapter_candi.py``);
  * score alignment to EasyTSAD's window-granularity contract.

Score alignment
---------------
Upstream scores one window per position: ``score[i]`` is the mean
reconstruction MSE of the window ``inputs[i * test_stride : i * test_stride +
window_size]``, i.e. it summarises the raw time steps ending at
``i * test_stride + window_size - 1``.  The number of windows is
``(len(test) - window_size) // test_stride + 1`` and the first
``window_size - 1`` raw positions are therefore never covered.  EasyTSAD
truncates the raw label vector to its last ``len(scores)`` entries, which for
the official ``test_stride = 1`` gives ``labels[window_size - 1:]`` -- an exact
match.  For ``test_stride > 1`` the window-ordered scores are still returned
unchanged (the strided windows simply leave gaps); no padding or shifting is
applied here.

Deliberate deviations from the upstream entrypoint (all side effects only)
-----------------------------------------------------------------------
``main.py`` / ``Predictor`` write ``config.txt``, ``best_result.txt``,
``test.txt``, ``*.npy`` and wandb logs under ``cfg.RESULT_DIR``, and
``datasets/build.py`` caches ``downsampled_*.npy`` under ``cfg.DATA.BASE_DIR``.
Those artefacts are dropped.  Upstream also hard-codes ``.cuda()``; the device
is resolved here (CUDA when available, CPU otherwise) without changing the
numerics.  DataLoader ``num_workers`` is forced to ``0`` so that no worker
processes are spawned from the evaluation harness.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from EasyTSAD.Methods import BaseMethod

# The vendored upstream tree uses absolute imports (``from layers... import``,
# ``from models.optimizer import``, ``from tta.adapter import``), so the tree
# root has to be importable.  ``config`` is loaded privately below to avoid
# shadowing any other top-level ``config`` module.
_UPSTREAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
if _UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, _UPSTREAM_DIR)


def _load_upstream_config():
    path = os.path.join(_UPSTREAM_DIR, "config.py")
    spec = importlib.util.spec_from_file_location("candi_upstream_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfg_module = _load_upstream_config()

# Imported lazily-ish, after sys.path is set up.
from models.mlp.modeling_mlp import MLP  # noqa: E402
from models.optimizer import construct_optimizer  # noqa: E402
from tta.candi.adapter_candi import SANA  # noqa: E402
from utils import lr_policy  # noqa: E402

__all__ = ["CANDI", "SANA"]


# --------------------------------------------------------------------------- #
# model_params (config.toml) -> yacs CfgNode
# --------------------------------------------------------------------------- #
# config.toml stays flat/readable; every key is mapped onto the official
# configuration path it overrides.
_PARAM_PATHS = {
    # data
    "n_vars": ("DATA", "N_VAR"),
    "window_size": ("DATA", "WIN_SIZE"),
    "train_stride": ("DATA", "TRAIN_STEP"),
    "test_stride": ("DATA", "TEST_STEP"),
    "train_ratio": ("DATA", "TRAIN_RATIO"),
    "downsample_rate": ("DATA", "DOWNSAMPLE_RATE"),
    "sampling_rates": ("DATA", "SAMPLING_RATES"),
    "dataset": ("DATA", "NAME"),
    # model
    "model_name": ("MODEL", "NAME"),
    "z_size": ("MLP", "Z_SIZE"),
    # SANA
    "sana_type": ("SANA", "TYPE"),
    "sana_d_model": ("SANA", "d_model"),
    "sana_n_heads": ("SANA", "n_heads"),
    "sana_e_layers": ("SANA", "e_layers"),
    "sana_d_ff": ("SANA", "d_ff"),
    "sana_dropout": ("SANA", "dropout"),
    "sana_activation": ("SANA", "activation"),
    "sana_output_attention": ("SANA", "output_attention"),
    # pretraining solver
    "epochs": ("SOLVER", "MAX_EPOCH"),
    "start_epoch": ("SOLVER", "START_EPOCH"),
    "optimizer": ("SOLVER", "OPTIMIZING_METHOD"),
    "lr": ("SOLVER", "BASE_LR"),
    "weight_decay": ("SOLVER", "WEIGHT_DECAY"),
    "lr_policy": ("SOLVER", "LR_POLICY"),
    "cosine_end_lr": ("SOLVER", "COSINE_END_LR"),
    "cosine_after_warmup": ("SOLVER", "COSINE_AFTER_WARMUP"),
    "warmup_epochs": ("SOLVER", "WARMUP_EPOCHS"),
    "warmup_start_lr": ("SOLVER", "WARMUP_START_LR"),
    "gradient_clip": ("SOLVER", "GRADIENT_CLIP"),
    "gradient_clip_norm": ("SOLVER", "GRADIENT_CLIP_NORM"),
    "batch_size": ("TRAIN", "BATCH_SIZE"),
    "val_batch_size": ("VAL", "BATCH_SIZE"),
    "test_batch_size": ("TEST", "BATCH_SIZE"),
    # test-time adaptation
    "tta_enable": ("TEST", "TTA", "ENABLE"),
    "tta_method": ("TEST", "TTA", "METHOD"),
    "tta_steps": ("TEST", "TTA", "STEPS"),
    "tta_lr": ("TEST", "TTA", "SOLVER", "BASE_LR"),
    "tta_momentum": ("TEST", "TTA", "SOLVER", "MOMENTUM"),
    "tta_nesterov": ("TEST", "TTA", "SOLVER", "NESTEROV"),
    "tta_weight_decay": ("TEST", "TTA", "SOLVER", "WEIGHT_DECAY"),
    "tta_dampening": ("TEST", "TTA", "SOLVER", "DAMPENING"),
    "tta_gradient_clip": ("TEST", "TTA", "SOLVER", "GRADIENT_CLIP"),
    "tta_gradient_clip_norm": ("TEST", "TTA", "SOLVER", "GRADIENT_CLIP_NORM"),
    "candi_use_fpm": ("TEST", "TTA", "CANDI", "USE_FPM"),
    "candi_use_sana": ("TEST", "TTA", "CANDI", "USE_SANA"),
    "candi_use_hard": ("TEST", "TTA", "CANDI", "USE_HARD"),
    "candi_use_moderate": ("TEST", "TTA", "CANDI", "USE_MODERATE"),
    "candi_gating_init": ("TEST", "TTA", "CANDI", "GATING_INIT"),
    "candi_min_samples": ("TEST", "TTA", "CANDI", "MIN_SAMPLES"),
    "m2n2_gamma": ("TEST", "TTA", "M2N2", "GAMMA"),
    # thresholding (the easytsad evaluator owns the decision threshold; CANDI
    # only reuses `anomaly_ratio` as the FPM curation ratio)
    "threshold_type": ("TEST", "THRESHOLD", "TYPE"),
    "anomaly_ratio": ("TEST", "THRESHOLD", "ANOMALY_RATIO"),
}

# Keys that are not part of the upstream CfgNode.
_IGNORED_KEYS = {
    "score_direction",
    "num_gpus",
    "visible_devices",
    "wandb_enable",
    "num_workers",
    "pin_memory",
    "eval_period",
    "seed",
    "fpm_chi2_quantile",
    "sana_tcn_num_layers",
    "sana_tcn_kernel_size",
    "sana_gating_activation",
}


def _build_cfg(model_params):
    """Map the flat ``model_params`` dict onto the official CfgNode."""
    cfg = _cfg_module.get_cfg_defaults()
    for key, value in (model_params or {}).items():
        if key in _IGNORED_KEYS:
            continue
        path = _PARAM_PATHS.get(key)
        if path is None:
            # Unknown keys stay inert rather than corrupting the config.
            continue
        node = cfg
        for name in path[:-1]:
            node = getattr(node, name)
        setattr(node, path[-1], value)

    # Device/batching knobs the harness must be able to control.
    cfg.SEED = int(model_params.get("seed", cfg.SEED))
    cfg.DATA_LOADER.NUM_WORKERS = 0  # no worker processes from the harness
    cfg.DATA_LOADER.PIN_MEMORY = bool(model_params.get("pin_memory", False))
    cfg.TRAIN.EVAL_PERIOD = int(model_params.get("eval_period", cfg.TRAIN.EVAL_PERIOD))
    cfg.TRAIN.SHUFFLE = True
    cfg.TRAIN.DROP_LAST = True
    cfg.VAL.SHUFFLE = False
    cfg.VAL.DROP_LAST = False
    cfg.TEST.SHUFFLE = False
    cfg.TEST.DROP_LAST = False

    # CANDI always curates with a ratio threshold (upstream predictor.py).
    cfg.TEST.THRESHOLD.TYPE = "ratio"
    assert cfg.MODEL.NAME == "MLP", "CANDI adapter only wires the official MLP backbone"
    return cfg


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_2d(array) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return data


def _as_1d(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------- #
# windowing -- mirrors upstream datasets/build.py::TSDataset
# --------------------------------------------------------------------------- #
class _WindowSet(Dataset):
    def __init__(self, data, labels, win_size, step):
        self.data = data
        self.labels = labels
        self.win_size = int(win_size)
        self.step = int(step)

    def __len__(self):
        return max((self.data.shape[0] - self.win_size) // self.step + 1, 0)

    def __getitem__(self, index):
        start = index * self.step
        inputs = self.data[start : start + self.win_size]
        if self.labels is None:
            return np.float32(inputs), 0
        labels = self.labels[start : start + self.win_size]
        return np.float32(inputs), int(np.any(labels == 1))


class _Thresholder:
    """Upstream ``threshold.py::Thresholder`` restricted to ``ratio``.

    ``ratio`` reads ``cfg.TEST.THRESHOLD.ANOMALY_RATIO`` and returns
    ``np.percentile(val_scores, 100 - ratio)``.
    """

    def __init__(self, cfg, val_scores):
        self.cfg = cfg
        self.threshold_type = cfg.TEST.THRESHOLD.TYPE
        assert self.threshold_type in ("ratio", "best_f1")
        self.threshold = None
        if self.threshold_type == "ratio":
            self.ratio = cfg.TEST.THRESHOLD.ANOMALY_RATIO
            self.threshold = float(np.percentile(val_scores, 100 - self.ratio))

    def set_threshold(self, threshold):
        self.threshold = threshold


# --------------------------------------------------------------------------- #
# CANDI test-time adapter -- upstream tta/candi/adapter_candi.py::CANDIAdapter
# --------------------------------------------------------------------------- #
class _CANDIAdapter:
    def __init__(self, cfg, model, thresholder, device, **kwargs):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.SOLVER = self.cfg.TEST.TTA.SOLVER
        self.model = model
        self.thresholder = thresholder
        self.device = device
        self.iter = 0
        self.n_adapt = 0

        if self.cfg.TEST.TTA.CANDI.USE_SANA:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.sana_in = SANA(cfg).to(device)
            for param in self.model.sana_in.parameters():
                param.requires_grad = True
            self.model.sana_out = SANA(cfg).to(device)
            for param in self.model.sana_out.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True
        self.optimizer = construct_optimizer(self.model, self.cfg)

        self.val_scores = _as_1d(kwargs["val_scores"])
        self.val_inputs = kwargs["val_inputs"]
        self.test_labels = _as_1d(kwargs["test_labels"])

        self.val_representations = self.model.get_representations(self.val_inputs)
        self.val_representations_mean = torch.mean(self.val_representations, dim=0)
        self.val_representations_cov = torch.cov(self.val_representations.T)
        self.representations_cov_inv = torch.linalg.pinv(self.val_representations_cov)

        # Top-k (hard) selection
        topk = int(len(self.val_scores) * self.cfg.TEST.THRESHOLD.ANOMALY_RATIO / 100)
        self.topk = max(min(topk, len(self.val_scores)), 1)
        topk_indices = np.argpartition(self.val_scores, -self.topk)[-self.topk :]
        self.topk_scores = self.val_scores[topk_indices]
        self.topk_inputs = self.val_inputs[topk_indices]
        self.topk_representations = self.val_representations[topk_indices]

        # Q1 ~ Q3 (moderate) selection
        self.q1 = np.percentile(self.val_scores, 25)
        self.q3 = np.percentile(self.val_scores, 75)
        moderate_mask = (self.val_scores > self.q1) & (self.val_scores < self.q3)
        moderate_indices = np.where(moderate_mask)[0]
        self.moderate_scores = self.val_scores[moderate_indices]
        self.moderate_inputs = self.val_inputs[moderate_indices]
        self.moderate_representations = self.val_representations[moderate_indices]

        self.samples_to_adapt_hard = []
        self.n_samples_to_adapt_hard = 0
        self.total_samples_to_adapt_hard = 0
        self.total_anomalies_in_hard = 0

        self.samples_to_adapt_moderate = []
        self.n_samples_to_adapt_moderate = 0
        self.total_samples_to_adapt_moderate = 0
        self.total_anomalies_in_moderate = 0

        self.chi2_quantile = float(
            getattr(self.cfg.TEST.TTA, "FPM_CHI2_QUANTILE", 0.05)
        )

    # -- helpers ---------------------------------------------------------- #
    def _labels_for(self, mask, scores):
        offset = self.iter * len(scores)
        indices = mask.nonzero(as_tuple=True)[0] + offset
        if len(self.test_labels) == 0:
            return np.zeros(len(mask), dtype=np.int64)
        indices = indices.clamp(max=len(self.test_labels) - 1)
        return self.test_labels[indices.cpu().numpy()]

    def _mahalanobis_mask(self, representations, reference):
        dist = torch.sum(
            (representations.unsqueeze(1) - reference.unsqueeze(0))
            @ self.representations_cov_inv
            * (representations.unsqueeze(1) - reference.unsqueeze(0)),
            dim=-1,
        )
        chi2_threshold = chi2.ppf(self.chi2_quantile, df=representations.shape[1])
        return (dist < chi2_threshold).any(dim=1)

    # -- curation --------------------------------------------------------- #
    @torch.no_grad()
    def get_samples_to_adapt(self, x, scores):
        if not self.cfg.TEST.TTA.CANDI.USE_FPM:
            # Handle as moderate for implementation convenience (upstream).
            mask_moderate = scores < self.thresholder.threshold
            selected_x_moderate = x[mask_moderate]
            if len(selected_x_moderate) > 0:
                labels = self._labels_for(mask_moderate, scores)
                self.n_samples_to_adapt_moderate += len(selected_x_moderate)
                self.samples_to_adapt_moderate.append(selected_x_moderate)
                self.total_samples_to_adapt_moderate += len(selected_x_moderate)
                self.total_anomalies_in_moderate += int((labels == 1).sum())
            return

        representations = self.model.get_representations(x)

        if self.cfg.TEST.TTA.CANDI.USE_HARD:
            mask_hard_sim = self._mahalanobis_mask(representations, self.topk_representations)
            # Also consider the anomaly score threshold.
            mask_hard = mask_hard_sim & (scores > self.thresholder.threshold)
            selected_x_hard = x[mask_hard]
            if len(selected_x_hard) > 0:
                labels = self._labels_for(mask_hard, scores)
                self.n_samples_to_adapt_hard += len(selected_x_hard)
                self.samples_to_adapt_hard.append(selected_x_hard)
                self.total_samples_to_adapt_hard += len(selected_x_hard)
                self.total_anomalies_in_hard += int((labels == 1).sum())

        if self.cfg.TEST.TTA.CANDI.USE_MODERATE:
            # Upstream computes the Q1/Q3 band and then overwrites it with the
            # score-threshold mask (adapter_candi.py:215-216); the effective
            # behaviour is reproduced here.
            mask_moderate_score = scores < self.thresholder.threshold
            mask_moderate_sim = self._mahalanobis_mask(
                representations, self.moderate_representations
            )
            mask_moderate = mask_moderate_score & mask_moderate_sim
            selected_x_moderate = x[mask_moderate]
            if len(selected_x_moderate) > 0:
                labels = self._labels_for(mask_moderate, scores)
                self.n_samples_to_adapt_moderate += len(selected_x_moderate)
                self.samples_to_adapt_moderate.append(selected_x_moderate)
                self.total_samples_to_adapt_moderate += len(selected_x_moderate)
                self.total_anomalies_in_moderate += int((labels == 1).sum())

        self.iter += 1

    # -- adaptation ------------------------------------------------------- #
    def _adapt_buffers(self, samples):
        is_eval = not copy.deepcopy(self.model.training)
        self.model.train()
        for _ in range(self.cfg.TEST.TTA.STEPS):
            loss = self.calculate_loss(samples)
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.TEST.TTA.SOLVER.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.TEST.TTA.SOLVER.GRADIENT_CLIP_NORM,
                )
            self.optimizer.step()
        if is_eval:
            self.model.eval()

    @torch.enable_grad()
    def adapt(self, x, scores):
        self.get_samples_to_adapt(x, scores)

        if self.n_samples_to_adapt_hard >= self.cfg.TEST.TTA.CANDI.MIN_SAMPLES:
            self.n_adapt += 1
            samples = torch.concat(self.samples_to_adapt_hard, dim=0)
            self._adapt_buffers(samples)
            self.samples_to_adapt_hard = []
            self.n_samples_to_adapt_hard = 0

        if self.n_samples_to_adapt_moderate >= self.cfg.TEST.TTA.CANDI.MIN_SAMPLES:
            self.n_adapt += 1
            samples = torch.concat(self.samples_to_adapt_moderate, dim=0)
            self._adapt_buffers(samples)
            self.samples_to_adapt_moderate = []
            self.n_samples_to_adapt_moderate = 0

    def calculate_loss(self, x):
        raise NotImplementedError(
            "The calculate_loss method should be implemented in subclasses."
        )


class _MLPAdapter(_CANDIAdapter):
    """Upstream ``MLPAdapter`` (equivalently ``tta/adapter.py::construct_adapter``)."""

    @torch.enable_grad()
    def calculate_loss(self, x):
        if hasattr(self.model, "sana_in"):
            x_new = x + self.model.sana_in(x)
            recon = self.model(x_new)
        else:
            recon = self.model(x)
        if hasattr(self.model, "sana_out"):
            recon = recon - self.model.sana_out(recon)
        return F.mse_loss(recon, x)


# --------------------------------------------------------------------------- #
# EasyTSAD method
# --------------------------------------------------------------------------- #
class CANDI(BaseMethod):
    """CANDI: Curated test-time adaptation for TSAD (EasyTSAD adapter)."""

    def __init__(self, model_params: dict = None, **kwargs):
        self.model_params = dict(model_params or {})
        self.cfg = _build_cfg(self.model_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._score = None
        self._curve = None
        self._train_input = None

    # -- data ------------------------------------------------------------- #
    def _prepare(self, curve):
        """Split / normalise / window exactly like upstream ``TSDataset``."""
        cfg = self.cfg
        train_raw = _as_2d(getattr(curve, "train"))
        test_raw = _as_2d(getattr(curve, "test"))
        train_label = getattr(curve, "train_label", None)
        test_label = getattr(curve, "test_label", None)

        # upstream datasets/build.py::_split_train_val
        train_len = int(len(train_raw) * cfg.DATA.TRAIN_RATIO)
        val_raw = train_raw[train_len:].copy()
        train_raw = train_raw[:train_len]
        if train_label is not None:
            train_label = _as_1d(train_label)
            val_label = train_label[train_len:].copy()
            train_label = train_label[:train_len]
        else:
            val_label = None

        # upstream datasets/build.py::_normalize_data ("standard")
        scaler = StandardScaler().fit(train_raw)
        train = scaler.transform(train_raw)
        val = scaler.transform(val_raw) if len(val_raw) else val_raw
        test = scaler.transform(test_raw)

        win = int(cfg.DATA.WIN_SIZE)
        self._train_set = _WindowSet(train, train_label, win, cfg.DATA.TRAIN_STEP)
        self._val_set = _WindowSet(val, val_label, win, cfg.DATA.TRAIN_STEP)
        self._test_set = _WindowSet(
            test, _as_1d(test_label) if test_label is not None else None, win, cfg.DATA.TEST_STEP
        )
        self._test_labels = (
            _as_1d(test_label) if test_label is not None else np.zeros(len(test))
        )
        if len(self._train_set) == 0 or len(self._test_set) == 0:
            raise ValueError(
                "CANDI needs at least window_size=%d time steps per split "
                "(train=%d, test=%d)." % (win, len(train), len(test))
            )

    # -- pretraining (upstream models/mlp/trainer_mlp.py::train_step) ------ #
    def _pretrain(self):
        cfg = self.cfg
        model = self.model
        loader = DataLoader(
            self._train_set,
            batch_size=int(cfg.TRAIN.BATCH_SIZE),
            shuffle=bool(cfg.TRAIN.SHUFFLE),
            drop_last=bool(cfg.TRAIN.DROP_LAST) and len(self._train_set) > cfg.TRAIN.BATCH_SIZE,
            num_workers=int(cfg.DATA_LOADER.NUM_WORKERS),
            pin_memory=bool(cfg.DATA_LOADER.PIN_MEMORY) and self.device.type == "cuda",
        )
        optimizer = construct_optimizer(model, cfg)
        data_size = max(len(loader), 1)
        model.train()
        for cur_epoch in range(int(cfg.SOLVER.START_EPOCH), int(cfg.SOLVER.MAX_EPOCH)):
            for cur_iter, (inputs, _) in enumerate(loader):
                lr = lr_policy.get_lr_at_epoch(
                    cfg, cur_epoch + float(cur_iter) / data_size
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                inputs = inputs.float().to(self.device)
                recon = model(inputs)
                loss = F.mse_loss(recon, inputs)
                optimizer.zero_grad()
                loss.backward()
                if cfg.SOLVER.GRADIENT_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.SOLVER.GRADIENT_CLIP_NORM,
                    )
                optimizer.step()
        model.eval()

    # -- scoring helpers --------------------------------------------------- #
    @torch.no_grad()
    def _scores_over(self, window_set, batch_size, collect_inputs=False):
        loader = DataLoader(
            window_set,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.cfg.DATA_LOADER.NUM_WORKERS),
            pin_memory=bool(self.cfg.DATA_LOADER.PIN_MEMORY)
            and self.device.type == "cuda",
        )
        scores_all, inputs_all = [], []
        for inputs, _ in loader:
            inputs = inputs.float().to(self.device)
            scores = self.model.get_anomaly_scores(inputs)
            scores_all.append(torch.flatten(torch.as_tensor(scores)))
            if collect_inputs:
                inputs_all.append(inputs)
        score = (
            torch.cat(scores_all, dim=0).cpu().numpy()
            if scores_all
            else np.zeros(0, dtype=np.float64)
        )
        if collect_inputs:
            tensor = (
                torch.cat(inputs_all, dim=0)
                if inputs_all
                else torch.zeros(0, self.cfg.DATA.WIN_SIZE, self.cfg.DATA.N_VAR)
            )
            return score, tensor
        return score, None

    # -- EasyTSAD hooks ---------------------------------------------------- #
    def train_valid_phase(self, tsTrain):
        """Fit preprocessing + pretrain the frozen MLP detector.

        The validation split is carved out of ``tsTrain.train`` (last
        ``1 - train_ratio``), matching upstream ``_split_train_val``.
        """
        _set_seeds(int(self.cfg.SEED))
        self._train_input = tsTrain
        self._prepare(tsTrain)
        self.model = MLP(self.cfg).to(self.device)
        self._pretrain()

    def train_valid_phase_all_in_one(self, tsTrains):  # pragma: no cover
        raise NotImplementedError(
            "CANDI is a single-dataset method; the all_in_one schema is not supported."
        )

    def test_phase(self, tsData):
        """Run CANDI test-time adaptation and collect aligned window scores."""
        cfg = self.cfg
        if self.model is None:
            self.train_valid_phase(tsData)
        elif tsData is not self._train_input:
            # Test split differs from the training curve: re-normalise only.
            self._prepare(tsData)

        # Validation scores -> FPM reference threshold.
        val_scores, val_inputs = self._scores_over(
            self._val_set, cfg.VAL.BATCH_SIZE, collect_inputs=True
        ) if len(self._val_set) else (np.zeros(1), torch.zeros(0, cfg.DATA.WIN_SIZE, cfg.DATA.N_VAR))
        if len(self._val_set) == 0:  # degenerate: fall back to the train split
            val_scores, val_inputs = self._scores_over(
                self._train_set, cfg.VAL.BATCH_SIZE, collect_inputs=True
            )

        thresholder = _Thresholder(cfg, val_scores)

        if cfg.TEST.TTA.ENABLE and cfg.TEST.TTA.METHOD == "CANDI":
            adapter = _MLPAdapter(
                cfg,
                self.model,
                thresholder,
                self.device,
                test_labels=self._test_labels,
                val_scores=val_scores,
                val_inputs=val_inputs,
            )
            loader = DataLoader(
                self._test_set,
                batch_size=int(cfg.TEST.BATCH_SIZE),
                shuffle=False,
                drop_last=False,
                num_workers=int(cfg.DATA_LOADER.NUM_WORKERS),
                pin_memory=bool(cfg.DATA_LOADER.PIN_MEMORY)
                and self.device.type == "cuda",
            )
            self.model.eval()
            scores_all = []
            for inputs, _ in loader:
                inputs = inputs.float().to(self.device)
                with torch.no_grad():
                    scores = self.model.get_anomaly_scores(inputs)
                scores_all.append(torch.flatten(torch.as_tensor(scores)))
                adapter.adapt(inputs, scores)
            score = (
                torch.cat(scores_all, dim=0).cpu().numpy()
                if scores_all
                else np.zeros(0, dtype=np.float64)
            )
        else:
            score, _ = self._scores_over(self._test_set, cfg.TEST.BATCH_SIZE)

        self._score = np.asarray(score, dtype=np.float64).reshape(-1)

    def anomaly_score(self) -> np.ndarray:
        if self._score is None:
            raise RuntimeError("test_phase must run before anomaly_score().")
        return self._score

    def get_y_hat(self) -> np.ndarray:
        # The EasyTSAD evaluator owns thresholding; expose the ratio-derived
        # threshold as a diagnostic prediction only.
        score = self.anomaly_score()
        return (score > float(np.percentile(score, 100 - self.cfg.TEST.THRESHOLD.ANOMALY_RATIO))).astype(int)
