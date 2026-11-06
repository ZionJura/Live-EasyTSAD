# Adapted from xuhongzuo/deep-iForest, algorithms/dif.py and utils.py.
# Original author: Xu Hongzuo (hongzuo.xu@gmail.com), 2022/8/19.
# Apache-2.0; see ../LICENSE and ../NOTICE.
# Modified: TS-only lazy windows; sparse split-path accumulation; paper Eqs.
# (7)-(8) over the complete forest, without leaf correction or zero exclusion.

import random

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .net_torch import DilatedConvEncoder


class WindowDataset(Dataset):
    """The official contiguous windows, stored as slices instead of copies."""

    def __init__(self, series, window_size, stride):
        series = np.asarray(series)
        if series.ndim == 1:
            series = series[:, None]
        if series.ndim != 2 or series.shape[1] == 0:
            raise ValueError("Expected a time series with shape [time, channels].")
        if window_size < 1 or stride < 1:
            raise ValueError("Window size and stride must be positive.")
        if len(series) < window_size:
            raise ValueError(
                f"DIFR requires at least {window_size} observations; got {len(series)}."
            )
        if not np.isfinite(series).all():
            raise ValueError("DIFR requires finite input observations.")
        self.series = np.ascontiguousarray(series, dtype=np.float32)
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (len(self.series) - self.window_size) // self.stride + 1

    def __getitem__(self, index):
        start = index * self.stride
        return self.series[start:start + self.window_size]


def average_path_length(n):
    """Official iForest normalizer (not a correction to the observed path)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n


def tree_statistics(representations, estimator):
    """Return raw edge depth and Eq. (7) for each sample, including zero gaps.

    Sparse decision paths avoid the official sample-by-all-nodes dense matrix.
    Internal node visits equal the number of root-to-leaf edges. Leaf nodes
    have no split feature and are removed before gathering feature values.
    """
    path = estimator.decision_path(representations)
    rows = np.repeat(np.arange(len(representations)), np.diff(path.indptr))
    nodes = path.indices
    tree = estimator.tree_
    is_split = tree.children_left[nodes] != -1
    rows, nodes = rows[is_split], nodes[is_split]
    depth = np.bincount(rows, minlength=len(representations)).astype(np.float64)
    gaps = np.abs(
        representations[rows, tree.feature[nodes]] - tree.threshold[nodes]
    )
    gap_sum = np.bincount(rows, weights=gaps, minlength=len(representations))
    deviation = np.divide(gap_sum, depth, out=np.zeros_like(depth), where=depth > 0)
    return depth, deviation


class DeepIsolationForest:
    """Optimization-free time-series DIF with the approved paper DEAS score.

    All model settings are supplied by the adapter's config dictionary.
    ``representation_preprocess`` and ``score_function`` carry the contract's
    descriptive definitions for the two fixed, non-selectable method stages.
    """

    def __init__(self, params):
        self.params = dict(params)
        p = self.params
        if (p["network_name"] != "dilated_conv" or p["data_type"] != "ts"
                or p["ensemble_method"] != "independent"):
            raise ValueError("DIFR requires independent time-series dilated CNNs.")
        if p["init_distribution"] != "normal":
            raise ValueError("DIFR uses normally initialized network weights.")
        if p["n_processes"] != 1:
            raise ValueError("The contracted DIFR implementation uses one scoring process.")
        for key in ("window_size", "train_stride", "test_stride", "n_ensemble",
                    "n_estimators", "max_samples", "hidden_dim", "rep_dim",
                    "kernel_size", "dilation_base", "batch_size"):
            if not isinstance(p[key], int) or p[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if not isinstance(p["layers"], int) or p["layers"] < 0:
            raise ValueError("layers must be a nonnegative integer.")
        if p["test_stride"] != 1:
            raise ValueError("DIFR's timestamp-aligned test stride must be 1.")
        self.representation_preprocess = p["representation_preprocess"]
        self.score_function = p["score_function"]
        requested_device = p["device"]
        if requested_device.startswith("cuda when available"):
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.net_lst = []
        self.clf_lst = []
        self.n_features = None
        self.normalizer_ = None

    @staticmethod
    def set_seed(seed):
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def make_network(self, n_features):
        p = self.params
        return DilatedConvEncoder(
            n_features, hidden_dim=p["hidden_dim"], rep_dim=p["rep_dim"],
            layers=p["layers"], kernel_size=p["kernel_size"],
            dilation_base=p["dilation_base"], activation=p["activation"],
            pooling=p["pooling"], dropout=p["dropout"],
        )

    def deep_transfer(self, windows, net):
        loader = DataLoader(
            windows, batch_size=self.params["batch_size"], drop_last=False,
            shuffle=False, pin_memory=(self.device.type == "cuda"),
        )
        reduced = []
        with torch.no_grad():
            for batch in loader:
                reduced.append(net(batch.to(self.device, dtype=torch.float32)).cpu())
        reduced = torch.cat(reduced).numpy()
        # Official behavior: this scaler is fitted anew for EACH collection,
        # including the full test collection; it is never retained from fit.
        reduced = StandardScaler().fit_transform(reduced)
        return np.ascontiguousarray(np.tanh(reduced), dtype=np.float32)

    def fit(self, series):
        p = self.params
        windows = WindowDataset(series, p["window_size"], p["train_stride"])
        self.n_features = windows.series.shape[1]
        self.net_lst, self.clf_lst = [], []
        self.set_seed(p["random_state"])
        self.ensemble_seeds_ = np.random.randint(0, 100000, p["n_ensemble"])
        sample_size = min(p["max_samples"], len(windows))
        self.normalizer_ = average_path_length(sample_size)
        for seed in self.ensemble_seeds_:
            self.set_seed(seed)
            net = self.make_network(self.n_features).to(self.device)
            for name, param in net.named_parameters():
                if name.endswith("weight"):
                    torch.nn.init.normal_(param, mean=p["init_mean"], std=p["init_std"])
            net.requires_grad_(False)
            net.eval()
            reduced = self.deep_transfer(windows, net)
            forest = IsolationForest(
                n_estimators=p["n_estimators"], max_samples=sample_size,
                random_state=int(seed),
            )
            forest.fit(reduced)
            self.net_lst.append(net)
            self.clf_lst.append(forest)
        return self

    def decision_function(self, series):
        if not self.clf_lst:
            raise RuntimeError("Fit DIFR before computing anomaly scores.")
        p = self.params
        windows = WindowDataset(series, p["window_size"], p["test_stride"])
        if windows.series.shape[1] != self.n_features:
            raise ValueError("Train and test channel counts must match.")
        depth_sum = np.zeros(len(windows), dtype=np.float64)
        deviation_sum = np.zeros_like(depth_sum)
        if self.normalizer_ == 0:
            return deviation_sum
        tree_count = 0
        for net, forest in zip(self.net_lst, self.clf_lst):
            reduced = self.deep_transfer(windows, net)
            for estimator in forest.estimators_:
                depth, deviation = tree_statistics(reduced, estimator)
                depth_sum += depth
                deviation_sum += deviation
                tree_count += 1
        # Eq. (8) is a product of global means over ALL representation trees.
        # It is not the mean of separate per-representation products.
        return np.exp2(-depth_sum / (tree_count * self.normalizer_)) * (
            deviation_sum / tree_count
        )
