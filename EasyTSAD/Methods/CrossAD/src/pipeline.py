"""CrossAD algorithm wrapper (no EasyTSAD imports).

This module reproduces the official TSB-AD-U training / scoring path of
CrossAD (github.com/decisionintelligence/CrossAD, commit ff0fb2d) using the
adapted official model modules in this folder.  It performs:

  * the official per-partition z-score standardisation (population std,
    ddof=0, epsilon only when std == 0),
  * the official window extraction (train stride 1, validation/inference
    stride == window_size),
  * the official Adam / type-1 learning-rate schedule / early-stopping loop,
  * the official elementwise squared-residual + linear-interpolation score and
    the sklearn MinMaxScaler post-processing from run_TSBAD.py.

The two documented integration departures (suffix alignment for EasyTSAD's
end alignment, and left-padding of partitions shorter than window_size) live
here as well, per the frozen contract.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler

from .Basic_CrossAD import Basic_CrossAD, _Configs


# ---------------------------------------------------------------------------
# Data helpers (mirror exp/exp_TSBAD.py: ReconstructDataset)
# ---------------------------------------------------------------------------

def _to_series_2d(series):
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    if series.size == 0 or series.shape[0] == 0:
        raise ValueError("CrossAD input partition must not be empty")
    return series


def _normalize(data, epsilon=1e-8):
    """Official ReconstructDataset._normalize_data.

    (x - mean) / std over axis 0 (per channel), population std (np.std ddof=0),
    std replaced only when exactly zero.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, epsilon, std)  # Avoid division by zero
    return (data - mean) / std


def _left_pad_first(data, window_size):
    """Left-pad a real partition with its first value to one full window."""
    n = data.shape[0]
    if n < window_size:
        pad = window_size - n
        data = np.concatenate([np.repeat(data[:1], pad, axis=0), data], axis=0)
    return data


def _windows(data, window_size, stride):
    """Official sample_num = max(0, (n - window) // stride + 1) windows."""
    n = data.shape[0]
    sample_num = max(0, (n - window_size) // stride + 1)
    if sample_num == 0:
        return None
    X = np.stack(
        [data[i * stride: i * stride + window_size] for i in range(sample_num)]
    )
    return X.astype(np.float32)


class _WindowDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.X[index]


# ---------------------------------------------------------------------------
# Model config from config.toml dict
# ---------------------------------------------------------------------------

def build_model(params, device):
    ms_kernels = [params["pool_kernel_{}".format(i)] for i in range(5)]
    configs = _Configs(
        {
            "seq_len": params["window_size"],
            "patch_len": params["patch_len"],
            "d_model": params["d_model"],
            "ms_kernels": ms_kernels,
            "ms_method": params["ms_method"],
            "patch_dropout": params["patch_dropout"],
            "norm": params["norm"],
            "attn_dropout": params["attn_dropout"],
            "proj_dropout": params["proj_dropout"],
            "ff_dropout": params["ff_dropout"],
            "activation": params["activation"],
            "d_ff": params["d_ff"],
            "e_layers": params["e_layers"],
            "d_layers": params["d_layers"],
            "m_layers": params["m_layers"],
            "n_heads": params["n_heads"],
            "topk": params["topk"],
            "bank_size": params["bank_size"],
            "n_query": params["n_query"],
            "query_len": params["query_len"],
            "decay": params["decay"],
            "epsilon": params["epsilon"],
        }
    )
    return Basic_CrossAD(configs).to(device)


# ---------------------------------------------------------------------------
# Training (mirror exp/exp_TSBAD.py: Exp_Anomaly_Detection.train / vali)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return True


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _restore(model, snapshot):
    model.load_state_dict({k: v.to(next(model.parameters()).device)
                           for k, v in snapshot.items()})


def _adjust_learning_rate(optimizer, epoch, base_lr, lradj="type1"):
    if lradj == "type1":
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
    else:
        raise ValueError(f"unsupported lr_schedule: {lradj}")

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train_model(train_arr, valid_arr, params, device):
    """Train on EasyTSAD-supplied train/valid arrays; return best model.

    Each partition is independently standardised on its full real length,
    short partitions are left-padded with their first value, then windows are
    extracted with the official strides (train 1, validation window_size).
    """
    window_size = params["window_size"]
    train_stride = params["train_stride"]
    batch_size = params["batch_size"]
    norm_eps = params["normalization_epsilon"]

    seed = params["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = build_model(params, device)

    model_optim = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(params["adam_beta1"], params["adam_beta2"]),
        eps=params["adam_epsilon"],
        weight_decay=params["weight_decay"],
    )

    train_data = _windows(
        _left_pad_first(_normalize(_to_series_2d(train_arr), norm_eps), window_size),
        window_size,
        train_stride,
    )
    valid_data = _windows(
        _left_pad_first(_normalize(_to_series_2d(valid_arr), norm_eps), window_size),
        window_size,
        window_size,
    )

    train_loader = DataLoader(
        _WindowDataset(train_data), batch_size=batch_size,
        shuffle=params["shuffle_train"], drop_last=params["drop_last"],
    )
    vali_loader = DataLoader(
        _WindowDataset(valid_data), batch_size=batch_size,
        shuffle=False, drop_last=params["drop_last"],
    )

    early_stopping = EarlyStopping(patience=params["patience"], verbose=False)
    best_snapshot = None

    for epoch in range(params["epochs"]):
        train_loss = []
        model.train()
        for i, (batch_x, _) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            ms_loss, _ = model(batch_x, None, None, None)
            loss = ms_loss
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)

        total_loss = []
        model.eval()
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(device)
                ms_loss, _ = model(batch_x, None, None, None)
                total_loss.append(ms_loss.item())
        vali_loss = np.average(total_loss)
        model.train()

        if early_stopping(vali_loss):
            best_snapshot = _snapshot(model)
        if early_stopping.early_stop:
            break
        _adjust_learning_rate(model_optim, epoch + 1, params["learning_rate"],
                              params["lr_schedule"])

    if best_snapshot is None:  # no epoch ran (e.g. degenerate tiny data)
        best_snapshot = _snapshot(model)
    _restore(model, best_snapshot)
    return model


# ---------------------------------------------------------------------------
# Scoring (mirror exp/exp_TSBAD.py: Exp_Anomaly_Detection.test + run_TSBAD.py)
# ---------------------------------------------------------------------------

def score_series(model, test_arr, params, device):
    """Return the min-max scaled continuous anomaly score for one test array."""
    window_size = params["window_size"]
    batch_size = params["batch_size"]

    test_data = _to_series_2d(test_arr)
    data = _normalize(test_data, params["normalization_epsilon"])

    n = data.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    if n < window_size:
        data = _left_pad_first(data, window_size)
        X = np.stack([data])
        X = X.astype(np.float32)
        model.eval()
        with torch.no_grad():
            batch_x = torch.tensor(X, dtype=torch.float32).to(device)
            ms_score, _ = model.infer(batch_x, None, None, None)
        scores = ms_score.detach().cpu().numpy()          # [1, window, c]
        scores = scores.reshape(-1, scores.shape[-1])     # [window, c]
        scores = np.mean(scores, axis=-1)                 # [window]
        scores = scores[window_size - n:]                 # drop synthetic positions
    else:
        offset = n % window_size
        n_win = n // window_size
        X = np.stack(
            [data[offset + i * window_size: offset + i * window_size + window_size]
             for i in range(n_win)]
        ).astype(np.float32)

        loader = DataLoader(
            _WindowDataset(X), batch_size=batch_size, shuffle=False
        )
        chunks = []
        model.eval()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.float().to(device)
                ms_score, _ = model.infer(batch_x, None, None, None)
                chunks.append(ms_score.detach().cpu().numpy())
        scores = np.concatenate(chunks, axis=0)           # [n_win, window, c]
        scores = scores.reshape(-1, scores.shape[-1])     # [n_win*window, c]
        scores = np.mean(scores, axis=-1)                 # [n_win*window]

    # Official TSB-AD-U final post-processing (run_TSBAD.py:L28)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scores = scaler.fit_transform(scores.reshape(-1, 1)).ravel()
    return scores.astype(np.float64)
