"""EasyTSAD adapter for the upstream ALoRa-T model."""
import copy
import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from .. import BaseMethod, BaseMethodMeta
from .ALoRa import ALoRaT as Network


class Windows(Dataset):
    def __init__(self, values, window, stride):
        self.values = torch.as_tensor(np.asarray(values, dtype=np.float32))
        if self.values.ndim != 2 or self.values.shape[1] < 2:
            raise ValueError("ALoRa-T requires multivariate input with at least two channels")
        if len(self.values) < window or not torch.isfinite(self.values).all():
            raise ValueError("Input must be finite and at least one window long")
        self.window, self.stride = window, stride

    def __len__(self):
        return (len(self.values) - self.window) // self.stride + 1

    def __getitem__(self, index):
        start = index * self.stride
        return self.values[start:start + self.window]


class ALoRaT(BaseMethod):
    def __init__(self, params):
        self.p = params
        self.window = int(params['window_size'])
        self.device = torch.device(params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.scores = None

    def loader(self, values, stride, shuffle=False):
        return DataLoader(Windows(values, self.window, stride), batch_size=self.p['batch_size'], shuffle=shuffle)

    def train_valid_phase(self, data):
        torch.manual_seed(self.p['seed'])
        train = self.loader(data.train, self.p['train_stride'], True)
        valid = self.loader(data.valid, self.p['train_stride'])
        channels = train.dataset.values.shape[1]
        if valid.dataset.values.shape[1] != channels:
            raise ValueError('Train and validation channel counts differ')
        width = min(self.p['d_model'], 512, channels * (channels - 1) // 2)
        self.model = Network(self.window, channels, channels, d_model=self.p['d_model'],
                             n_heads=min(self.p['n_heads'], width), e_layers=self.p['e_layers']).to(self.device)
        if channels * (channels - 1) // 2 > 512:
            ranked = np.apply_along_axis(rankdata, 0, np.asarray(data.train))
            ranked -= ranked.mean(axis=0)
            norm = np.linalg.norm(ranked, axis=0)
            norm[norm == 0] = 1
            corr = (ranked / norm).T @ (ranked / norm)
            left, right = np.triu_indices(channels, 1)
            order = np.argsort(-np.abs(corr[left, right]))[:width]
            pairs = np.stack((left[order], right[order]), axis=1)
            self.model.embedding.value_embedding.pairs_idx.copy_(torch.as_tensor(pairs, device=self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p['learning_rate'])
        best, best_state, bad = float('inf'), None, 0
        for epoch in range(self.p['epochs']):
            self.model.train()
            for batch in train:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output, attention, _ = self.model(batch)
                penalty = 0
                for matrix in attention:
                    singular = torch.linalg.svdvals(matrix.mean(dim=1))[:, 1:]
                    penalty = penalty + (singular / (singular + 1)).sum()
                loss = (output - batch).square().mean() + self.p['lambda_reg'] * penalty / len(attention)
                loss.backward()
                optimizer.step()
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in valid:
                    batch = batch.to(self.device)
                    output, _, _ = self.model(batch)
                    losses.append(((output - batch).square().mean().item(), len(batch)))
            value = sum(a * b for a, b in losses) / sum(b for _, b in losses)
            if value < best:
                best, best_state, bad = value, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.p['patience']:
                    break
            for group in optimizer.param_groups:
                group['lr'] = self.p['learning_rate'] * (0.5 ** epoch)
        self.model.load_state_dict(best_state)

    def test_phase(self, data):
        if self.model is None:
            raise RuntimeError('Train the model before testing')
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in self.loader(data.test, 1):
                batch = batch.to(self.device)
                output, attention, _ = self.model(batch)
                error = (output[:, -1] - batch[:, -1]).square().mean(dim=-1)
                rank = (torch.linalg.svdvals(attention[-1].mean(dim=1)) > self.p['rank_threshold']).sum(dim=-1)
                scores.append((error * rank).cpu().numpy())
        self.scores = np.concatenate(scores)

    def anomaly_score(self):
        if self.scores is None:
            raise RuntimeError('Call test_phase first')
        return self.scores


BaseMethodMeta.registry['ALoRa-T'] = ALoRaT
