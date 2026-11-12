"""Exercise model training, scoring and suffix alignment on synthetic MTS."""
import importlib
from types import SimpleNamespace

import numpy as np
import torch
import EasyTSAD.Controller

module = importlib.import_module('EasyTSAD.Methods.ALoRa-T.ALoRa-T')
rng = np.random.default_rng(42)
data = SimpleNamespace(train=rng.normal(size=(40, 5)), valid=rng.normal(size=(24, 5)),
                       test=rng.normal(size=(26, 5)))
params = dict(window_size=8, train_stride=2, batch_size=4, d_model=10, n_heads=2,
              e_layers=1, epochs=1, learning_rate=0.0001, lambda_reg=10.0,
              rank_threshold=0.01, patience=3, seed=42)
model = module.ALoRaT(params)
model.train_valid_phase(data)
model.test_phase(data)
scores = model.anomaly_score()
assert scores.shape == (len(data.test) - params['window_size'] + 1,)
assert np.isfinite(scores).all()
with torch.no_grad():
    first = torch.as_tensor(data.test[:params['window_size']], dtype=torch.float32, device=model.device).unsqueeze(0)
    reconstructed, attention, _ = model.model(first)
    error = (reconstructed[:, -1] - first[:, -1]).square().mean(dim=-1)
    rank = (torch.linalg.svdvals(attention[-1].mean(dim=1)) > params['rank_threshold']).sum(dim=-1)
    np.testing.assert_allclose(scores[0], (error * rank).cpu().numpy()[0], rtol=1e-5)
print('ALoRa-T selftest passed:', scores.shape, model.device)
