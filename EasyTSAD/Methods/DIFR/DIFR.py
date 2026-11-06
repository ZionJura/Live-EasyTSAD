"""EasyTSAD adapter for the contract-defined time-series deep isolation forest."""

import numpy as np

from .. import BaseMethod
from ...DataFactory import TSData
from .src import DeepIsolationForest


class DIFR(BaseMethod):
    def __init__(self, params: dict):
        super().__init__()
        self.model = DeepIsolationForest(params)
        self._scores = None

    def train_valid_phase(self, tsTrain: TSData):
        # The official TS experiment fits on train alone, without optimization.
        self._scores = None
        self.model.fit(tsTrain.train)

    def test_phase(self, tsData: TSData):
        self._scores = self.model.decision_function(tsData.test)

    def anomaly_score(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("Call test_phase before requesting DIFR scores.")
        # Full-window scores are a suffix: index i belongs to i+window_size-1.
        return self._scores

    def param_statistic(self, save_file):
        # This hook is called before fitting, so the channel count is unknown.
        # State the assumption explicitly instead of constructing a dummy net.
        with open(save_file, "w", encoding="utf-8") as handle:
            handle.write(
                "DIFR uses fixed random CNNs and isolation trees; no trainable "
                "optimization parameters.\n"
                f"Representation ensemble: {self.model.params['n_ensemble']}\n"
                f"Trees per representation: {self.model.params['n_estimators']}\n"
            )
