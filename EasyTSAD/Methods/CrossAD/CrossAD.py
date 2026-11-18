import numpy as np
import torch

from .. import BaseMethod
from ...DataFactory import TSData
from .src.pipeline import train_model, score_series


class CrossAD(BaseMethod):
    def __init__(self, params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._score = None

    def train_valid_phase(self, tsTrain: TSData):
        self.model = train_model(tsTrain.train, tsTrain.valid, self.params, self.device)

    def test_phase(self, tsData: TSData):
        if self.model is None:
            raise RuntimeError("CrossAD must be trained before testing")
        self._score = score_series(self.model, tsData.test, self.params, self.device)

    def anomaly_score(self) -> np.ndarray:
        return self._score

    def param_statistic(self, save_file):
        pass
