# ALoRa-T

Multivariate anomaly detection from [Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](https://openreview.net/forum?id=ZtPIBpVojC), also available as [arXiv:2602.08467](https://arxiv.org/abs/2602.08467). The network files are from the official [ALoRa repository](https://github.com/CharisShimillas/ALoRa), commit `97dcc4a337710e6dc72c1a67893717c9538bae1a`, under the EUPL 1.2; see LICENSE.txt.

Use method `ALoRa-T` with schema `mts` and a multivariate dataset of at least two channels. The framework loads the module `EasyTSAD.Methods.ALoRa-T.ALoRa-T`. The dedicated environment is defined by this directory's `pyproject.toml` and `uv.lock`. Sync it from the repository root:

```bash
scripts/tsad sync alora --locked
```

Run the controller example with `scripts/tsad run ALoRa-T -- python <script.py>`. The model is fitted separately for each series. Input arrays must be finite, two-dimensional and at least `window_size` long. Preprocessing defaults to the framework's training-based z-score. Validation uses the framework's validation split, which MTSData currently sets equal to the training set.

For an existing EasyTSAD-format MTS dataset, run via the controller:

```python
import importlib
from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations.Protocols import PointPrc, PointRoc

importlib.import_module("EasyTSAD.Methods.ALoRa-T.ALoRa-T")
controller = TSADController()
controller.set_dataset(["SMD"], dirname="/path/to/datasets", dataset_type="MTS")
controller.run_exps("ALoRa-T", "mts")
controller.set_evals([PointPrc(), PointRoc()])
controller.do_evals("ALoRa-T", "mts")
```

Each score is the mean squared reconstruction error at the last point of a window multiplied by the thresholded rank of the last layer's head-averaged attention matrix. The returned array has length `len(test) - window_size + 1`, and its first value corresponds to index `window_size - 1`; the framework aligns scores to the end of the test labels. No test labels are used in training or scoring.

For more than 512 channel pairs, pairs are selected by absolute Spearman correlation from training data, matching the upstream preprocessing. For fewer pairs the upstream combination order is retained. Unlike upstream, validation does not use test data, the regularizer truncates the first singular value per sample (upstream slices the batch axis), and the initial partial-window scores are omitted. Small channel counts use at most one attention head per embedding dimension, and odd embedding widths are supported. These changes mean results are not an exact execution of the published repository.

The isolated environment uses Python 3.11, PyTorch 2.6 and CUDA 12.4 wheels. NVIDIA driver 550 or later is required for the CUDA 12.4 build; CPU execution is supported. The authors report Linux, Python 3.7.6, PyTorch 1.10.2, CUDA 10.2, driver 440.64 and two Tesla V100 32 GB GPUs. The local host has NVIDIA driver 550.144.03 and RTX 3080 Ti GPUs (12 GiB each). Runtime and memory depend on channels, windows and batch size; reduce `batch_size` if needed.

## Verification run

On 2026-11-12, the framework MTS controller trained and evaluated the SMD CSV data distributed in the upstream repository, using the default configuration and one RTX 3080 Ti. The environment was created from this method's lock file. The time-index column was removed from test values and labels; the 38 sensor columns and all 28,479 train and 28,479 test rows were retained. The framework used z-score preprocessing and its default SMD margin `(0, 5)`. Training and validation took 27.94 s, scoring 1.34 s, metric computation 0.21 s, and peak PyTorch reserved GPU memory was 0.311 GiB. The 28,460 finite scores yielded point AUPRC 0.3941 and point AUROC 0.8301. These are integration checks, not a comparison to the paper's reported metrics: preprocessing, validation, rank regularization and score alignment differ as described above. Times include neither Python startup nor environment setup.
