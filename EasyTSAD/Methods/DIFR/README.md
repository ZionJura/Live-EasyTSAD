# DIFR

EasyTSAD adapter for Deep Isolation Forest for Anomaly Detection
(arXiv:2206.06602), using the approved time-series reproduction contract.

`src/net_torch.py` adapts the official TS2Vec-style residual dilated CNN.
`src/dif.py` contains window construction, random representation extraction,
isolation forests, and scoring, with no framework imports. `DIFR.py` passes
training and test values to the algorithm and returns the score suffix.

All 27 contract claims carrying a `config_key` are copied verbatim into
`config.toml`. The long representation preprocessing, score function, and
device strings are intentional: they match the approved contract exactly.
No model defaults are duplicated in the adapter or algorithm constructors.

The method uses 100-point full windows, training stride 10 and test stride 1,
50 fixed random CNNs, and 6 isolation trees per representation with at most
256 training windows per tree. There is no optimization, loss, learning rate,
or training epoch loop. Training uses only the supplied training series.
Both univariate and jointly processed multivariate arrays are supported.

Following the official time-series code, every network's representations are
standardized and passed through tanh. The scaler is fitted separately to each
complete training or test representation collection. Test scaling therefore
uses the unlabeled test collection's statistics and is not an online procedure.

Following the approved paper interpretation, DEAS combines statistics across
all 300 trees: `2**(-mean(path_length)/c(sample_size))*mean(deviation)`.
Path length counts traversed edges without a terminal-leaf population
correction. Deviation averages over every visited internal split node,
including exact-zero gaps. These choices intentionally differ from the
official code and are recorded in the harness notes. Trees without splits
contribute zero deviation; a forest fitted on one window returns zero scores.

Each test score belongs to its window's final timestamp. The returned array
has length `len(test)-window_size+1`; EasyTSAD aligns it from the end. No
padding, score smoothing, normalization, label-based segment adjustment, or
threshold selection occurs in the adapter. Splits shorter than the window
raise an informative error. Input observations must be finite.

DIFR is registered only in the existing `torch2` environment. Run its quick
CPU self-test from the repository root with the framework's normal import
order (the same order used by `scripts/tsad check`):

```sh
scripts/tsad run DIFR -- python -c "import EasyTSAD.Controller; import runpy; runpy.run_module('EasyTSAD.Methods.DIFR.selftest', run_name='__main__')"
```

This checks exact configuration mappings, single environment registration,
lazy window equivalence, UTS/MTS scoring, independent literal tree traversal
of paper DEAS, official seed/weight initialization, repeatability, zero split
gaps, constant and one-window forests, and short-input handling. It uses the
full configured ensemble on small random arrays and no benchmark labels.

See `NOTICE` and `LICENSE` for the original source attribution and Apache-2.0
license. The paper/code scoring differences and operational assumptions are
also documented in the task's approved `contract.json` and harness notes.
