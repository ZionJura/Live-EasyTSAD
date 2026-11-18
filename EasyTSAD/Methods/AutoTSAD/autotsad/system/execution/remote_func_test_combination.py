import json
import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from autotsad.config import METRIC_MAPPING
from autotsad.dataset import TestDataset
from autotsad.system.execution.aggregation import load_scores, algorithm_instances, aggregate_scores
from autotsad.system.execution.algo_selection import select_algorithm_instances
from autotsad.util import save_serialized_results


def _safe_compute_metrics(dataset: TestDataset, combined_score: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
    """Official ``compute_metrics`` with a per-metric guard for undefined inputs.

    ``autotsad.evaluation.compute_metrics`` dispatches every metric of
    ``METRIC_MAPPING`` with exactly one of two inputs:

    * continuous scorings -> ``metric(y_label, y_score)`` (``combined_score``)
    * binary predictions   -> ``metric(y_label, y_pred)`` (``combined_score > 0``)

    The range-based metrics are implemented by ``prts``, which is only defined
    if the ground truth *and* the prediction contain at least one anomaly
    point: ``prts/base/time_series_metrics.py:_prepare_data`` asserts that both
    inputs are binary with a positive entry and then indexes the first anomaly
    position (``real_anomalies_[0]`` / ``predicted_anomalies_[0]``).  A curve
    without any ground-truth anomaly (e.g. WSD curve 90, whose
    ``test_label`` only contains 0) therefore aborts the whole run with an
    ``AssertionError``; ``sklearn``'s AUC helpers raise ``ValueError`` for the
    same reason (a single class in ``y_true``).

    This wrapper keeps the official per-metric call and inputs unchanged and
    only substitutes the neutral value ``0.0`` for the individual metric that
    cannot be computed for the given input.  In particular, metrics that *are*
    well defined are never altered: only a metric that actually raises for an
    undefined input is replaced, so e.g. the continuous
    ``RangePrAUC``/``RocAUC`` values of a combination whose binary prediction
    happens to be degenerate stay untouched, and constant scorings keep the
    documented TimeEval behaviour (``timeeval/metrics/metric.py`` warns and
    returns ``0.0``).  ``combined-score.csv`` is written by the caller before
    this function is invoked, hence the returned score vector is unaffected.

    ``0.0`` is the value TimeEval itself reports for scorings it cannot score,
    so it is used as the neutral ``prts``-undefined placeholder.
    """
    y_label = dataset.label
    y_pred = y_pred.astype(np.int_)

    log = logging.getLogger("autotsad.execution")
    metrics: Dict[str, float] = {}
    for metric_name, metric in METRIC_MAPPING.items():
        try:
            if metric.supports_continuous_scorings():
                score = metric(y_label, combined_score)
            else:
                score = metric(y_label, y_pred)
        except (AssertionError, IndexError, ValueError) as e:
            # prts is undefined for an all-zero ground truth or prediction
            # (no anomaly range to index); sklearn's AUC needs both classes.
            log.warning(f"{dataset.name}: metric {metric_name} is undefined for this input ({e!r}); "
                        f"reporting 0.0.")
            score = 0.0
        metrics[metric_name] = score
    return metrics


def test_single_combination(dataset: TestDataset, results: pd.DataFrame, result_dir: Path, selection: str,
                            normalization: str, aggregation: str, max_instances: int,
                            scores_path: Path) -> Dict[str, Union[str, float]]:
    log = logging.getLogger(f"autotsad.execution.selection.{selection}-{normalization}-{aggregation}")

    # prepare result folder for this combination
    result_dir = result_dir / f"{selection}-{normalization}-{aggregation}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Selecting {max_instances} instances with {selection} and {normalization} normalization and "
             f"{aggregation} aggregation")
    log.debug(f"Loading scores from {scores_path}")
    instances = algorithm_instances(results)
    scores = load_scores(instances,
                         dataset_id=dataset.hexhash,
                         base_scores_path=scores_path,
                         normalization_method=normalization,
                         ensure_finite=True)

    # also filter instances that have non-finite results (in addition to scores)
    n_invalid = len(instances) - scores.shape[1]
    if n_invalid > 0:
        invalid_instances = list(set(instances) - set(scores.columns))
        log.warning(f"{selection}-{normalization}-{aggregation}: Filtering out {n_invalid} instances with non-finite "
                    f"scores: {invalid_instances}!")
    results = results[instances.isin(scores.columns)].copy()

    log.debug(f"Performing algorithm selection with {selection} and max_instances={max_instances}")
    results = select_algorithm_instances(results, scores, selection_method=selection, max_instances=max_instances)
    save_serialized_results(results, result_dir / "selected-instances.csv")

    log.debug(f"Combining scores with {aggregation} aggregation")
    scores = scores[algorithm_instances(results)]
    combined_score = aggregate_scores(scores.values, agg_method=aggregation)
    np.savetxt(result_dir / "combined-score.csv", combined_score, delimiter=",")

    metrics: Dict[str, Union[str, float]] = {"selection-method": selection, "normalization-method": normalization,
                                            "aggregation-method": aggregation}
    metrics.update(_safe_compute_metrics(dataset, combined_score, combined_score > 0))
    log.debug(f"{selection}-{normalization}-{aggregation} metrics: {metrics}")
    with (result_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f)

    return metrics
