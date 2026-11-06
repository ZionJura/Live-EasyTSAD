"""Short CPU checks; preload EasyTSAD.Controller as the managed runner does."""

from pathlib import Path
from types import SimpleNamespace
import json
import tomllib

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor

from .DIFR import DIFR
from .src import DeepIsolationForest
from .src.dif import WindowDataset, average_path_length, tree_statistics


def reference_scores(model, series):
    """Independent literal root-to-leaf traversal and full-forest Eq. (8)."""
    p = model.params
    windows = WindowDataset(series, p["window_size"], p["test_stride"])
    all_depths, all_deviations = [], []
    for net, forest in zip(model.net_lst, model.clf_lst):
        reps = model.deep_transfer(windows, net)
        for estimator in forest.estimators_:
            tree = estimator.tree_
            depths, deviations = [], []
            for row in reps:
                node, gaps = 0, []
                while tree.children_left[node] != -1:
                    value = row[tree.feature[node]]
                    gaps.append(abs(float(value) - tree.threshold[node]))
                    node = (tree.children_left[node] if value <= tree.threshold[node]
                            else tree.children_right[node])
                depths.append(len(gaps))
                deviations.append(sum(gaps) / len(gaps) if gaps else 0.0)
            all_depths.append(depths)
            all_deviations.append(deviations)
    if model.normalizer_ == 0:
        return np.zeros(len(windows))
    return np.exp2(-np.mean(all_depths, axis=0) / model.normalizer_) * np.mean(
        all_deviations, axis=0
    )


def main():
    # This only limits the tiny CPU self-test, never the detector's runtime.
    torch.set_num_threads(1)
    method = Path(__file__).resolve().parent
    config = tomllib.loads((method / "config.toml").read_text())
    params = config["Model_Params"]["Default"]
    root = method.parents[2]
    contract = json.loads((root / "EasyTSAD/Harness/tasks/DIFR/contract.json").read_text())
    for claim in contract["claims"]:
        key = claim.get("config_key")
        if key:
            actual = (config["Data_Params"][key.split(".", 1)[1]]
                      if key.startswith("Data_Params.") else params[key])
            assert actual == claim["value"], claim["id"]
    registry = tomllib.loads((root / "envs/registry.toml").read_text())
    assert [name for name, env in registry["envs"].items()
            if "DIFR" in env["methods"]] == ["torch2"]
    params = dict(params, device="cpu")
    rng = np.random.RandomState(19)
    train = rng.normal(size=(360, 1)).astype(np.float32)
    test = rng.normal(size=(118, 1)).astype(np.float32)
    windows = WindowDataset(train, params["window_size"], params["train_stride"])
    expected = np.array([train[i:i + params["window_size"]]
                         for i in range(0, len(train) - params["window_size"] + 1,
                                        params["train_stride"])])
    np.testing.assert_array_equal(np.stack([windows[i] for i in range(len(windows))]), expected)

    adapter = DIFR(params)
    # Validation and labels deliberately are not provided: the adapter uses neither.
    adapter.train_valid_phase(SimpleNamespace(train=train[:, 0]))
    adapter.test_phase(SimpleNamespace(test=test[:, 0]))
    scores = adapter.anomaly_score()
    assert scores.shape == (len(test) - params["window_size"] + 1,)
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert len(adapter.model.net_lst) == params["n_ensemble"]
    assert sum(len(f.estimators_) for f in adapter.model.clf_lst) == (
        params["n_ensemble"] * params["n_estimators"]
    )
    np.testing.assert_allclose(scores, reference_scores(adapter.model, test), rtol=1e-12)
    assert all(not parameter.requires_grad
               for net in adapter.model.net_lst for parameter in net.parameters())
    for seed, net in zip(adapter.model.ensemble_seeds_, adapter.model.net_lst):
        # The official seed/init order must reproduce every parameter exactly.
        adapter.model.set_seed(seed)
        clone = adapter.model.make_network(1)
        for name, param in clone.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.normal_(param, mean=params["init_mean"], std=params["init_std"])
        assert all(torch.equal(a, b) for a, b in zip(net.parameters(), clone.parameters()))
    np.testing.assert_array_equal(adapter.model.decision_function(test), scores)

    # A threshold of exactly zero verifies that zero gaps count in the denominator.
    tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        np.array([[-1.0], [1.0]], dtype=np.float32), [0.0, 1.0]
    )
    depth, deviation = tree_statistics(np.array([[-1.0], [0.0], [1.0]], dtype=np.float32), tree)
    np.testing.assert_array_equal(depth, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(deviation, [1.0, 0.0, 1.0])
    assert average_path_length(1) == 0 and average_path_length(2) == 1

    mts = DeepIsolationForest(params).fit(rng.normal(size=(360, 3)))
    mts_test = rng.normal(size=(118, 3))
    mts_scores = mts.decision_function(mts_test)
    assert mts_scores.shape == scores.shape and np.isfinite(mts_scores).all()
    np.testing.assert_allclose(mts_scores, reference_scores(mts, mts_test), rtol=1e-12)
    constant = DeepIsolationForest(params).fit(np.zeros(360))
    np.testing.assert_array_equal(constant.decision_function(np.zeros(118)), np.zeros_like(scores))
    one_window = DeepIsolationForest(params).fit(np.zeros(params["window_size"]))
    np.testing.assert_array_equal(one_window.decision_function(test), np.zeros_like(scores))
    try:
        adapter.model.decision_function(np.zeros(params["window_size"] - 1))
    except ValueError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("Short input must raise an informative error")
    print("PASS: 27 config mappings, unique registration, UTS/MTS forward fitting/scoring,")
    print("independent full-forest DEAS reference, exact initialization, repeatability,")
    print("window alignment, zero gaps, constant trees, one-window forest, short-input error.")


if __name__ == "__main__":
    main()
