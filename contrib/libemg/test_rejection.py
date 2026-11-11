"""Tests for the proposed libemg/rejection.py, including against LibEMG's own
EMGClassifier rejection block."""
import numpy as np
import pytest
from rejection import (misfires, misfire_curve, conformal_rejection_threshold,
                       AdaptiveRejection)


def test_rest_never_misfires():
    assert misfires([0, 1, 2], [1, 1, 1], rest_class=0).tolist() == [False, False, True]


def test_curve_matches_libemg_rejection_block():
    from libemg.emg_predictor import EMGClassifier
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(600, 8)), rng.integers(0, 4, 600)
    X[np.arange(600), y] += 1.5
    clf = EMGClassifier("LDA"); clf.fit(feature_dictionary={"training_features": X[:300], "training_labels": y[:300]})
    pred, prob = clf.run(X[300:])
    for t in (0.5, 0.8, 0.9, 0.97):
        clf.add_rejection(threshold=t)
        out, _ = clf.run(X[300:])
        ref = ((out != -1) & (out != y[300:]) & (out != 0)).mean()
        got = misfire_curve(prob, misfires(pred, y[300:], 0), np.array([t]))[0]
        assert np.isclose(ref, got)


def test_conformal_expected_misfire_below_target():
    rng = np.random.default_rng(1)
    def person():
        y = rng.integers(0, 4, 400)
        acc = rng.uniform(0.4, 0.95)
        p = np.where(rng.random(400) < acc, y, rng.integers(0, 4, 400))
        c = np.clip(rng.beta(5, 1.5, 400) + 0.1 * (p == y), 0, 1)
        return p, c, y
    res = []
    for _ in range(300):
        cal = [person() for _ in range(30)]
        t = conformal_rejection_threshold(cal, 0.1, rest_class=0)
        p, c, y = person()
        res.append(((c > t) & misfires(p, y, 0)).mean())
    assert np.mean(res) <= 0.1 + 0.01


@pytest.mark.parametrize("seed", range(10))
def test_adaptive_bound_any_sequence(seed):
    rng = np.random.default_rng(seed)
    rej = AdaptiveRejection(0.05, 0.9)
    mis = 0
    for k in range(5000):
        c = rng.uniform(0.3, 1)
        wrong = rng.random() < (0.05 if k < 2500 else 0.8)
        act = c > rej.threshold
        mis += act and wrong
        rej.update(act and wrong)
    assert mis / 5000 <= rej.bound() + 1e-12
