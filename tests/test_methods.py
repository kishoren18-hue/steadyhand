"""Checks that the threshold and voting code does what the paper says.

Run:  python -m pytest tests
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from methods import (misfire_curve, crc_threshold, candidate_thresholds,
                     vote_predict, voted_misfire_curve)
from libemg.emg_predictor import EMGClassifier

RNG = np.random.default_rng(0)
K, N = 6, 3000
PRED = RNG.integers(0, K, N)
CONF = np.round(RNG.random(N), 2)          # rounding creates ties on purpose
Y = RNG.integers(0, K, N)
SEG = np.repeat(np.arange(30), 100)        # 30 recordings of 100 windows


def test_misfire_curve_matches_brute_force():
    ts = np.r_[0, np.sort(RNG.random(200)), CONF[:50], 1]
    wrong = PRED != Y
    brute = np.array([np.mean((CONF > t) & wrong) for t in ts])
    assert np.allclose(misfire_curve(CONF, wrong, ts), brute)


def test_vote_predict_matches_libemg():
    """Our rejection + majority vote equals LibEMG's own code, recording by recording."""
    c = EMGClassifier("LDA")
    c.rejection, c.rejection_threshold, c.majority_vote = True, 0.5, 5
    lib = []
    for s in range(30):
        k = SEG == s
        rejected = np.array([c._rejection_helper(p, q) for p, q in zip(PRED[k], CONF[k])])
        lib.append(c._majority_vote_helper(rejected))
    assert (np.concatenate(lib) == vote_predict(PRED, CONF, SEG, 5, 0.5, K)).all()


def test_voted_curve_matches_brute_force():
    """The fast exact algorithm equals voting at every cutoff one by one."""
    grid = np.unique(np.r_[0, np.linspace(0, 1, 101), CONF, 1])
    fast = voted_misfire_curve(PRED, CONF, Y, SEG, 5, grid, K)
    brute = []
    for t in grid:
        v = vote_predict(PRED, CONF, SEG, 5, t, K)
        brute.append(np.mean((v != -1) & (v != Y)))
    assert np.allclose(fast, brute)


def test_conformal_threshold_is_smallest_passing():
    """The chosen cutoff passes (n*avg + 1)/(n + 1) <= alpha and the next lower one fails."""
    people = [(RNG.random(300), RNG.random(300) < 0.25) for _ in range(35)]
    grid = candidate_thresholds(np.concatenate([c for c, _ in people]))
    curves = np.array([misfire_curve(c, w, grid) for c, w in people])
    t = crc_threshold(grid, curves, 0.05)
    bound = lambda x: (35 * curves[:, grid == x].mean() + 1) / 36
    assert bound(t) <= 0.05
    assert bound(grid[grid < t].max()) > 0.05
