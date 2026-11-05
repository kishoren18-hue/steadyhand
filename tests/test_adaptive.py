"""The adaptive cutoff's guarantee holds on every prefix of every sequence,
including sequences built to defeat it."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steadyhand import AdaptiveCutoff, simulate
from steadyhand.adaptive import S_MAX, to_score


def _check(conf, wrong, target, start, eta):
    acted, _ = simulate(conf, wrong, target, start, eta)
    T = np.arange(1, len(conf) + 1)
    running = np.cumsum(acted & wrong) / T
    bound = target + (S_MAX + eta - min(to_score(start), S_MAX)) / (eta * T)
    assert np.all(running <= bound + 1e-12)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("eta", [0.05, 0.5, 2.0])
def test_bound_random_sequences(seed, target, eta):
    rng = np.random.default_rng(seed)
    n = 3000
    conf = rng.beta(5, 1, n)
    wrong = rng.random(n) < rng.uniform(0, 1)   # error rates from tiny to huge
    _check(conf, wrong, target, rng.uniform(0, 1), eta)


@pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
def test_bound_adversarial_sequence(target):
    """Every decision is wrong and maximally confident: the worst case."""
    n = 5000
    _check(np.ones(n), np.ones(n, bool), target, 0.0, 0.5)


def test_bound_with_drift():
    """Accurate for a while, then a sudden shift makes most decisions wrong."""
    rng = np.random.default_rng(1)
    conf = rng.beta(8, 1, 6000)
    wrong = np.r_[rng.random(3000) < 0.05, rng.random(3000) < 0.6]
    _check(conf, wrong, 0.05, 0.9, 0.5)


def test_class_matches_simulate():
    rng = np.random.default_rng(3)
    conf, wrong = rng.beta(5, 1, 500), rng.random(500) < 0.3
    acted, _ = simulate(conf, wrong, 0.05, 0.9, 0.5)
    ad = AdaptiveCutoff(target=0.05, start=0.9, eta=0.5)
    online = []
    for c, w in zip(conf, wrong):
        a = ad.should_act(c)
        online.append(a)
        ad.update(acted=a, correct=not w)
    assert np.array_equal(acted, online)
    assert ad.n_misfires / ad.n_updates <= ad.bound()


def test_conformal_api_matches_function():
    from steadyhand import ConformalCutoff, crc_threshold, misfire_curve, candidate_thresholds
    rng = np.random.default_rng(4)
    people = [dict(conf=rng.random(200), pred=rng.integers(0, 4, 200),
                   y=rng.integers(0, 4, 200)) for _ in range(30)]
    cut = ConformalCutoff(target=0.1).fit(people)
    grid = candidate_thresholds(np.concatenate([p["conf"] for p in people]))
    curves = [misfire_curve(p["conf"], p["pred"] != p["y"], grid) for p in people]
    assert cut.threshold_ == crc_threshold(grid, curves, 0.1)
    assert cut.floor == pytest.approx(1 / 31)
