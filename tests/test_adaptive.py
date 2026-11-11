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


# ---------------------------------------------------------------- v0.3 additions
from steadyhand import harmful, ConformalCutoff
from steadyhand.adaptive import simulate as _sim, S_MAX as _S, to_score as _ts


def test_harmful_rest_is_never_a_misfire():
    pred = np.array([0, 1, 2, 0, 2])
    y = np.array([1, 1, 0, 0, 1])
    assert harmful(pred, y, rest=0).tolist() == [False, False, True, False, True]
    assert harmful(pred, y).tolist() == [True, False, True, False, True]


@pytest.mark.parametrize("p", [0.05, 0.1, 0.3])
@pytest.mark.parametrize("seed", range(20))
def test_ipw_weighted_bound_is_deterministic(p, seed):
    rng = np.random.default_rng(seed)
    T = 3000
    conf = rng.uniform(0.3, 1.0, T)
    wrong = rng.random(T) < np.r_[np.full(T // 2, 0.05), np.full(T - T // 2, 0.7)]  # shift
    fb = rng.random(T) < p
    alpha, eta, start = 0.05, 0.5, 0.9
    acted, _ = _sim(conf, wrong, alpha, start, eta, fb, rate=p)
    weighted = ((acted & wrong) & fb).sum() / p / T
    bound = alpha + (_S + eta / p - min(_ts(start), _S)) / (eta * T)
    assert weighted <= bound + 1e-12


def test_ipw_tracks_true_rate_in_expectation():
    rng = np.random.default_rng(0)
    T, p, alpha = 4000, 0.1, 0.05
    conf = rng.uniform(0.3, 1.0, T)
    wrong = rng.random(T) < 0.5
    rates = []
    for s in range(200):
        fb = np.random.default_rng(s).random(T) < p
        acted, _ = _sim(conf, wrong, alpha, 0.9, 0.5, fb, rate=p)
        rates.append((acted & wrong).mean())
    assert np.mean(rates) <= alpha + (_S + 0.5 / p) / (0.5 * T) + 0.005


def test_adaptive_class_ipw_matches_simulate():
    rng = np.random.default_rng(3)
    T, p = 2000, 0.2
    conf = rng.uniform(0.3, 1.0, T)
    wrong = rng.random(T) < 0.3
    fb = rng.random(T) < p
    ad = AdaptiveCutoff(target=0.05, start=0.9, feedback_rate=p)
    acts = []
    for c, w, f in zip(conf, wrong, fb):
        a = ad.should_act(c)
        acts.append(a)
        ad.update(misfire=a and w, labeled=f)
    ref, _ = _sim(conf, wrong, 0.05, 0.9, 0.5, fb, rate=p)
    assert np.array_equal(np.array(acts), ref)


def test_conformal_cutoff_rest_option():
    rng = np.random.default_rng(1)
    people = []
    for _ in range(30):
        y = rng.integers(0, 4, 500)
        pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 4, 500))
        people.append(dict(pred=pred, conf=rng.uniform(0.25, 1, 500), y=y))
    t_all = ConformalCutoff(target=0.1).fit(people).threshold_
    t_rest = ConformalCutoff(target=0.1, rest=0).fit(people).threshold_
    assert t_rest <= t_all          # fewer decisions count as misfires
