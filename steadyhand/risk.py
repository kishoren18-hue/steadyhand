"""Misfire curves: how often a rejection controller acts on a wrong gesture.

A *decision* is one classifier output (one window, typically every 50 ms).
LibEMG's rejection rule acts on a decision only if its confidence is above a
cutoff t; otherwise it outputs -1 ("hold still"). A *misfire* is a decision
that acts and is wrong. The misfire rate at t is misfires / all decisions.

All functions here are exact and vectorized.
"""
import numpy as np


def misfire_curve(conf, wrong, thresholds):
    """Misfire rate at each threshold: mean of (conf > t and wrong).

    conf:       (n,) top-class confidence of each decision
    wrong:      (n,) bool, prediction != true label
    thresholds: (m,) cutoffs to evaluate
    """
    conf = np.asarray(conf, float)
    bad = np.sort(conf[np.asarray(wrong, bool)])
    acting_and_wrong = len(bad) - np.searchsorted(bad, thresholds, side="right")
    return acting_and_wrong / len(conf)


def candidate_thresholds(conf):
    """Every cutoff at which some misfire curve can change, plus 0 and 1."""
    return np.unique(np.r_[0.0, np.ravel(conf), 1.0])


# ---------------------------------------------------------------- majority voting
def _trailing(seg, k):
    """(n, k) indices of each decision's last k decisions in the same segment,
    oldest first; -1 where the segment has not started yet. LibEMG likewise
    votes over a shorter group at the start of a stream."""
    seg = np.asarray(seg)
    n = len(seg)
    idx = np.arange(n)[:, None] - np.arange(k)[None, ::-1]
    ok = idx >= 0
    idx = np.where(ok, idx, 0)
    ok &= seg[idx] == seg[:, None]
    return np.where(ok, idx, -1)


def _mode_with_reject(P, present, n_classes):
    """Mode of each row of P (values -1..K-1), ignoring absent entries.
    Ties go to the smallest value, as scipy.stats.mode does inside LibEMG,
    so a tie with 'hold still' (-1) means hold still."""
    counts = np.stack([((P == c) & present).sum(1) for c in range(-1, n_classes)], 1)
    return np.argmax(counts, 1) - 1


def vote_predict(pred, conf, seg, k, t, n_classes):
    """LibEMG order: reject (conf <= t -> -1), then majority vote over the
    last k decisions in the same segment. k <= 1 means no voting."""
    pred, conf = np.asarray(pred), np.asarray(conf)
    if k <= 1:
        return np.where(conf > t, pred, -1)
    idx = _trailing(seg, k)
    P = np.where(conf[np.maximum(idx, 0)] > t, pred[np.maximum(idx, 0)], -1)
    return _mode_with_reject(P, idx >= 0, n_classes)


def voted_misfire_curve(pred, conf, y, seg, k, grid, n_classes):
    """Exact misfire rate after rejection + majority voting, at every cutoff in grid.

    A decision's vote looks at its last k (pred, conf) pairs. As t rises those
    pairs turn into -1 one at a time, least confident first, so its output can
    only change at its own k confidence values. We compute the vote in each of
    the k + 1 intervals and accumulate the changes on the grid: O(n k) votes
    instead of O(n k |grid|)."""
    pred, conf, y = np.asarray(pred), np.asarray(conf, float), np.asarray(y)
    idx = _trailing(seg, k)
    present = idx >= 0
    C = np.where(present, conf[np.maximum(idx, 0)], np.inf)   # absent: never rejected
    order = np.argsort(C, 1, kind="stable")
    rank = np.argsort(order, 1, kind="stable")
    bounds = np.take_along_axis(C, order, 1)
    Pk = pred[np.maximum(idx, 0)]
    deltas = np.zeros(len(grid))
    base, m_prev = 0.0, None
    for j in range(k + 1):
        v = _mode_with_reject(np.where(rank < j, -1, Pk), present, n_classes)
        m = ((v != -1) & (v != y)).astype(float)
        if j == 0:
            base = m.sum()
        else:
            b = bounds[:, j - 1]
            pos = np.searchsorted(grid, b, side="left")     # first grid t >= b
            keep = np.isfinite(b) & (pos < len(grid))
            np.add.at(deltas, pos[keep], (m - m_prev)[keep])
        m_prev = m
    return (base + np.cumsum(deltas)) / len(pred)
