"""Ways of choosing the rejection threshold for a LibEMG classifier.

LibEMG's rejection rule (EMGClassifier.add_rejection) acts on a window only when
the classifier's top probability is above a threshold t; otherwise it outputs
-1 ("do nothing"). Everything here keeps that exact rule and changes only how
t is picked.

A "misfire" is a decision where the controller acts AND the gesture is wrong.
Misfire rate = misfires / all decisions.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from libemg.emg_predictor import EMGClassifier


# ---------------------------------------------------------------- model
class Model:
    """A LibEMG EMGClassifier plus the feature scaler fit on its training data."""

    def __init__(self, X, y, kind="LDA"):
        self.scaler = StandardScaler().fit(X)
        # 50 trees (sklearn default is 100) to keep the robustness run short
        params = ({"n_estimators": 50, "n_jobs": -1, "random_state": 0}
                  if kind == "RF" else None)
        self.clf = EMGClassifier(kind, model_parameters=params)
        self.clf.fit(feature_dictionary={
            "training_features": self.scaler.transform(X),
            "training_labels": y})

    def predict(self, X):
        """Return (predicted gesture, top probability) with no rejection."""
        return self.clf.run(self.scaler.transform(X))


# ---------------------------------------------------------------- per-window risk
def misfire_curve(conf, wrong, thresholds):
    """Misfire rate at each threshold: mean of (conf > t and wrong). Exact."""
    bad = np.sort(conf[wrong])
    acting_and_wrong = len(bad) - np.searchsorted(bad, thresholds, side="right")
    return acting_and_wrong / len(conf)


def candidate_thresholds(conf):
    return np.unique(np.r_[0.0, conf, 1.0])


# ---------------------------------------------------------------- threshold rules
def threshold_fixed(t=0.9):
    """LibEMG's default: a fixed confidence cutoff."""
    return t


def threshold_empirical(conf, wrong, alpha):
    """Smallest t whose observed per-window misfire rate is <= alpha."""
    ts = candidate_thresholds(conf)
    ok = misfire_curve(conf, wrong, ts) <= alpha
    return ts[np.argmax(ok)]


def crc_threshold(grid, curves, alpha, B=1.0):
    """Conformal risk control with people as the exchangeable unit.

    grid:   increasing thresholds
    curves: (n_people, len(grid)) misfire rate of each calibration person at
            each threshold, scored by a model that never trained on them

    Each person's curve is first made non-increasing (running max from the
    right), which can only raise it, so the guarantee still covers the true
    rate. Then pick the smallest t with (n * mean_j R_j(t) + B) / (n + 1) <= alpha
    (Angelopoulos et al., "Conformal Risk Control", ICLR 2024). If a new person
    is exchangeable with the calibration people, their expected misfire rate
    is <= alpha. Returns 1.0 (never act) if alpha is too small for n people.
    """
    curves = np.maximum.accumulate(np.asarray(curves)[:, ::-1], axis=1)[:, ::-1]
    n = len(curves)
    ok = (n * curves.mean(0) + B) / (n + 1) <= alpha
    return grid[np.argmax(ok)] if ok.any() else 1.0


def threshold_conformal(per_user, alpha, B=1.0):
    """Per-window conformal threshold from out-of-person scores (exact grid)."""
    grid = candidate_thresholds(np.concatenate([u["conf"] for u in per_user]))
    curves = [misfire_curve(u["conf"], u["pred"] != u["y"], grid) for u in per_user]
    return crc_threshold(grid, curves, alpha, B)


def out_of_person_scores(X, y, subj, series, segment, kind="LDA", folds=5, seed=0):
    """Cross-fit: score every training person with a model that never saw them.
    Returns one dict per person with their series-2 predictions, the same
    windows a new user is tested on."""
    people = np.unique(subj)
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(people), folds)
    per_user = []
    for g in groups:
        held = np.isin(subj, g)
        m = Model(X[~held], y[~held], kind)
        for p in g:
            k = (subj == p) & (series == 2)
            pred, conf = m.predict(X[k])
            per_user.append(dict(person=int(p), pred=pred, conf=conf, y=y[k],
                                 seg=segment[k]))
    return per_user


# ---------------------------------------------------------------- majority voting
def _trailing(seg, k):
    """Index matrix (n, k) of each window's last k windows in the same segment
    (oldest first). Missing entries at the start of a segment are -1; LibEMG
    likewise uses a shorter group at the start of a stream."""
    n = len(seg)
    idx = np.arange(n)[:, None] - np.arange(k)[None, ::-1]
    ok = idx >= 0
    idx = np.where(ok, idx, 0)
    ok &= seg[idx] == seg[:, None]
    return np.where(ok, idx, -1)


def _mode_with_reject(P, present, n_classes):
    """Mode of each row of P (values -1..K-1), ignoring absent entries.
    Ties go to the smallest value, as scipy.stats.mode does inside LibEMG,
    so a tie with 'do nothing' (-1) means do nothing."""
    counts = np.stack([((P == c) & present).sum(1) for c in range(-1, n_classes)], 1)
    return np.argmax(counts, 1) - 1


def vote_predict(pred, conf, seg, k, t, n_classes):
    """LibEMG order: reject (conf <= t -> -1), then majority vote over the
    last k decisions in the same segment. k <= 1 means no voting."""
    if k <= 1:
        return np.where(conf > t, pred, -1)
    idx = _trailing(seg, k)
    P = np.where(conf[idx] > t, pred[idx], -1)
    return _mode_with_reject(P, idx >= 0, n_classes)


def voted_misfire_curve(pred, conf, y, seg, k, grid, n_classes):
    """Exact misfire rate after rejection + voting at every threshold in grid.

    Each window's vote looks at its last k (pred, conf) pairs. As t rises those
    pairs turn into -1 one at a time, least confident first, so a window's
    output can only change at its own k confidence values. We compute the vote
    in each of the k+1 intervals and add the changes onto the grid."""
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
            pos = np.searchsorted(grid, b, side="left")      # first grid t >= b
            keep = np.isfinite(b) & (pos < len(grid))
            np.add.at(deltas, pos[keep], (m - m_prev)[keep])
        m_prev = m
    return (base + np.cumsum(deltas)) / len(pred)
