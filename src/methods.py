"""Ways of choosing the rejection threshold for a LibEMG classifier.

LibEMG's rejection rule (EMGClassifier.add_rejection) acts on a window only when
the classifier's top probability is above a threshold t; otherwise it outputs
-1 ("do nothing"). Everything here keeps that exact rule and changes only how
t is picked.

A "misfire" is a window where the controller acts AND the gesture is wrong.
Misfire rate = misfires / all windows. It can only go down as t goes up.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from libemg.emg_predictor import EMGClassifier


# ---------------------------------------------------------------- model
class Model:
    """A LibEMG EMGClassifier plus the feature scaler fit on its training data."""

    def __init__(self, X, y, kind="LDA"):
        self.scaler = StandardScaler().fit(X)
        # 50 trees (sklearn default is 100) to keep the 216-fit robustness run short
        params = ({"n_estimators": 50, "n_jobs": -1, "random_state": 0}
                  if kind == "RF" else None)
        self.clf = EMGClassifier(kind, model_parameters=params)
        self.clf.fit(feature_dictionary={
            "training_features": self.scaler.transform(X),
            "training_labels": y})

    def predict(self, X):
        """Return (predicted gesture, top probability) with no rejection."""
        return self.clf.run(self.scaler.transform(X))


# ---------------------------------------------------------------- risk
def misfire_curve(conf, wrong, thresholds):
    """Misfire rate at each threshold: mean of (conf > t and wrong)."""
    bad = np.sort(conf[wrong])
    acting_and_wrong = len(bad) - np.searchsorted(bad, thresholds, side="right")
    return acting_and_wrong / len(conf)


def candidate_thresholds(conf):
    return np.unique(np.r_[0.0, conf, 1.0])


# ---------------------------------------------------------------- methods
def threshold_fixed(t=0.9):
    """LibEMG's default: a fixed confidence cutoff."""
    return t


def threshold_empirical(conf, wrong, alpha):
    """Smallest t whose observed misfire rate is <= alpha (no correction)."""
    ts = candidate_thresholds(conf)
    ok = misfire_curve(conf, wrong, ts) <= alpha
    return ts[np.argmax(ok)]


def threshold_conformal(per_user, alpha, B=1.0):
    """Conformal risk control with people as the exchangeable unit.

    per_user: list of (conf, wrong) pairs, one per calibration person, where
    conf came from a model that never saw that person.

    Picks the smallest t with (n * mean_j R_j(t) + B) / (n + 1) <= alpha,
    where R_j(t) is person j's misfire rate (Angelopoulos et al., "Conformal
    Risk Control", ICLR 2024). If a new person is exchangeable with the
    calibration people, their expected misfire rate is <= alpha. It is a
    guarantee on average over people, not for every single person.
    Returns 1.0 (never act) if alpha is too small for n people.
    """
    n = len(per_user)
    ts = candidate_thresholds(np.concatenate([c for c, _ in per_user]))
    mean_risk = np.mean([misfire_curve(c, w, ts) for c, w in per_user], axis=0)
    ok = (n * mean_risk + B) / (n + 1) <= alpha
    return ts[np.argmax(ok)] if ok.any() else 1.0


def out_of_person_scores(X, y, subj, kind="LDA", folds=5, seed=0):
    """Cross-fit: score every training person with a model that never saw them."""
    people = np.unique(subj)
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(people), folds)
    per_user = []
    for g in groups:
        held = np.isin(subj, g)
        m = Model(X[~held], y[~held], kind)
        pred, conf = m.predict(X[held])
        for p in g:
            k = subj[held] == p
            per_user.append((conf[k], pred[k] != y[held][k]))
    return per_user
