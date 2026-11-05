"""Shared experiment helpers: the LibEMG classifier wrapper and cross-fitting."""
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from libemg.emg_predictor import EMGClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the steadyhand package


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
        """(predicted gesture, top probability), no rejection."""
        return self.clf.run(self.scaler.transform(X))

    def proba(self, X):
        """Full class probabilities, columns in label order 0..K-1."""
        return self.clf.model.predict_proba(self.scaler.transform(X))


def out_of_person_scores(X, y, subj, series, segment, kind="LDA", folds=5, seed=0):
    """Cross-fit: score every person's series-2 windows with a model that never
    saw them. One dict per person."""
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
