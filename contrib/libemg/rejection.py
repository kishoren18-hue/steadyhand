"""Rejection thresholds chosen from a target misfire rate.

Proposed addition to LibEMG (libemg/rejection.py). Pure numpy, no new
dependencies. Works with EMGClassifier.add_rejection() unchanged.

A *misfire* is a decision that moves the device in a way the user did not
intend: the output is not rejected, is not the rest class, and differs from
the intended class.

    conformal_rejection_threshold   pick a threshold so a new user's expected
                                    misfire rate is <= target (conformal risk
                                    control, people as the exchangeable unit)
    AdaptiveRejection               update the threshold online from feedback
                                    so each user's misfire rate is <= target
                                    plus O(1/T), on any data sequence

References
    Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal risk control. ICLR 2024.
    Gibbs, Candes. Adaptive conformal inference under distribution shift. NeurIPS 2021.
"""
import numpy as np

S_MAX = 6.0   # confidences above 1 - 1e-6 are treated as 1 - 1e-6


def misfires(predictions, labels, rest_class=None):
    """Bool per decision: would acting on this prediction be a misfire?"""
    p, y = np.asarray(predictions), np.asarray(labels)
    bad = p != y
    return bad if rest_class is None else bad & (p != rest_class)


def misfire_curve(probabilities, misfire, thresholds):
    """Misfire rate at each threshold (act if probability > threshold)."""
    bad = np.sort(np.asarray(probabilities, float)[np.asarray(misfire, bool)])
    n = len(probabilities)
    return (len(bad) - np.searchsorted(bad, thresholds, side="right")) / n


def conformal_rejection_threshold(people, target=0.05, rest_class=None):
    """Smallest threshold t with (n * mean_j R_j(t) + 1) / (n + 1) <= target.

    people: list of (predictions, probabilities, labels), one per calibration
            person, each scored by a model that never trained on that person
            (e.g. leave-one-subject-out or cross-fitting).
    Guarantee: if a new user is exchangeable with the calibration people,
    their expected misfire rate at t is <= target. Needs target > 1/(n+1).
    Returns 1.0 (never act) if no threshold meets the target.
    """
    confs = np.concatenate([np.asarray(c, float) for _, c, _ in people])
    grid = np.unique(np.r_[0.0, confs, 1.0])
    curves = np.array([misfire_curve(c, misfires(p, y, rest_class), grid)
                       for p, c, y in people])
    n = len(curves)
    ok = (n * curves.mean(0) + 1.0) / (n + 1) <= target
    return float(grid[np.argmax(ok)]) if ok.any() else 1.0


def _score(prob):
    return -np.log10(1 - np.clip(prob, 0.0, 1 - 10 ** (-S_MAX)))


class AdaptiveRejection:
    """Online rejection threshold with a per-user misfire guarantee.

    Each decision: act if probability > self.threshold. When feedback for a
    decision arrives (e.g. during a cued task), call update().

        rej = AdaptiveRejection(target=0.05, start=0.9)
        act = probability > rej.threshold
        ...
        rej.update(misfire=act and prediction not in (rest, intended))

    Guarantee (feedback_rate=1): for every T and every sequence of decisions,
        misfire rate over T decisions <= target + (6 + eta - tau_1) / (eta T),
    where tau_1 = -log10(1 - start). With feedback on a random share p of
    decisions, set feedback_rate=p and call update(..., labeled=False) on
    decisions without feedback; the bound then holds in expectation with
    eta/p in place of eta in the numerator.
    """

    def __init__(self, target=0.05, start=0.9, eta=0.5, feedback_rate=1.0):
        self.target, self.eta, self.p = target, eta, feedback_rate
        self.tau = self.tau_start = min(float(_score(start)), S_MAX)
        self.n = 0

    @property
    def threshold(self):
        return float(1 - 10 ** (-min(self.tau, S_MAX)))

    def update(self, misfire, labeled=True):
        m = float(bool(misfire) and labeled) / self.p
        self.tau = max(self.tau + self.eta * (m - self.target), 0.0)
        self.n += 1
        return self.threshold

    def bound(self):
        return self.target + (S_MAX + self.eta / self.p - self.tau_start) / (self.eta * max(self.n, 1))
