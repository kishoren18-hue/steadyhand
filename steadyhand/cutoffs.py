"""Choosing a rejection cutoff from a target misfire rate.

ConformalCutoff implements conformal risk control (Angelopoulos, Bates, Fisch,
Lei, Schuster, ICLR 2024) with *people* as the exchangeable unit. Given n
calibration people, each scored by a model that never trained on them, it picks

    t = smallest t with  (n * mean_j R_j(t) + B) / (n + 1) <= target

where R_j(t) is person j's misfire rate and B = 1 bounds it. If a new person
is exchangeable with the calibration people, their expected misfire rate is at
most `target`. The guarantee is on average over people, not for each person,
and needs target > B / (n + 1).

personal_cutoff gives the smallest cutoff meeting the target on one person's
own calibration recording (no guarantee). Taking max(conformal, personal) can
only lower that person's misfire rate, so the guarantee is kept.
"""
from dataclasses import dataclass, field
import numpy as np

from .risk import misfire_curve, candidate_thresholds, voted_misfire_curve, harmful


def crc_threshold(grid, curves, target, B=1.0):
    """Conformal risk control on precomputed per-person curves.

    grid:   increasing cutoffs
    curves: (n_people, len(grid)) misfire rate of each person at each cutoff

    Each curve is first replaced by its running maximum from the right, a
    non-increasing upper bound (needed after majority voting, where raising t
    can occasionally raise misfires). Returns 1.0 (never act) if no cutoff
    passes."""
    curves = np.maximum.accumulate(np.asarray(curves, float)[:, ::-1], axis=1)[:, ::-1]
    n = len(curves)
    ok = (n * curves.mean(0) + B) / (n + 1) <= target
    return float(grid[np.argmax(ok)]) if ok.any() else 1.0


def personal_cutoff(conf, wrong, target):
    """Smallest cutoff whose misfire rate on this recording is <= target."""
    ts = candidate_thresholds(conf)
    ok = misfire_curve(conf, wrong, ts) <= target
    return float(ts[np.argmax(ok)])


@dataclass
class ConformalCutoff:
    """Pick a LibEMG rejection threshold that keeps the expected misfire rate
    of a new user at or below `target`.

    Example
    -------
    >>> cut = ConformalCutoff(target=0.05).fit(people)   # list of dicts
    >>> classifier.add_rejection(threshold=cut.threshold_)

    Each person is a dict with arrays 'pred', 'conf', 'y' (and 'seg' when
    vote > 1), from a model that never trained on that person. Pass `rest`
    (the class index of "rest") so that holding still is never a misfire.
    """
    target: float = 0.05
    vote: int = 1                 # LibEMG majority-vote length; 1 = off
    n_classes: int | None = None  # needed when vote > 1
    rest: int | None = None       # class index of "rest"; its predictions never misfire
    B: float = 1.0
    threshold_: float = field(default=None, init=False)
    n_people_: int = field(default=0, init=False)

    def fit(self, people):
        self.n_people_ = len(people)
        if self.vote <= 1:
            grid = candidate_thresholds(np.concatenate([p["conf"] for p in people]))
            curves = [misfire_curve(p["conf"], harmful(p["pred"], p["y"], self.rest), grid)
                      for p in people]
        else:
            if self.n_classes is None:
                raise ValueError("n_classes is required when vote > 1")
            grid = voting_grid(people)
            curves = [voted_misfire_curve(p["pred"], p["conf"], p["y"], p["seg"],
                                          self.vote, grid, self.n_classes, self.rest)
                      for p in people]
        self.threshold_ = crc_threshold(grid, curves, self.target, self.B)
        return self

    @property
    def floor(self):
        """Lowest reachable target with this many calibration people."""
        return self.B / (self.n_people_ + 1)

    def with_personal(self, conf, wrong):
        """The stricter of this cutoff and the user's own personal cutoff."""
        return max(self.threshold_, personal_cutoff(conf, wrong, self.target))


def voting_grid(people):
    """Cutoffs for voted curves: 0.1% quantiles of the calibration confidences
    plus fine steps near 1, where LDA cutoffs land."""
    c = np.concatenate([p["conf"] for p in people])
    q = np.quantile(c, np.linspace(0, 1, 1001))
    return np.unique(np.r_[0.0, q, 1 - np.logspace(-6, -1, 200), 0.9, 1.0])
