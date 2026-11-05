"""An adaptive rejection cutoff with a per-user misfire guarantee.

The conformal cutoff holds on average across people. This module gives a
guarantee for *each user*, on *any* sequence of decisions, including
electrode shift, arm-position changes and fatigue, at the price of needing
feedback: after a decision, whether it was a misfire.

Rule (online quantile tracking, as in adaptive conformal inference,
Gibbs & Candes, NeurIPS 2021, and conformal PID control, Angelopoulos,
Candes & Tibshirani, NeurIPS 2023), applied to a rejection cutoff:

    score    s = -log10(1 - conf), capped at s_max      (0.9 -> 1, 0.99 -> 2)
    act      if s > tau
    update   tau <- max(tau + eta * (m - target), 0)    after feedback,
             where m = 1 if the decision acted and was wrong, else 0

Guarantee. If tau starts at tau_1 <= s_max, then for every T and every
sequence of decisions,

    (1/T) * sum_{k<=T} m_k  <=  target + (s_max + eta - tau_1) / (eta * T).

Proof. tau only rises after a misfire, which needs s > tau, so tau < s_max
before the rise and tau < s_max + eta after it; when tau >= s_max the
controller never acts and tau falls. Hence tau_k <= s_max + eta for all k.
Summing the update, eta * sum_k (m_k - target) = tau_{T+1} - tau_1 - c, where
c >= 0 is the total added by the floor at 0. So
sum_k (m_k - target) <= (s_max + eta - tau_1) / eta. Divide by T.

With partial feedback, update only on decisions whose label is known; the
bound then holds for those decisions.
"""
from dataclasses import dataclass, field
import numpy as np

S_MAX = 6.0          # confidences above 1 - 1e-6 are treated as 1 - 1e-6


def score(conf):
    """Confidence on a log scale: 0.9 -> 1, 0.99 -> 2, 0.999 -> 3."""
    c = np.clip(np.asarray(conf, float), 0.0, 1 - 10 ** (-S_MAX))
    return -np.log10(1 - c)


def to_score(t):
    """Convert a LibEMG confidence cutoff to the score scale."""
    return float(score(t))


def to_conf(tau):
    """Convert a score-scale cutoff back to a LibEMG confidence cutoff."""
    return float(1 - 10 ** (-min(tau, S_MAX)))


@dataclass
class AdaptiveCutoff:
    """Online rejection cutoff that keeps each user's misfire rate near `target`.

    Example (inside your own control loop)
    --------------------------------------
    >>> ad = AdaptiveCutoff(target=0.05, start=conformal_cutoff)
    >>> pred, conf = emg_classifier.run(features)        # LibEMG, no rejection
    >>> acts = ad.should_act(conf[0])
    >>> ...                                   # later, when the outcome is known
    >>> ad.update(acted=acts, correct=was_right)
    """
    target: float = 0.05
    start: float = 0.9        # initial cutoff, on LibEMG's confidence scale
    eta: float = 0.5          # step size on the log scale
    tau: float = field(default=None, init=False)
    tau_start: float = field(default=None, init=False)
    n_updates: int = field(default=0, init=False)
    n_misfires: int = field(default=0, init=False)

    def __post_init__(self):
        self.tau = self.tau_start = min(to_score(self.start), S_MAX)

    @property
    def threshold(self):
        """Current cutoff on LibEMG's confidence scale."""
        return to_conf(self.tau)

    def should_act(self, conf):
        return bool(score(conf) > self.tau)

    def update(self, acted, correct):
        m = float(acted and not correct)
        self.tau = max(self.tau + self.eta * (m - self.target), 0.0)
        self.n_updates += 1
        self.n_misfires += int(m)
        return self.threshold

    def bound(self, T=None):
        """Guaranteed upper bound on the misfire rate after T feedback updates."""
        T = self.n_updates if T is None else T
        return self.target + (S_MAX + self.eta - self.tau_start) / (self.eta * max(T, 1))


def simulate(conf, wrong, target, start, eta=0.5, feedback=None):
    """Run the adaptive cutoff over a recorded stream of decisions.

    conf, wrong: (n,) arrays in time order
    feedback:    (n,) bool, which decisions return a label (default: all)
    Returns (acted, tau_trace).
    """
    s = score(conf)
    n = len(s)
    feedback = np.ones(n, bool) if feedback is None else np.asarray(feedback, bool)
    tau = min(to_score(start), S_MAX)
    acted = np.zeros(n, bool)
    trace = np.empty(n)
    for k in range(n):
        trace[k] = tau
        a = s[k] > tau
        acted[k] = a
        if feedback[k]:
            tau = max(tau + eta * (float(a and wrong[k]) - target), 0.0)
    return acted, trace
