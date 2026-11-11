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

Label-efficient version (feedback on a random share p of decisions).
Updating only on labeled decisions makes tau move p times slower, which is
too slow after a large change such as an electrode shift. Instead, update on
*every* decision with an inverse-probability-weighted misfire:

    tau <- max(tau + eta * (F * m / p - target), 0)

where F = 1 if this decision got feedback (chosen at random with probability
p, independently of everything else). The same argument with the jump eta/p
gives, for every T and every sequence,

    (1/T) * sum_k F_k m_k / p  <=  target + (s_max + eta/p - tau_1) / (eta * T),

and because E[F_k m_k / p | past] = m_k, the expected misfire rate over *all*
decisions, labeled or not, obeys the same bound (for any data sequence fixed
in advance, or one that depends only on earlier feedback draws, such as a
model retrained on the labels received so far). p = 1 recovers the
full-feedback rule.
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
    >>> ad.update(misfire=acts and pred[0] != REST and pred[0] != intended)

    With feedback on only a random share p of decisions, set `feedback_rate=p`
    and call `update(misfire=..., labeled=...)` on every decision; unlabeled
    decisions pass labeled=False.
    """
    target: float = 0.05
    start: float = 0.9        # initial cutoff, on LibEMG's confidence scale
    eta: float = 0.5          # step size on the log scale
    feedback_rate: float = 1.0
    tau: float = field(default=None, init=False)
    tau_start: float = field(default=None, init=False)
    n_updates: int = field(default=0, init=False)
    n_misfires: int = field(default=0, init=False)

    def __post_init__(self):
        if not 0 < self.feedback_rate <= 1:
            raise ValueError("feedback_rate must be in (0, 1]")
        self.tau = self.tau_start = min(to_score(self.start), S_MAX)

    @property
    def threshold(self):
        """Current cutoff on LibEMG's confidence scale."""
        return to_conf(self.tau)

    def should_act(self, conf):
        return bool(score(conf) > self.tau)

    def update(self, misfire=None, labeled=True, *, acted=None, correct=None):
        """Record one decision. `misfire`: the device acted and moved wrongly
        (a confident "rest" prediction is never a misfire). The older call
        update(acted=..., correct=...) is still accepted."""
        if misfire is None:
            misfire = bool(acted) and not bool(correct)
        m = float(bool(misfire) and labeled) / self.feedback_rate
        self.tau = max(self.tau + self.eta * (m - self.target), 0.0)
        self.n_updates += 1
        self.n_misfires += int(bool(misfire) and labeled)
        return self.threshold

    def bound(self, T=None):
        """Upper bound on the (expected, if feedback_rate < 1) misfire rate
        after T decisions."""
        T = self.n_updates if T is None else T
        jump = self.eta / self.feedback_rate
        return self.target + (S_MAX + jump - self.tau_start) / (self.eta * max(T, 1))


def simulate(conf, wrong, target, start, eta=0.5, feedback=None, rate=None):
    """Run the adaptive cutoff over a recorded stream of decisions.

    conf, wrong: (n,) arrays in time order; wrong = acting would misfire
    feedback:    (n,) bool, which decisions return a label (default: all)
    rate:        None  -> update only on labeled decisions (the plain rule;
                          the bound then covers labeled decisions only)
                 p     -> label-efficient rule: update every decision with the
                          misfire weighted by 1/p on labeled decisions
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
        m = float(a and wrong[k])
        if rate is None:
            if feedback[k]:
                tau = max(tau + eta * (m - target), 0.0)
        else:
            tau = max(tau + eta * (m * feedback[k] / rate - target), 0.0)
    return acted, trace
