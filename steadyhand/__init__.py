"""SteadyHand: rejection cutoffs for myoelectric control that keep their promise.

    from steadyhand import ConformalCutoff, AdaptiveCutoff

ConformalCutoff  picks a LibEMG rejection threshold so a new user's expected
                 misfire rate is at most a target (conformal risk control,
                 people as the unit).
AdaptiveCutoff   updates the threshold online from feedback so each user's
                 long-run misfire rate is at most the target plus a vanishing
                 term, on any data sequence.
"""
from .risk import (harmful, misfire_curve, voted_misfire_curve, vote_predict,
                   candidate_thresholds)
from .cutoffs import ConformalCutoff, crc_threshold, personal_cutoff
from .adaptive import AdaptiveCutoff, simulate

__version__ = "0.3.0"
__author__ = "Kishore Nuthalapati"

__all__ = ["harmful", "ConformalCutoff", "AdaptiveCutoff", "crc_threshold", "personal_cutoff",
           "misfire_curve", "voted_misfire_curve", "vote_predict",
           "candidate_thresholds", "simulate"]
