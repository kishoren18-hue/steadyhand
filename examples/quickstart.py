"""SteadyHand in five minutes, on the UCI gesture dataset.

    git clone https://github.com/UNNLobachevsky/EMG_data_for_gestures.git data/uci
    pip install -e ".[experiments]"
    python examples/quickstart.py data/uci

1. Train a LibEMG classifier on 35 people.
2. Pick a rejection threshold that keeps the 36th person's expected misfire
   rate at or below 5% (ConformalCutoff).
3. Replay the 36th person's recording through LibEMG with that threshold.
4. Replay it again with AdaptiveCutoff, which guarantees the target for this
   person given feedback on each decision.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from data import build_features
from common import Model, out_of_person_scores
from steadyhand import ConformalCutoff, simulate, harmful

REST = 0                                        # UCI class 0 is rest

data = sys.argv[1] if len(sys.argv) > 1 else "data/uci"
X, y, subj, series, seg = build_features("uci", data, ROOT / "results" / "features_uci.npz")
new_user = 22                                   # the hardest person in the dataset
train = subj != new_user

# 1. a LibEMG classifier trained on everyone else
model = Model(X[train], y[train])

# 2. calibrate: previous users scored by models that never saw them
people = out_of_person_scores(X[train], y[train], subj[train], series[train], seg[train])
cut = ConformalCutoff(target=0.05, rest=REST).fit(people)
print(f"Conformal threshold: {cut.threshold_:.4f}   (LibEMG default: 0.9000)")

# 3. replay the new user's recording with LibEMG's own rejection block
test = (subj == new_user) & (series == 2)
model.clf.add_rejection(threshold=cut.threshold_)
pred, conf = model.clf.run(model.scaler.transform(X[test]))
moving = (pred != -1) & (pred != REST)
print(f"Conformal: misfires {np.mean(moving & (pred != y[test])):.1%}, "
      f"hand moves on {moving.mean():.1%} of decisions")

model.clf.add_rejection(threshold=0.9)
pred9, _ = model.clf.run(model.scaler.transform(X[test]))
moving9 = (pred9 != -1) & (pred9 != REST)
print(f"LibEMG 0.9: misfires {np.mean(moving9 & (pred9 != y[test])):.1%}, "
      f"hand moves on {moving9.mean():.1%} of decisions")

# 4. adaptive: guaranteed for this person, given feedback after each decision
model.clf.rejection = False
raw_pred, raw_conf = model.clf.run(model.scaler.transform(X[test]))
wrong = harmful(raw_pred, y[test], REST)
acted_ad, _ = simulate(raw_conf, wrong, target=0.05, start=cut.threshold_)
print(f"Adaptive:  misfires {np.mean(acted_ad & wrong):.1%}, "
      f"hand moves on {np.mean(acted_ad & (raw_pred != REST)):.1%} of decisions")
