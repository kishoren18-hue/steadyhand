"""Act on what's certain: rejection on motions instead of gestures (3DC).

Three 3DC gestures close the hand: power grip, chuck grip and pinch grip.
A standard controller acts only if one gesture is confident. When the model is
torn between grips, it holds still even though every option closes the hand.

The motion-level controller sums class probabilities within each motion
(close = power + chuck + pinch; every other gesture is its own motion), acts
if the top motion's probability clears the cutoff, and performs that motion.
Both controllers are scored on the same motion-level misfire (the hand made a
different motion than the user intended) and both cutoffs come from the same
conformal risk control across people, so the comparison is like for like.

Setups
  shared   leave one person out; calibration people scored by cross-fitted
           models (the hard case: 33% gesture accuracy)
  own      each person's own model (first half of their calibration session);
           calibration = other people's held-out calibration recordings

Usage
  python experiments/groups.py --data data/3dc
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import build_features
from common import Model
from steadyhand.risk import misfire_curve, candidate_thresholds
from steadyhand.cutoffs import crc_threshold

TARGETS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
# 3DC labels: 0 rest, 1 radial, 2 flexion, 3 ulnar, 4 extension, 5 supination,
# 6 pronation, 7 power grip, 8 open hand, 9 chuck grip, 10 pinch grip
MOTION = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 7])     # grips -> motion 7 (close)
GRIPS = [7, 9, 10]


def score(P, y):
    """Per-gesture and per-motion (prediction, confidence) from probabilities."""
    g_pred, g_conf = P.argmax(1), P.max(1)
    M = np.zeros((len(P), MOTION.max() + 1))
    np.add.at(M.T, MOTION, P.T)
    m_pred, m_conf = M.argmax(1), M.max(1)
    ym = MOTION[y]
    # misfire: a movement (not rest, motion 0) other than the intended motion
    return dict(gesture=dict(conf=g_conf, wrong=(MOTION[g_pred] != ym) & (g_pred != 0), pred=g_pred),
                motion=dict(conf=m_conf, wrong=(m_pred != ym) & (m_pred != 0), pred=m_pred), y=y)


def run(setup, X, y, subj, series, seg):
    people = np.unique(subj)
    scored = {}
    if setup == "shared":
        rng = np.random.default_rng(0)
        for p in people:                       # test scores
            tr, te = subj != p, (subj == p) & (series == 2)
            scored[int(p)] = {"test": score(Model(X[tr], y[tr]).proba(X[te]), y[te])}
        cal = {}
        for p in people:                       # cross-fitted calibration scores
            others = people[people != p]
            groups = np.array_split(rng.permutation(others), 5)
            cal[int(p)] = []
            for g in groups:
                tr = np.isin(subj, others) & ~np.isin(subj, g)
                m = Model(X[tr], y[tr])
                for j in g:
                    k = (subj == j) & (series == 2)
                    cal[int(p)].append(score(m.proba(X[k]), y[k]))
    else:
        for p in people:
            s1 = (subj == p) & (series == 1)
            segs = np.unique(seg[s1])
            first = np.isin(seg, segs[: len(segs) // 2]) & s1
            m = Model(X[first], y[first])
            c, te = s1 & ~first, (subj == p) & (series == 2)
            scored[int(p)] = {"cal": score(m.proba(X[c]), y[c]),
                              "test": score(m.proba(X[te]), y[te])}
        cal = {int(p): [scored[int(j)]["cal"] for j in people if j != p] for p in people}

    rows = []
    for p in people:
        p = int(p)
        t_ = scored[p]["test"]
        for level in ("gesture", "motion"):
            grid = candidate_thresholds(np.concatenate([c[level]["conf"] for c in cal[p]]))
            curves = [misfire_curve(c[level]["conf"], c[level]["wrong"], grid) for c in cal[p]]
            for al in TARGETS:
                t = crc_threshold(grid, curves, al)
                d = t_[level]
                act = d["conf"] > t
                closing = act & np.isin(t_["y"], GRIPS) & (MOTION[d["pred"]] == 7) if level == "gesture" \
                    else act & np.isin(t_["y"], GRIPS) & (d["pred"] == 7)
                rows.append(dict(setup=setup, person=p, controller=level, alpha=al, t=t,
                                 misfire=float((act & d["wrong"]).mean()),
                                 active=float((act & (MOTION[d["pred"]] != 0 if level == "gesture" else d["pred"] != 0)).mean()),
                                 grip_moves=float(closing.sum() / max(np.isin(t_["y"], GRIPS).sum(), 1))))
        print(f"{setup} person {p}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    X, y, subj, series, seg = build_features("3dc", a.data, Path(a.out) / "features_3dc.npz")
    rows = run("own", X, y, subj, series, seg) + run("shared", X, y, subj, series, seg)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "groups_3dc.csv", index=False)
    s = (df.groupby(["setup", "controller", "alpha"])
           .agg(mean_misfire=("misfire", "mean"), worst=("misfire", "max"),
                active=("active", "mean"), grip_moves=("grip_moves", "mean")).reset_index())
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
