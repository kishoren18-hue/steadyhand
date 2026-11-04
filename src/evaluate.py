"""Leave-one-person-out evaluation of rejection thresholds.

For each of the 36 people:
  1. Train a LibEMG classifier on the other 35.
  2. Pick a threshold with each method.
  3. Test on the held-out person's second recording series.

Methods
  fixed      LibEMG default, act only if confidence > 0.9
  tuned      threshold tuned on the training people's own data (in-sample)
  conformal  threshold from conformal risk control on out-of-person scores
  personal   threshold tuned on the new person's first series (a short
             guided calibration, like commercial devices do); no guarantee
  conformal+personal
             the stricter (higher) of conformal and personal; keeps the
             conformal guarantee and adds a per-person safety check

Usage:  python src/evaluate.py --data /path/to/EMG_data_for_gestures
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import build_features, GESTURES
from methods import (Model, misfire_curve, threshold_fixed, threshold_empirical,
                     threshold_conformal, out_of_person_scores)

ALPHAS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


def score(conf, wrong, t):
    act = conf > t
    return dict(misfire=float(np.mean(act & wrong)),
                active=float(np.mean(act)),
                acc_when_active=float(1 - wrong[act].mean()) if act.any() else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="LDA")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    X, y, subj, series = build_features(args.data, out / "features.npz")
    rows, base_acc, demo = [], [], {}

    for p in np.unique(subj):
        train = subj != p
        test2 = (subj == p) & (series == 2)
        test1 = (subj == p) & (series == 1)

        model = Model(X[train], y[train], args.model)
        pred, conf = model.predict(X[test2])
        wrong = pred != y[test2]
        base_acc.append(1 - wrong.mean())

        # in-sample scores on training people (what you'd see if you tuned
        # the threshold on the data you trained on)
        pred_in, conf_in = model.predict(X[train])
        wrong_in = pred_in != y[train]

        # out-of-person scores for the 35 training people
        per_user = out_of_person_scores(X[train], y[train], subj[train], args.model)

        # new person's first series, for the personal method
        pred1, conf1 = model.predict(X[test1])
        wrong1 = pred1 != y[test1]

        t = threshold_fixed(0.9)
        rows.append(dict(person=int(p), method="fixed", alpha=np.nan, t=t,
                         **score(conf, wrong, t)))
        for a in ALPHAS:
            t_conf = threshold_conformal(per_user, a)
            t_pers = threshold_empirical(conf1, wrong1, a)
            for name, t in [
                ("tuned", threshold_empirical(conf_in, wrong_in, a)),
                ("conformal", t_conf),
                ("personal", t_pers),
                # stricter of the two: raising t can only lower misfires, so
                # the conformal average guarantee still holds
                ("conformal+personal", max(t_conf, t_pers)),
            ]:
                rows.append(dict(person=int(p), method=name, alpha=a, t=float(t),
                                 **score(conf, wrong, t)))
                demo.setdefault(int(p), {}).setdefault(name, {})[str(a)] = float(t)
        demo[int(p)].update(fixed=0.9, y=y[test2].tolist(), pred=pred.tolist(),
                            conf=np.round(conf, 5).tolist(),
                            accuracy=float(1 - wrong.mean()))
        print(f"person {p:2d}: accuracy {base_acc[-1]:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"per_person_{args.model}.csv", index=False)

    df["over_target"] = df["misfire"] > df["alpha"]
    summary = (df.groupby(["method", "alpha"], dropna=False)
                 .agg(mean_misfire=("misfire", "mean"),
                      median_misfire=("misfire", "median"),
                      worst_misfire=("misfire", "max"),
                      share_over_target=("over_target", "mean"),
                      mean_active=("active", "mean"),
                      acc_when_active=("acc_when_active", "mean"))
                 .reset_index())
    summary.to_csv(out / f"summary_{args.model}.csv", index=False)
    print(f"\nNo rejection: mean accuracy on unseen people {np.mean(base_acc):.3f}")
    print(summary.round(3).to_string(index=False))

    with open(out / f"demo_{args.model}.json", "w") as f:
        json.dump(dict(gestures=list(GESTURES.values()), alphas=ALPHAS,
                       increment_ms=50, people=demo,
                       base_accuracy=float(np.mean(base_acc))), f)


if __name__ == "__main__":
    main()
