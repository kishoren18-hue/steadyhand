"""Leave-one-person-out evaluation of rejection thresholds.

For each person in the dataset:
  1. Train a LibEMG classifier on everyone else.
  2. Pick a threshold with each method.
  3. Score the held-out person's test session (series 2).

Methods without voting (one decision every 50 ms)
  fixed               LibEMG default: act only if confidence > 0.9
  tuned               threshold that meets the target on the training people's
                      own data (in-sample)
  personal            threshold that meets the target on the new person's
                      calibration session (series 1); no guarantee
  conformal           conformal risk control on out-of-person scores
  conformal+personal  the higher of conformal and personal (SteadyHand)

With LibEMG majority voting over the last 5 decisions (--vote 5)
  fixed               0.9, then vote
  conformal-naive     the per-window conformal threshold, then vote
  conformal-vote      conformal risk control on the voted output
  conformal-vote+personal   the higher of conformal-vote and a personal
                      threshold computed on the voted calibration session

Personalized model (--personalize)
  pers-conformal      model trained on everyone else + the new person's
                      calibration session; conformal threshold from people
                      scored the same way (their own calibration session in
                      training, their test session held out)
  own-conformal       user-specific model trained only on the new person's
                      calibration session; conformal threshold from other
                      people's user-specific models
  pers-none, own-none the same models with no rejection (accuracy reference)

Usage
  python src/evaluate.py --dataset uci --data data/uci
  python src/evaluate.py --dataset 3dc --data data/3dc --vote 5 --personalize
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import build_features, GESTURES
from methods import (Model, misfire_curve, threshold_fixed, threshold_empirical,
                     threshold_conformal, crc_threshold, out_of_person_scores,
                     vote_predict, voted_misfire_curve, candidate_thresholds)

ALPHAS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


def score(out, y):
    """Metrics for a stream of outputs (-1 = hold still)."""
    act = out != -1
    wrong = act & (out != y)
    return dict(misfire=float(wrong.mean()), active=float(act.mean()),
                acc_when_active=float(1 - wrong[act].mean() / 1) if act.any() else np.nan)


def vote_grid(per_user):
    """Thresholds to evaluate voted curves on: every 0.1% quantile of the
    calibration confidences plus fine steps near 1, where LDA's cutoffs land."""
    c = np.concatenate([u["conf"] for u in per_user])
    q = np.quantile(c, np.linspace(0, 1, 1001))
    return np.unique(np.r_[0.0, q, 1 - np.logspace(-6, -1, 200), 0.9, 1.0])


def personalized_scores(X, y, subj, series, segment, cal_people, kind):
    """For each calibration person j: train on the other calibration people plus
    j's own series 1, score j's series 2. Matches how the new user is served."""
    out = []
    for j in cal_people:
        tr = np.isin(subj, cal_people) & ((subj != j) | (series == 1))
        m = Model(X[tr], y[tr], kind)
        k = (subj == j) & (series == 2)
        pred, conf = m.predict(X[k])
        out.append(dict(person=int(j), pred=pred, conf=conf, y=y[k], seg=segment[k]))
    return out


def own_model_scores(X, y, subj, series, segment, cal_people, kind):
    """For each calibration person j: a model trained only on j's series 1,
    scored on j's series 2 (the same setup the new user gets)."""
    out = []
    for j in cal_people:
        tr = (subj == j) & (series == 1)
        k = (subj == j) & (series == 2)
        pred, conf = Model(X[tr], y[tr], kind).predict(X[k])
        out.append(dict(person=int(j), pred=pred, conf=conf, y=y[k], seg=segment[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["uci", "3dc"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="LDA")
    ap.add_argument("--vote", type=int, default=0, help="majority vote length (0 = off)")
    ap.add_argument("--personalize", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"{a.dataset}_{a.model}"
    K = len(GESTURES[a.dataset])

    X, y, subj, series, seg = build_features(a.dataset, a.data, out / f"features_{a.dataset}.npz")
    people = np.unique(subj)
    rows, info, demo = [], [], {}
    t0 = time.time()

    def add(p, method, alpha, t, o, yy, vote=1):
        rows.append(dict(person=int(p), vote=vote, method=method, alpha=alpha,
                         t=float(t), **score(o, yy)))

    for p in people:
        train = subj != p
        te, ca = (subj == p) & (series == 2), (subj == p) & (series == 1)
        model = Model(X[train], y[train], a.model)
        pred, conf = model.predict(X[te])
        pred1, conf1 = model.predict(X[ca])
        pred_in, conf_in = model.predict(X[train])
        per_user = out_of_person_scores(X[train], y[train], subj[train],
                                        series[train], seg[train], a.model)
        yt, y1 = y[te], y[ca]
        base_acc = float((pred == yt).mean())

        # ---- no voting
        add(p, "fixed", np.nan, 0.9, vote_predict(pred, conf, seg[te], 1, 0.9, K), yt)
        t_conf_by_alpha = {}
        for al in ALPHAS:
            t_tun = threshold_empirical(conf_in, pred_in != y[train], al)
            t_per = threshold_empirical(conf1, pred1 != y1, al)
            t_con = threshold_conformal(per_user, al)
            t_conf_by_alpha[al] = t_con
            for name, t in [("tuned", t_tun), ("personal", t_per), ("conformal", t_con),
                            ("conformal+personal", max(t_con, t_per))]:
                add(p, name, al, t, vote_predict(pred, conf, seg[te], 1, t, K), yt)
                demo.setdefault(int(p), {}).setdefault(name, {})[str(al)] = float(t)
            info.append(dict(person=int(p), alpha=al, base_acc=base_acc,
                             cal_acc=float((pred1 == y1).mean()),
                             cal_misfire_at_conformal=float(misfire_curve(conf1, pred1 != y1, np.array([t_con]))[0]),
                             test_misfire_at_conformal=rows[-2]["misfire"]))
        demo[int(p)].update(y=yt.tolist(), pred=pred.tolist(),
                            conf=np.round(conf, 5).tolist(), accuracy=base_acc)

        # ---- majority voting
        if a.vote > 1:
            k = a.vote
            grid = vote_grid(per_user)
            curves = np.array([voted_misfire_curve(u["pred"], u["conf"], u["y"], u["seg"], k, grid, K)
                               for u in per_user])
            cal_curve = voted_misfire_curve(pred1, conf1, y1, seg[ca], k, grid, K)
            add(p, "fixed", np.nan, 0.9, vote_predict(pred, conf, seg[te], k, 0.9, K), yt, k)
            for al in ALPHAS:
                t_naive = t_conf_by_alpha[al]
                t_cv = crc_threshold(grid, curves, al)
                ok = cal_curve <= al
                t_pv = grid[np.argmax(ok)] if ok.any() else 1.0
                for name, t in [("conformal-naive", t_naive), ("conformal-vote", t_cv),
                                ("conformal-vote+personal", max(t_cv, t_pv))]:
                    add(p, name, al, t, vote_predict(pred, conf, seg[te], k, t, K), yt, k)

        # ---- personalized model
        if a.personalize:
            cal_people = people[people != p]
            pmodel = Model(X[train | ca], y[train | ca], a.model)
            ppred, pconf = pmodel.predict(X[te])
            pu = personalized_scores(X, y, subj, series, seg, cal_people, a.model)
            add(p, "pers-none", np.nan, 0.0, ppred, yt)
            # user-specific model: trained only on the new person's own
            # calibration session, the usual clinical setup
            omodel = Model(X[ca], y[ca], a.model)
            opred, oconf = omodel.predict(X[te])
            ou = own_model_scores(X, y, subj, series, seg, cal_people, a.model)
            add(p, "own-none", np.nan, 0.0, opred, yt)
            for al in ALPHAS:
                t = threshold_conformal(pu, al)
                add(p, "pers-conformal", al, t, vote_predict(ppred, pconf, seg[te], 1, t, K), yt)
                t = threshold_conformal(ou, al)
                add(p, "own-conformal", al, t, vote_predict(opred, oconf, seg[te], 1, t, K), yt)

        print(f"person {p:2d}: accuracy {base_acc:.3f}   ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}_per_person.csv", index=False)
    pd.DataFrame(info).to_csv(out / f"{tag}_person_info.csv", index=False)
    df["over"] = df.misfire > df.alpha
    summ = (df.groupby(["vote", "method", "alpha"], dropna=False)
              .agg(mean_misfire=("misfire", "mean"), median_misfire=("misfire", "median"),
                   worst_misfire=("misfire", "max"), share_over_target=("over", "mean"),
                   mean_active=("active", "mean"), acc_when_active=("acc_when_active", "mean"))
              .reset_index())
    summ.to_csv(out / f"{tag}_summary.csv", index=False)
    with open(out / f"{tag}_demo.json", "w") as f:
        json.dump(dict(gestures=GESTURES[a.dataset], alphas=ALPHAS, increment_ms=50,
                       people=demo), f)
    print(f"\nNo rejection: mean accuracy on unseen people "
          f"{np.mean([d['accuracy'] for d in demo.values()]):.3f}")
    with pd.option_context("display.width", 200):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
