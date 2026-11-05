"""Do cutoffs keep their promise when conditions change after calibration?

Each person gets their own (user-specific) LDA model, the usual clinical setup:
  model        trained on the first half of the person's series-1 recordings
  calibration  second half of series 1 (same conditions as training)
  test stream  all of series 2, in recording order (after the change)

The change is a 45-degree armband rotation (ciil), new arm positions
(limbpos), or simply a later session (uci, 3dc).

Cutoffs compared, per held-out person p (calibration uses only other people):
  fixed              LibEMG default 0.90
  personal           meets the target on p's own calibration recording
  conformal-before   conformal risk control on other people's calibration
                     recordings (what you have if nobody recorded the change)
  conformal-after    conformal risk control on other people's test streams
                     (requires recordings of others after the same change)
  conformal-before+personal
  adaptive           starts at conformal-before, updated online from feedback
                     on every decision (steadyhand.adaptive)
  adaptive-10%       the same with feedback on a random 10% of decisions

Usage
  python experiments/drift.py --dataset ciil --data data/ciil/ElectrodeShift
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import build_features
from common import Model
from steadyhand.risk import misfire_curve, candidate_thresholds
from steadyhand.cutoffs import crc_threshold, personal_cutoff
from steadyhand.adaptive import simulate, S_MAX, to_score

TARGETS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
ETA = 0.5
ETAS = [0.1, 0.25, 0.5, 1.0]


def user_specific(X, y, subj, series, seg, p):
    """Train on the first half of p's series-1 segments; score the rest."""
    s1 = (subj == p) & (series == 1)
    segs = np.unique(seg[s1])
    first = np.isin(seg, segs[: len(segs) // 2]) & s1
    cal = s1 & ~first
    test = (subj == p) & (series == 2)
    m = Model(X[first], y[first])
    out = {}
    for name, k in (("cal", cal), ("test", test)):
        pred, conf = m.predict(X[k])
        out[name] = dict(conf=conf, wrong=pred != y[k])
    return out


def crc(people, key, grid, alpha):
    curves = [misfire_curve(q[key]["conf"], q[key]["wrong"], grid) for q in people]
    return crc_threshold(grid, curves, alpha)


def stats(acted, wrong):
    mis = acted & wrong
    return dict(misfire=float(mis.mean()), active=float(acted.mean()),
                acc_when_active=float(1 - mis[acted].mean()) if acted.any() else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    X, y, subj, series, seg = build_features(a.dataset, a.data, out / f"features_{a.dataset}.npz")
    people = np.unique(subj)
    scored = {int(p): user_specific(X, y, subj, series, seg, p) for p in people}
    grid_all = candidate_thresholds(np.concatenate(
        [q[k]["conf"] for q in scored.values() for k in ("cal", "test")]))

    rows, traces = [], {}
    for p in people:
        p = int(p)
        me = scored[p]
        others = [scored[int(j)] for j in people if int(j) != p]
        conf, wrong = me["test"]["conf"], me["test"]["wrong"]
        T = len(conf)
        acc = float(1 - wrong.mean())

        def add(method, alpha, acted, t=np.nan, eta=np.nan, bound=np.nan, fb=None):
            # bound_misfire: the misfire rate the guarantee covers, i.e. over
            # the decisions that received feedback (all of them unless fb given)
            k = np.ones(T, bool) if fb is None else fb
            rows.append(dict(dataset=a.dataset, person=p, method=method, alpha=alpha,
                             t=t, eta=eta, bound=bound, decisions=T, accuracy=acc,
                             bound_misfire=float((acted & wrong)[k].mean()),
                             **stats(acted, wrong)))

        add("fixed", np.nan, conf > 0.9, 0.9)
        rng = np.random.default_rng(p)
        sparse = rng.random(T) < 0.10
        for al in TARGETS:
            t_per = personal_cutoff(me["cal"]["conf"], me["cal"]["wrong"], al)
            t_bef = crc(others, "cal", grid_all, al)
            t_aft = crc(others, "test", grid_all, al)
            # oracle: the best single fixed cutoff for this person's test stream,
            # chosen with hindsight; not achievable, a reference for activity
            t_orc = personal_cutoff(conf, wrong, al)
            for name, t in [("personal", t_per), ("conformal-before", t_bef),
                            ("conformal-after", t_aft),
                            ("conformal-before+personal", max(t_bef, t_per)),
                            ("oracle-static", t_orc)]:
                add(name, al, conf > t, t)
            for eta in ETAS if al == 0.05 else [ETA]:
                acted, tr = simulate(conf, wrong, al, t_bef, eta)
                slack = (S_MAX + eta - min(to_score(t_bef), S_MAX)) / (eta * T)
                add("adaptive" if eta == ETA else f"adaptive-eta{eta}", al, acted,
                    t_bef, eta, al + slack)
                if eta == ETA and al == 0.05:
                    traces[p] = dict(tau=tr[::4].round(4).tolist(),
                                     run_adaptive=(np.cumsum(acted & wrong) / np.arange(1, T + 1))[::4].round(5).tolist(),
                                     run_before=(np.cumsum((conf > t_bef) & wrong) / np.arange(1, T + 1))[::4].round(5).tolist())
            acted, _ = simulate(conf, wrong, al, t_bef, ETA, feedback=sparse)
            lab = sparse.sum()
            add("adaptive-10%", al, acted, t_bef, ETA,
                al + (S_MAX + ETA - min(to_score(t_bef), S_MAX)) / (ETA * max(lab, 1)), fb=sparse)
        print(f"{a.dataset} person {p:2d}: own-model accuracy after change {acc:.3f}, {T} decisions", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"drift_{a.dataset}.csv", index=False)
    with open(out / f"drift_{a.dataset}_traces.json", "w") as f:
        json.dump(traces, f)
    df["over"] = df.misfire > df.alpha
    summ = (df.groupby(["method", "alpha"], dropna=False)
              .agg(mean_misfire=("misfire", "mean"), worst=("misfire", "max"),
                   over=("over", "mean"), active=("active", "mean"),
                   bound_max=("bound", "max")).reset_index())
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"\nown-model accuracy after change: mean {df.groupby('person').accuracy.first().mean():.3f}")
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
