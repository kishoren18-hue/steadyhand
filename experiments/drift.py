"""Do cutoffs keep their promise when conditions change after calibration?

Each person gets their own (user-specific) LDA model, the usual clinical setup:
  model        trained on the first half of the person's series-1 recordings
  calibration  second half of series 1 (same conditions as training)
  test stream  all of series 2, in recording order (after the change)

The change is a 45-degree armband rotation (ciil), new arm positions
(limbpos), or simply a later session (uci, 3dc).

Misfire = the device moves in a way the user did not intend (output is a
movement class, not "rest", and differs from the intended class). A confident
"rest" prediction holds the device still and is never a misfire.

Methods, per held-out person p (calibration uses only other people unless
marked personal):
  fixed                LibEMG default 0.90
  personal             meets the target on p's own calibration recording
  roc-per-class        one cutoff per predicted class from p's calibration
                       ROC (Youden's J), in the spirit of Menon et al. 2015
  conformal-before     conformal risk control on other people's calibration
                       recordings (what you have if nobody recorded the change)
  conformal-after      conformal risk control on other people's test streams
                       (requires recordings of others after the same change)
  conformal-before+personal
  oracle-static        best single cutoff for p's own test stream (hindsight)
  adaptive             starts at conformal-before, updated from a misfire bit
                       on every decision (steadyhand.adaptive)
  adaptive-10%         plain rule, feedback on a random 10% of decisions
  adaptive-10%-ipw     label-efficient rule, same 10% of decisions
  retrain+fixed        the model is refit every BLOCK decisions on its
                       training data plus every labeled test decision so far
                       (prequential; needs the full label, not just a bit)
  retrain+conformal    the same model stream, conformal-before cutoff
  retrain+adaptive     the same model stream, adaptive cutoff on top
  retrain-10%+...      the same with labels on the random 10% only

Usage
  python experiments/drift.py --dataset ciil --data data/ciil/ElectrodeShift
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import build_features, REST
from common import Model
from steadyhand.risk import misfire_curve, candidate_thresholds, harmful
from steadyhand.cutoffs import crc_threshold, personal_cutoff
from steadyhand.adaptive import simulate, S_MAX, to_score

TARGETS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
ETA = 0.5
ETAS = [0.1, 0.25, 0.5, 1.0]
FB = 0.10           # share of decisions with feedback in the sparse setting
BLOCK = 100         # retraining period, in decisions (5 s at a 50 ms step)


def stream(m, Xk, yk, rest):
    pred, conf = m.predict(Xk)
    return dict(pred=pred, conf=conf, y=yk, wrong=harmful(pred, yk, rest))


def prequential(Xtr, ytr, Xte, yte, labeled, rest):
    """Refit every BLOCK decisions on the training data plus all labeled test
    decisions seen so far; each block is scored by a model that never saw it."""
    preds, confs = [], []
    for b in range(0, len(yte), BLOCK):
        k = np.flatnonzero(labeled[:b])
        m = Model(np.vstack([Xtr, Xte[k]]), np.r_[ytr, yte[k]])
        pr, cf = m.predict(Xte[b:b + BLOCK])
        preds.append(pr); confs.append(cf)
    pred, conf = np.concatenate(preds), np.concatenate(confs)
    return dict(pred=pred, conf=conf, y=yte, wrong=harmful(pred, yte, rest))


def user_specific(X, y, subj, series, seg, p, rest, sparse_seed):
    s1 = (subj == p) & (series == 1)
    segs = np.unique(seg[s1])
    first = np.isin(seg, segs[: len(segs) // 2]) & s1
    cal = s1 & ~first
    test = (subj == p) & (series == 2)
    m = Model(X[first], y[first])
    out = {"cal": stream(m, X[cal], y[cal], rest), "test": stream(m, X[test], y[test], rest)}
    T = int(test.sum())
    sparse = np.random.default_rng(sparse_seed).random(T) < FB
    out["sparse"] = sparse
    out["retrain"] = prequential(X[first], y[first], X[test], y[test], np.ones(T, bool), rest)
    out["retrain10"] = prequential(X[first], y[first], X[test], y[test], sparse, rest)
    return out


def crc(people, key, grid, alpha):
    curves = [misfire_curve(q[key]["conf"], q[key]["wrong"], grid) for q in people]
    return crc_threshold(grid, curves, alpha)


def roc_per_class(cal, n_classes):
    """Per predicted class, the cutoff maximizing TPR - FPR for telling safe
    decisions from misfires on the calibration recording."""
    t = np.full(n_classes, 0.0)
    for c in range(n_classes):
        k = cal["pred"] == c
        good, bad = cal["conf"][k & ~cal["wrong"]], cal["conf"][k & cal["wrong"]]
        if len(good) == 0 or len(bad) == 0:
            continue
        cand = np.unique(np.r_[0.0, cal["conf"][k]])
        tpr = np.array([(good > u).mean() for u in cand])
        fpr = np.array([(bad > u).mean() for u in cand])
        t[c] = cand[np.argmax(tpr - fpr)]
    return t


def stats(acted, s, rest):
    mis = acted & s["wrong"]
    move = s["y"] != rest
    correct_move = acted & (s["pred"] == s["y"]) & move
    false_act = acted & (s["pred"] != rest) & ~move
    return dict(misfire=float(mis.mean()),
                correct_moves=float(correct_move.sum() / max(move.sum(), 1)),
                false_activation=float(false_act.sum() / max((~move).sum(), 1)),
                moving=float((acted & (s["pred"] != rest)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    rest = REST[a.dataset]
    X, y, subj, series, seg = build_features(a.dataset, a.data, out / f"features_{a.dataset}.npz")
    K = int(y.max()) + 1
    people = np.unique(subj)
    scored = {}
    for p in people:
        scored[int(p)] = user_specific(X, y, subj, series, seg, int(p), rest, int(p))
        print(f"{a.dataset} person {int(p):2d} scored", flush=True)
    grid_all = candidate_thresholds(np.concatenate(
        [q[k]["conf"] for q in scored.values() for k in ("cal", "test")]))

    rows, traces = [], {}
    for p in people:
        p = int(p)
        me = scored[p]
        others = [scored[int(j)] for j in people if int(j) != p]
        te, sparse = me["test"], me["sparse"]
        T = len(te["conf"])
        acc = float((te["pred"] == te["y"]).mean())

        def add(method, alpha, acted, s=te, t=np.nan, eta=np.nan, bound=np.nan,
                bound_value=np.nan):
            rows.append(dict(dataset=a.dataset, person=p, method=method, alpha=alpha,
                             t=t, eta=eta, bound=bound, bound_value=bound_value,
                             decisions=T, accuracy=acc,
                             accuracy_stream=float((s["pred"] == s["y"]).mean()),
                             **stats(acted, s, rest)))

        add("fixed", np.nan, te["conf"] > 0.9, t=0.9)
        t_roc = roc_per_class(me["cal"], K)
        add("roc-per-class", np.nan, te["conf"] > t_roc[te["pred"]])
        for s_name in ("retrain", "retrain10"):
            add(f"{s_name}+fixed", np.nan, me[s_name]["conf"] > 0.9, s=me[s_name], t=0.9)
        for al in TARGETS:
            t_per = personal_cutoff(me["cal"]["conf"], me["cal"]["wrong"], al)
            t_bef = crc(others, "cal", grid_all, al)
            t_aft = crc(others, "test", grid_all, al)
            t_orc = personal_cutoff(te["conf"], te["wrong"], al)
            for name, t in [("personal", t_per), ("conformal-before", t_bef),
                            ("conformal-after", t_aft),
                            ("conformal-before+personal", max(t_bef, t_per)),
                            ("oracle-static", t_orc)]:
                add(name, al, te["conf"] > t, t=t)
            tau1 = min(to_score(t_bef), S_MAX)
            for eta in (ETAS if al == 0.05 else [ETA]):
                acted, tr = simulate(te["conf"], te["wrong"], al, t_bef, eta)
                bnd = al + (S_MAX + eta - tau1) / (eta * T)
                add("adaptive" if eta == ETA else f"adaptive-eta{eta}", al, acted,
                    t=t_bef, eta=eta, bound=bnd, bound_value=float((acted & te["wrong"]).mean()))
                if eta == ETA and al == 0.05:
                    run = lambda v: (np.cumsum(v) / np.arange(1, T + 1))[::4].round(5).tolist()
                    acted_i, tr_i = simulate(te["conf"], te["wrong"], al, t_bef, ETA, sparse, rate=FB)
                    traces[p] = dict(tau=tr[::4].round(4).tolist(),
                                     run_adaptive=run(acted & te["wrong"]),
                                     run_before=run((te["conf"] > t_bef) & te["wrong"]),
                                     run_ipw=run(acted_i & te["wrong"]))
            # plain rule, sparse feedback: bound covers labeled decisions only
            acted, _ = simulate(te["conf"], te["wrong"], al, t_bef, ETA, sparse)
            nl = max(int(sparse.sum()), 1)
            add("adaptive-10%", al, acted, t=t_bef, eta=ETA,
                bound=al + (S_MAX + ETA - tau1) / (ETA * nl),
                bound_value=float((acted & te["wrong"])[sparse].mean()))
            # label-efficient rule: deterministic bound on the weighted rate,
            # expectation bound on the true rate over all decisions
            acted, _ = simulate(te["conf"], te["wrong"], al, t_bef, ETA, sparse, rate=FB)
            add("adaptive-10%-ipw", al, acted, t=t_bef, eta=ETA,
                bound=al + (S_MAX + ETA / FB - tau1) / (ETA * T),
                bound_value=float(((acted & te["wrong"]) & sparse).sum() / FB / T))
            # retraining baselines and the combination
            for s_name, fb, rate, tag in (("retrain", None, None, "retrain"),
                                          ("retrain10", sparse, FB, "retrain-10%")):
                s = me[s_name]
                add(f"{tag}+conformal", al, s["conf"] > t_bef, s=s, t=t_bef)
                acted, _ = simulate(s["conf"], s["wrong"], al, t_bef, ETA, fb, rate=rate)
                add(f"{tag}+adaptive", al, acted, s=s, t=t_bef, eta=ETA)
        print(f"{a.dataset} person {p:2d}: own-model accuracy after change {acc:.3f}, "
              f"retrained {float((me['retrain']['pred'] == te['y']).mean()):.3f}, {T} decisions",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"drift_{a.dataset}.csv", index=False)
    with open(out / f"drift_{a.dataset}_traces.json", "w") as f:
        json.dump(traces, f)
    df["over"] = df.misfire > df.alpha
    summ = (df.groupby(["method", "alpha"], dropna=False)
              .agg(mean_misfire=("misfire", "mean"), worst=("misfire", "max"),
                   over=("over", "mean"), correct_moves=("correct_moves", "mean"),
                   acc=("accuracy_stream", "mean")).reset_index())
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(summ[summ.alpha.isna() | (summ.alpha == 0.05)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
