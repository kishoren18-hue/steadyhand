"""Print every table and number used in the paper, straight from results/.

  python experiments/report.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

R = Path("results")
MAIN = ["fixed", "tuned", "personal", "conformal", "conformal+personal",
        "pers-conformal", "own-conformal"]
VOTE = ["fixed", "conformal-naive", "conformal-vote", "conformal-vote+personal"]


def pct(x):
    return f"{100 * x:.1f}%"


def table(df, methods, alphas, vote=1):
    d = df[df.vote == vote] if "vote" in df.columns else df
    out = []
    for m in methods:
        for a in ([np.nan] if m in ("fixed", "pers-none", "own-none") else alphas):
            g = d[(d.method == m) & (d.alpha.isna() if np.isnan(a) else np.isclose(d.alpha, a))]
            if g.empty:
                continue
            out.append(dict(method=m, target="none" if np.isnan(a) else pct(a),
                            mean=pct(g.misfire.mean()), worst=pct(g.misfire.max()),
                            over="n/a" if np.isnan(a) else f"{(g.misfire > a).sum()}/{len(g)}",
                            moves=pct(g.active.mean())))
    return pd.DataFrame(out)


def promise(df, method, vote=1):
    """How many targets the method kept on average."""
    d = df[(df.vote == vote) & (df.method == method)] if "vote" in df.columns else df[df.method == method]
    g = d.groupby("alpha").misfire.mean()
    return int((g <= g.index + 1e-12).sum()), len(g)


for f in sorted(R.glob("*_per_person.csv")):
    if f.name.startswith("old_"):
        continue
    tag = f.name.replace("_per_person.csv", "")
    df = pd.read_csv(f)
    print(f"\n==================== {tag}")
    acc = df[(df.method == "fixed") & (df.vote == 1)] if "vote" in df.columns else None
    for name in ("pers-none", "own-none"):
        g = df[df.method == name]
        if not g.empty:
            print(f"accuracy, {name}: {pct(g.acc_when_active.mean())}")
    info = R / f"{tag}_person_info.csv"
    if info.exists():
        i = pd.read_csv(info)
        print(f"accuracy, shared model, no rejection: {pct(i[i.alpha == 0.05].base_acc.mean())}")
    print("\nMain results")
    print(table(df, MAIN, [0.03, 0.05, 0.10]).to_string(index=False))
    print("\nTargets kept on average (mean misfire <= target):")
    for m in MAIN[1:]:
        if m in set(df.method):
            k, n = promise(df, m)
            print(f"  {m:20s} {k}/{n}")
    if "vote" in df.columns and (df.vote == 5).any():
        print("\nMajority voting (k = 5)")
        print(table(df, VOTE, [0.05, 0.10], vote=5).to_string(index=False))
        for m in VOTE[1:]:
            k, n = promise(df, m, vote=5)
            print(f"  {m:24s} kept {k}/{n}")
    if info.exists():
        j = i[i.alpha == 0.05]
        r1 = stats.spearmanr(j.base_acc, j.test_misfire_at_conformal)
        r2 = stats.spearmanr(j.cal_misfire_at_conformal, j.test_misfire_at_conformal)
        print(f"\nWho goes over (5% target, conformal):")
        print(f"  Spearman(shared-model accuracy, misfire) = {r1.statistic:.2f} (p={r1.pvalue:.4f})")
        print(f"  Spearman(calibration-session misfire, test misfire) = {r2.statistic:.2f} (p={r2.pvalue:.4f})")
        own = df[df.method == "own-none"].set_index("person").acc_when_active
        worst = j.sort_values("test_misfire_at_conformal", ascending=False).head(4)
        for _, r in worst.iterrows():
            extra = (f"own-model accuracy {pct(own[int(r.person)])}, "
                     if int(r.person) in own.index else "")
            print(f"  person {int(r.person):2d}: shared-model accuracy {pct(r.base_acc)}, "
                  f"{extra}misfire at conformal cutoff {pct(r.test_misfire_at_conformal)}")
        if len(own):
            print(f"  own-model accuracy across people: median {pct(own.median())}, min {pct(own.min())}")


# ---------------------------------------------------------------- drift and motion-level
NAMES = {"ciil": "electrode shift", "limbpos": "arm position", "uci": "UCI later session",
         "3dc": "3DC later session"}
for d, name in NAMES.items():
    f = R / f"drift_{d}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    print(f"\n==================== drift: {name} ({df.person.nunique()} people, own models)")
    print(f"own-model accuracy after the change: {pct(df.groupby('person').accuracy.first().mean())}")
    fx = df[df.method == "fixed"]
    print(f"fixed 0.90: mean {pct(fx.misfire.mean())}, worst {pct(fx.misfire.max())}, moves {pct(fx.active.mean())}")
    for m in ["personal", "conformal-before", "conformal-before+personal", "conformal-after",
              "adaptive", "adaptive-10%", "oracle-static"]:
        g = df[df.method == m]
        kept = int((g.groupby("alpha").misfire.mean() <= g.groupby("alpha").misfire.mean().index + 1e-12).sum())
        parts = []
        for a in (0.05, 0.10):
            h = g[np.isclose(g.alpha, a)]
            parts.append(f"{pct(a)}: mean {pct(h.misfire.mean())} worst {pct(h.misfire.max())} "
                         f"over {(h.misfire > a).sum()}/{len(h)} moves {pct(h.active.mean())}")
        print(f"  {m:26s} kept {kept}/6 | " + " | ".join(parts))
    ad = df[df.method.str.startswith("adaptive")]
    print(f"  adaptive guarantee violations: {int((ad.bound_misfire > ad.bound + 1e-12).sum())} of {len(ad)} runs")

f = R / "groups_3dc.csv"
if f.exists():
    df = pd.read_csv(f)
    print("\n==================== motion-level controller (3DC)")
    s = df.groupby(["setup", "controller", "alpha"]).agg(
        mean=("misfire", "mean"), worst=("misfire", "max"), moves=("active", "mean"),
        grip_moves=("grip_moves", "mean")).reset_index()
    print(s[s.alpha.isin([0.05, 0.075, 0.10])].round(3).to_string(index=False))
