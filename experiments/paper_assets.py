"""Generate every table, statistic and figure in paper/ and the README from results/.

    python experiments/paper_assets.py

Writes paper/tables/*.tex, paper/figures/*.pdf, figures/*.png and
paper/numbers.tex (LaTeX macros for every number quoted in the text), and
prints the same numbers so README claims can be checked against them.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, P, F = Path("results"), Path("paper"), Path("figures")
for d in (P / "tables", P / "figures", F):
    d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
macros = {}
DS = [("ciil", "Electrode shift"), ("limbpos", "Arm position"),
      ("uci", "UCI, later session"), ("3dc", "3DC, later session")]
TAG = {"ciil": "shift", "limbpos": "arm", "uci": "uci", "3dc": "tdc"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def pc(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def num(x, d=1):
    return f"{100 * x:.{d}f}"


def boot_ci(x, n=10000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.quantile(m, [0.025, 0.975])


def at(df, method, alpha, vote=None):
    d = df if vote is None or "vote" not in df else df[df.vote == vote]
    d = d[d.method == method]
    return d[d.alpha.isna()] if alpha is None else d[np.isclose(d.alpha, alpha)]


def kept(df, method, vote=None):
    d = df if vote is None else df[df.vote == vote]
    g = d[d.method == method].groupby("alpha").misfire.mean()
    return int((g <= g.index + 1e-12).sum()), len(g)


def over(g, a):
    return int((g.misfire > a + 1e-12).sum())


# ============================================================ shared models
sh = {d: pd.read_csv(R / f"{d}_LDA_per_person.csv") for d in ("uci", "3dc")}
info = {d: pd.read_csv(R / f"{d}_LDA_person_info.csv") for d in ("uci", "3dc")}
rf = pd.read_csv(R / "uci_RF_per_person.csv")

rows = []
for d, name in (("uci", "UCI"), ("3dc", "3DC")):
    df = sh[d]
    for label, m, a in [("Fixed 0.90 (LibEMG)", "fixed", None),
                        ("Tuned on training data", "tuned", 0.05),
                        ("Conformal", "conformal", 0.05),
                        ("Conformal + personal", "conformal+personal", 0.05),
                        ("Tuned on training data", "tuned", 0.10),
                        ("Conformal", "conformal", 0.10),
                        ("Conformal + personal", "conformal+personal", 0.10)]:
        g = at(df, m, a, 1)
        rows.append([name if label.startswith("Fixed") else "", label,
                     "--" if a is None else num(a, 0) if a == 0.10 else num(a, 0),
                     num(g.misfire.mean()), num(g.misfire.max()),
                     "--" if a is None else f"{over(g, a)}/{len(g)}",
                     num(g.correct_moves.mean())])
with open(P / "tables" / "shared.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
            "Data & Cutoff & $\\alpha$ & Mean & Worst & Over & Correct \\\\\n\\midrule\n")
    for i, r in enumerate(rows):
        if i == 7:
            f.write("\\midrule\n")
        f.write(" & ".join(r) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

for d, name in (("uci", "Uci"), ("3dc", "Tdc")):
    df = sh[d]
    macros[f"acc{name}"] = pc(info[d][info[d].alpha == 0.05].base_acc.mean())
    macros[f"tunedFive{name}"] = pc(at(df, "tuned", 0.05, 1).misfire.mean())
    macros[f"tunedKept{name}"] = str(kept(df, "tuned", 1)[0])
    macros[f"fixedMean{name}"] = pc(at(df, "fixed", None, 1).misfire.mean())
    macros[f"fixedWorst{name}"] = pc(at(df, "fixed", None, 1).misfire.max())
    macros[f"fixedMoves{name}"] = pc(at(df, "fixed", None, 1).correct_moves.mean())
    for m, k in (("conformal", "Conf"), ("conformal+personal", "ConfPers"),
                 ("own-conformal", "Own"), ("pers-conformal", "Pers")):
        macros[f"kept{k}{name}"] = str(kept(df, m, 1)[0])
    for m, k in (("conformal-vote", "Vote"), ("conformal-vote+personal", "VotePers"),
                 ("conformal-naive", "Naive")):
        macros[f"kept{k}{name}"] = str(kept(df, m, 5)[0])
    macros[f"worstConf{name}"] = pc(at(df, "conformal", 0.05, 1).misfire.max())
    macros[f"worstConfPers{name}"] = pc(at(df, "conformal+personal", 0.05, 1).misfire.max())
    macros[f"confFiveMoves{name}"] = pc(at(df, "conformal", 0.05, 1).correct_moves.mean())
    macros[f"ownMoves{name}"] = pc(df[df.method == "own-none"].correct_moves.mean())
    j = info[d][info[d].alpha == 0.05]
    r1 = stats.spearmanr(j.base_acc, j.test_misfire_at_conformal)
    r2 = stats.spearmanr(j.cal_misfire_at_conformal, j.test_misfire_at_conformal)
    macros[f"rhoAcc{name}"] = f"{r1.statistic:.2f}"
    macros[f"rhoCal{name}"] = f"{r2.statistic:.2f}"
    macros[f"pCal{name}"] = "<0.001" if r2.pvalue < 0.001 else f"{r2.pvalue:.3f}"

macros["rfFixed"] = pc(at(rf, "fixed", None, 1).misfire.mean())
macros["rfTunedFive"] = pc(at(rf, "tuned", 0.05, 1).misfire.mean())
macros["rfKeptConf"] = str(kept(rf, "conformal", 1)[0])
macros["rfKeptTuned"] = str(kept(rf, "tuned", 1)[0])

# ============================================================ drift
dr = {d: pd.read_csv(R / f"drift_{d}.csv") for d, _ in DS}
allp = sum(dr[d].person.nunique() for d, _ in DS)
macros["nPeople"] = str(allp)
ad_full = pd.concat([dr[d][dr[d].method.str.match(r"adaptive(-eta|$)")] for d, _ in DS])
ad_ipw = pd.concat([dr[d][dr[d].method == "adaptive-10%-ipw"] for d, _ in DS])
macros["nRuns"] = f"{len(ad_full):,}".replace(",", "{,}")
macros["nViol"] = str(int((ad_full.bound_value > ad_full.bound + 1e-12).sum()))
macros["nViolIpw"] = str(int((ad_ipw.bound_value > ad_ipw.bound + 1e-12).sum()))
macros["nRunsIpw"] = str(len(ad_ipw))

METHODS = [("Fixed 0.90 (LibEMG)", "fixed", None),
           ("ROC per class", "roc-per-class", None),
           ("Conformal, before change", "conformal-before", 0.05),
           ("Conformal, after change", "conformal-after", 0.05),
           ("Best fixed, hindsight", "oracle-static", 0.05),
           ("Adaptive", "adaptive", 0.05),
           ("Retrain + fixed 0.90", "retrain+fixed", None),
           ("Retrain + conformal", "retrain+conformal", 0.05),
           ("Retrain + adaptive", "retrain+adaptive", 0.05),
           ("10\\% labels: adaptive (plain)", "adaptive-10%", 0.05),
           ("10\\% labels: adaptive (weighted)", "adaptive-10%-ipw", 0.05),
           ("10\\% labels: retrain + fixed", "retrain10+fixed", None),
           ("10\\% labels: retrain + adaptive", "retrain-10%+adaptive", 0.05)]
with open(P / "tables" / "drift.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l" + "rrr" * 4 + "@{}}\n\\toprule\n & ")
    f.write(" & ".join(f"\\multicolumn{{3}}{{c}}{{{n} ({dr[d].person.nunique()})}}"
                       for d, n in DS) + " \\\\\n")
    f.write("".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(4)) + "\n")
    f.write("Cutoff" + " & Mean & Worst & Corr." * 4 + " \\\\\n\\midrule\n")
    for i, (label, m, a) in enumerate(METHODS):
        if label.startswith("Retrain + fixed") or label.startswith("10\\% labels: adaptive (plain"):
            f.write("\\midrule\n")
        cells = []
        for d, _ in DS:
            g = at(dr[d], m, a)
            cells += [num(g.misfire.mean()), num(g.misfire.max()), num(g.correct_moves.mean(), 0)]
        f.write(label + " & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

for d, _ in DS:
    t, df = TAG[d], dr[d]
    acc = df.groupby("person").accuracy.first()
    macros[f"{t}Acc"] = pc(acc.mean())
    macros[f"{t}RetrainAcc"] = pc(at(df, "retrain+fixed", None).accuracy_stream.mean())
    for label, m, a in METHODS:
        g = at(df, m, a)
        key = "".join(w.capitalize() for w in
                      m.replace("+", " ").replace("-", " ").replace("%", "pct").split()
                      ).replace("10", "ten")
        macros[f"{t}{key}Mean"] = pc(g.misfire.mean())
        macros[f"{t}{key}Worst"] = pc(g.misfire.max())
        macros[f"{t}{key}Moves"] = pc(g.correct_moves.mean())
        if a is not None:
            macros[f"{t}{key}Over"] = str(over(g, a))
        else:
            macros[f"{t}{key}Over"] = str(over(g, 0.05))
        kk = df[df.method == m]
        if a is not None:
            macros[f"{t}{key}Kept"] = str(kept(df, m)[0])
            macros[f"{t}{key}KeptAll"] = str(int(sum(
                over(kk[np.isclose(kk.alpha, al)], al) == 0 for al in kk.alpha.unique())))
    lo, hi = boot_ci(at(df, "retrain+adaptive", 0.05).misfire)
    macros[f"{t}RetrainAdaptiveCI"] = f"{100 * lo:.1f}--{100 * hi:.1f}\\%"
    macros[f"{t}N"] = str(df.person.nunique())

# pooled across datasets at 5%
pool = pd.concat(dr.values())
for m, k in (("retrain+adaptive", "RA"), ("retrain+fixed", "RF"), ("adaptive", "A"),
             ("fixed", "F"), ("retrain-10%+adaptive", "RAten"), ("retrain10+fixed", "RFten")):
    g = at(pool, m, 0.05 if "adaptive" in m else None)
    macros[f"pool{k}Over"] = str(over(g, 0.05))
    macros[f"pool{k}Moves"] = pc(g.correct_moves.mean())
    macros[f"pool{k}Mean"] = pc(g.misfire.mean())
    macros[f"pool{k}Worst"] = pc(g.misfire.max())
# paired tests, 5% target, pooled people
ra = at(pool, "retrain+adaptive", 0.05).set_index(["dataset", "person"])
rfx = at(pool, "retrain+fixed", None).set_index(["dataset", "person"])
adp = at(pool, "adaptive", 0.05).set_index(["dataset", "person"])
orc = at(pool, "oracle-static", 0.05).set_index(["dataset", "person"])
w1 = stats.wilcoxon(ra.correct_moves, adp.loc[ra.index].correct_moves)
w2 = stats.wilcoxon(adp.correct_moves, orc.loc[adp.index].correct_moves)
w3 = stats.wilcoxon(ra.misfire, rfx.loc[ra.index].misfire)
for k, w in (("wilRAvsA", w1), ("wilAvsOracle", w2), ("wilRAvsRF", w3)):
    macros[k] = f"10^{{{int(np.floor(np.log10(max(w.pvalue, 1e-300))))+1}}}"
macros["deltaMovesRAvsRF"] = f"{100 * (rfx.correct_moves.mean() - ra.correct_moves.mean()):.1f}"

# ============================================================ motion-level (3DC)
g3 = pd.read_csv(R / "groups_3dc.csv")
o = g3[(g3.setup == "own") & np.isclose(g3.alpha, 0.075)]
macros["gripGesture"] = pc(o[o.controller == "gesture"].grip_moves.mean(), 0)
macros["gripMotion"] = pc(o[o.controller == "motion"].grip_moves.mean(), 0)
macros["gripMisG"] = pc(o[o.controller == "gesture"].misfire.mean())
macros["gripMisM"] = pc(o[o.controller == "motion"].misfire.mean())

with open(P / "numbers.tex", "w") as f:
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")

# ============================================================ figures
COL = {"fixed": "#9a9a9a", "conformal-before": "#d98c2b", "adaptive": "#3b7dd8",
       "retrain+fixed": "#7a5195", "retrain+adaptive": "#1a9e5c"}
LAB = {"fixed": "Fixed 0.90", "conformal-before": "Conformal (before)",
       "adaptive": "Adaptive", "retrain+fixed": "Retrain + fixed 0.90",
       "retrain+adaptive": "Retrain + adaptive"}


def drift_figure(path, dpi=None):
    fig, ax = plt.subplots(2, 4, figsize=(7.16, 3.3), sharex=True)
    for j, (d, name) in enumerate(DS):
        df = dr[d]
        for m in COL:
            if m in ("fixed", "retrain+fixed"):
                g = at(df, m, None)
                for row, col in ((0, "misfire"), (1, "correct_moves")):
                    v = g[col].max() if row == 0 else g[col].mean()
                    ax[row, j].axhline(100 * v, color=COL[m], ls=":", lw=1,
                                       label=LAB[m] if j == 0 and row == 0 else None)
                continue
            g = df[df.method == m].groupby("alpha")
            ax[0, j].plot(100 * g.misfire.max().index, 100 * g.misfire.max(), "o-", ms=2.5,
                          color=COL[m], lw=1.2, label=LAB[m] if j == 0 else None)
            ax[1, j].plot(100 * g.correct_moves.mean().index, 100 * g.correct_moves.mean(),
                          "o-", ms=2.5, color=COL[m], lw=1.2)
        a = np.array([3, 20])
        ax[0, j].plot(a, a, "k--", lw=0.8)
        worst_any = max(df[df.method.isin(list(COL))].groupby(["method", "alpha"], dropna=False)
                        .misfire.max().max(), at(df, "fixed", None).misfire.max())
        top = max(45, 100 * worst_any + 5)
        ax[0, j].set_ylim(0, min(top, 100)); ax[1, j].set_ylim(0, 100)
        ax[0, j].set_title(name, fontsize=8)
        ax[1, j].set_xlabel("Target misfire rate (%)")
    ax[0, 0].set_ylabel("Worst user\nmisfire (%)")
    ax[1, 0].set_ylabel("Correct\nmovements (%)")
    fig.align_ylabels(ax[:, 0])
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=dpi); plt.close(fig)


drift_figure(P / "figures" / "drift.pdf")
drift_figure(F / "drift_overview.png", dpi=170)

# one electrode-shift user: running misfire
tr = json.load(open(R / "drift_ciil_traces.json"))
ex = at(dr["ciil"], "conformal-before", 0.05).sort_values("misfire").iloc[-1].person
macros_extra = {"examplePerson": str(int(ex))}
t = tr[str(int(ex))]
x = np.arange(len(t["run_before"])) * 4 * 0.05
for path, dpi in ((P / "figures" / "one_user.pdf", None), (F / "drift_one_user.png", 170)):
    fig, axx = plt.subplots(figsize=(3.5, 1.9))
    axx.plot(x, 100 * np.array(t["run_before"]), color=COL["conformal-before"], label="Conformal (before shift)")
    axx.plot(x, 100 * np.array(t["run_adaptive"]), color=COL["adaptive"], label="Adaptive, all feedback")
    axx.plot(x, 100 * np.array(t["run_ipw"]), color="#5fb0e8", ls="--", label="Adaptive, 10% feedback (weighted)")
    axx.axhline(5, color="k", ls="--", lw=0.8)
    axx.set_xlabel("Time after electrode shift (s)"); axx.set_ylabel("Running misfire (%)")
    axx.set_ylim(0, 60); axx.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)

# shared models
for path, dpi in ((P / "figures" / "shared.pdf", None), (F / "shared_models.png", 170)):
    fig, axx = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)
    for j, (d, name) in enumerate((("uci", "UCI"), ("3dc", "3DC"))):
        df = sh[d][sh[d].vote == 1]
        for m, c, l in (("tuned", "#c44e52", "Tuned"), ("conformal", "#d98c2b", "Conformal"),
                        ("conformal+personal", "#1a9e5c", "Conformal + personal")):
            g = df[df.method == m].groupby("alpha").misfire.mean()
            axx[j].plot(100 * g.index, 100 * g, "o-", ms=2.5, color=c, label=l if j == 0 else None)
        axx[j].plot([3, 20], [3, 20], "k:", lw=0.8)
        axx[j].set_title(name, fontsize=8); axx[j].set_xlabel("Target (%)")
    axx[0].set_ylabel("Mean misfire (%)")
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(path, dpi=dpi); plt.close(fig)

with open(P / "numbers.tex", "a") as f:
    for k, v in macros_extra.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
macros.update(macros_extra)
for k in sorted(macros):
    print(f"{k:32s} {macros[k]}")
