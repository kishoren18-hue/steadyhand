"""Generate every table, statistic and figure in paper/ from results/.

    python experiments/paper_assets.py

Writes paper/tables/*.tex, paper/figures/*.pdf and paper/numbers.tex
(LaTeX macros for every number quoted in the text).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, P = Path("results"), Path("paper")
(P / "tables").mkdir(parents=True, exist_ok=True)
(P / "figures").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
macros = {}


def pc(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def boot_ci(x, n=10000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.quantile(m, [0.025, 0.975])


def at(df, method, alpha, vote=None):
    d = df if vote is None else df[df.vote == vote]
    d = d[d.method == method]
    return d[d.alpha.isna()] if alpha is None else d[np.isclose(d.alpha, alpha)]


def kept(df, method, vote=None):
    d = df if vote is None else df[df.vote == vote]
    g = d[d.method == method].groupby("alpha").misfire.mean()
    return int((g <= g.index + 1e-12).sum()), len(g)


# ------------------------------------------------------------- shared-model results
sh = {d: pd.read_csv(R / f"{d}_LDA_per_person.csv") for d in ("uci", "3dc")}
rf = pd.read_csv(R / "uci_RF_per_person.csv")
info = {d: pd.read_csv(R / f"{d}_LDA_person_info.csv") for d in ("uci", "3dc")}

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
        g = at(df, m, a, vote=1)
        rows.append((name, label, "--" if a is None else pc(a, 0), pc(g.misfire.mean()),
                     pc(g.misfire.max()),
                     "--" if a is None else f"{(g.misfire > a).sum()}/{len(g)}",
                     pc(g.active.mean())))
with open(P / "tables" / "shared.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
            "Data & Cutoff & $\\alpha$ & Mean & Worst & Over & Moves \\\\\n\\midrule\n")
    for i, r in enumerate(rows):
        if i == 7:
            f.write("\\midrule\n")
        f.write(" & ".join(r).replace("%", "\\%").replace("\\\\%", "\\%") + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

for d, name in (("uci", "Uci"), ("3dc", "Tdc")):
    df = sh[d]
    macros[f"acc{name}"] = pc(info[d][info[d].alpha == 0.05].base_acc.mean())
    macros[f"tunedFive{name}"] = pc(at(df, "tuned", 0.05, 1).misfire.mean())
    macros[f"tunedKept{name}"] = str(kept(df, "tuned", 1)[0])
    macros[f"fixedMean{name}"] = pc(at(df, "fixed", None, 1).misfire.mean())
    macros[f"fixedWorst{name}"] = pc(at(df, "fixed", None, 1).misfire.max())
    for m, key in (("conformal", "Conf"), ("conformal+personal", "ConfPers"),
                   ("own-conformal", "Own")):
        macros[f"kept{key}{name}"] = str(kept(df, m, 1)[0])
    for m, key in (("conformal-naive", "Naive"), ("conformal-vote", "Vote"),
                   ("conformal-vote+personal", "VotePers")):
        macros[f"kept{key}{name}"] = str(kept(df, m, 5)[0])
    own = df[df.method == "own-none"].acc_when_active.mean()
    macros[f"ownAcc{name}"] = pc(own)
    j = info[d][info[d].alpha == 0.05]
    r = stats.spearmanr(j.cal_misfire_at_conformal, j.test_misfire_at_conformal)
    macros[f"rhoCal{name}"] = f"{r.statistic:.2f}"
    macros[f"pCal{name}"] = f"{r.pvalue:.3f}" if r.pvalue >= 0.001 else "<0.001"
u = sh["uci"]
macros["worstConfUci"] = pc(at(u, "conformal", 0.05, 1).misfire.max())
macros["worstConfPersUci"] = pc(at(u, "conformal+personal", 0.05, 1).misfire.max())
macros["ownFiveMovesUci"] = pc(at(u, "own-conformal", 0.05, 1).active.mean())
macros["sharedFiveMovesUci"] = pc(at(u, "conformal+personal", 0.05, 1).active.mean())
macros["rfKeptConf"] = str(kept(rf, "conformal", 1)[0])
macros["rfTunedFive"] = pc(at(rf, "tuned", 0.05, 1).misfire.mean())
macros["rfFixed"] = pc(at(rf, "fixed", None, 1).misfire.mean())
j = info["uci"][info["uci"].alpha == 0.05]
r = stats.spearmanr(j.base_acc, j.test_misfire_at_conformal)
macros["rhoAccUci"] = f"{r.statistic:.2f}"
p22 = sh["uci"][(sh["uci"].method == "own-none") & (sh["uci"].person == 22)].acc_when_active.iloc[0]
macros["pTwentyTwoShared"] = pc(j[j.person == 22].base_acc.iloc[0])
macros["pTwentyTwoOwn"] = pc(p22)

# ------------------------------------------------------------- drift results
NAMES = {"ciil": "Electrode shift", "limbpos": "Arm position",
         "uci": "UCI, later session", "3dc": "3DC, later session"}
dr = {d: pd.read_csv(R / f"drift_{d}.csv") for d in NAMES}
methods = [("Fixed 0.90 (LibEMG)", "fixed", None), ("Personal", "personal", 0.05),
           ("Conformal, before change", "conformal-before", 0.05),
           ("Conformal, after change", "conformal-after", 0.05),
           ("Adaptive (ours)", "adaptive", 0.05),
           ("Adaptive, 10\\% feedback", "adaptive-10%", 0.05),
           ("Best fixed, hindsight", "oracle-static", 0.05)]
with open(P / "tables" / "drift.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l" + "rrr" * 4 + "@{}}\n\\toprule\n")
    f.write(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NAMES[d]} ({dr[d].person.nunique()})}}"
                               for d in NAMES) + " \\\\\n")
    f.write("".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(4)) + "\n")
    f.write("Cutoff" + " & Mean & Worst & Moves" * 4 + " \\\\\n\\midrule\n")
    for label, m, a in methods:
        cells = []
        for d in NAMES:
            g = at(dr[d], m, a)
            cells += [f"{100 * g.misfire.mean():.1f}", f"{100 * g.misfire.max():.1f}",
                      f"{100 * g.active.mean():.0f}"]
        f.write(label + " & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

c = dr["ciil"]
macros["shiftAcc"] = pc(c.groupby("person").accuracy.first().mean())
macros["shiftFixedMean"] = pc(at(c, "fixed", None).misfire.mean())
macros["shiftFixedWorst"] = pc(at(c, "fixed", None).misfire.max())
macros["shiftBeforeMean"] = pc(at(c, "conformal-before", 0.05).misfire.mean())
macros["shiftBeforeWorst"] = pc(at(c, "conformal-before", 0.05).misfire.max())
macros["shiftBeforeKept"] = str(kept(c, "conformal-before")[0])
a5 = at(c, "adaptive", 0.05)
macros["shiftAdaptMean"] = pc(a5.misfire.mean())
lo, hi = boot_ci(a5.misfire)
macros["shiftAdaptCI"] = f"{100 * lo:.1f}--{100 * hi:.1f}\\%"
macros["shiftAdaptWorst"] = pc(a5.misfire.max())
macros["shiftAdaptMoves"] = pc(a5.active.mean())
macros["shiftAfterMoves"] = pc(at(c, "conformal-after", 0.05).active.mean())
macros["shiftOracleMoves"] = pc(at(c, "oracle-static", 0.05).active.mean())
macros["shiftSparseKept"] = str(kept(c, "adaptive-10%")[0])
lp = dr["limbpos"]
macros["armFixedMean"] = pc(at(lp, "fixed", None).misfire.mean())
macros["armBeforeTen"] = pc(at(lp, "conformal-before", 0.10).misfire.mean())
macros["armAdaptTenMean"] = pc(at(lp, "adaptive", 0.10).misfire.mean())
macros["armAdaptTenWorst"] = pc(at(lp, "adaptive", 0.10).misfire.max())
macros["armAdaptTenMoves"] = pc(at(lp, "adaptive", 0.10).active.mean())
macros["armAfterTenMoves"] = pc(at(lp, "conformal-after", 0.10).active.mean())

n_runs = viol = 0
for d, df in dr.items():
    ad = df[df.method.str.startswith("adaptive")]
    n_runs += len(ad)
    viol += int((ad.bound_misfire > ad.bound + 1e-12).sum())
macros["nRuns"] = f"{n_runs:,}".replace(",", "{,}")
macros["nViol"] = str(viol)
macros["nPeople"] = str(sum(df.person.nunique() for df in dr.values()))

# paired test: adaptive vs conformal-after activity at 5%, pooled over datasets
a = np.concatenate([at(df, "adaptive", 0.05).sort_values("person").active.values for df in dr.values()])
b = np.concatenate([at(df, "conformal-after", 0.05).sort_values("person").active.values for df in dr.values()])
w = stats.wilcoxon(a, b)
macros["wilcoxP"] = "10^{" + str(int(np.ceil(np.log10(w.pvalue)))) + "}" if w.pvalue < 1e-3 else f"{w.pvalue:.3f}"
o = np.concatenate([at(df, "oracle-static", 0.05).sort_values("person").active.values for df in dr.values()])
w2 = stats.wilcoxon(a, o)
macros["wilcoxOracleP"] = "10^{" + str(int(np.ceil(np.log10(w2.pvalue)))) + "}" if w2.pvalue < 1e-3 else f"{w2.pvalue:.3f}"
for dname in ("limbpos", "uci", "3dc"):
    macros[f"sparseKept{dname.replace('3dc', 'tdc')}"] = str(kept(dr[dname], "adaptive-10%")[0])

# ------------------------------------------------------------- motion-level controller
g = pd.read_csv(R / "groups_3dc.csv")
own = g[(g.setup == "own") & np.isclose(g.alpha, 0.075)]
macros["gripGesture"] = pc(own[own.controller == "gesture"].grip_moves.mean(), 0)
macros["gripMotion"] = pc(own[own.controller == "motion"].grip_moves.mean(), 0)
macros["gripMisG"] = pc(own[own.controller == "gesture"].misfire.mean())
macros["gripMisM"] = pc(own[own.controller == "motion"].misfire.mean())

with open(P / "numbers.tex", "w") as f:
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ------------------------------------------------------------- figures (print size)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.5, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
C = {"conformal-before": "#eb6834", "conformal-after": "#1baf7a", "adaptive": "#2a78d6",
     "tuned": "#8a8a85", "conformal+personal": "#2a78d6"}

fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
for ax, (d, df) in zip(axes, dr.items()):
    for m, lab in (("conformal-before", "Conformal, calibrated before change"),
                   ("conformal-after", "Conformal, calibrated after change"),
                   ("adaptive", "Adaptive (ours)")):
        s = df[df.method == m].groupby("alpha").misfire.max()
        ax.plot(s.index, s.values, color=C[m], marker="o", ms=2.5, label=lab)
    ax.plot([0, .21], [0, .21], ls="--", color="#52514e", lw=0.8)
    ax.set_title(f"{NAMES[d]} ({df.person.nunique()})", fontsize=8, loc="left")
    ax.set_xlim(0, .21); ax.set_xticks([0, .10, .20])
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Target")
axes[0].set_ylabel("Worst user's misfire rate")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
fig.savefig(P / "figures" / "worst_user.pdf"); plt.close(fig)

tr = json.loads((R / "drift_ciil_traces.json").read_text())
b5 = at(c, "conformal-before", 0.05).set_index("person").misfire
pid = str(int(b5.idxmax()))
t = tr[pid]
x = np.arange(len(t["run_before"])) * 4 * 0.05
fig, ax = plt.subplots(figsize=(3.4, 1.8))
ax.plot(x, t["run_before"], color=C["conformal-before"], label="Conformal, calibrated before shift")
ax.plot(x, t["run_adaptive"], color=C["adaptive"], label="Adaptive (ours)")
ax.axhline(0.05, ls="--", color="#52514e", lw=0.8)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Seconds of use after a 45-degree armband shift")
ax.set_ylabel("Misfire rate so far")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(P / "figures" / "one_user.pdf"); plt.close(fig)
macros_pid = pid

fig, ax = plt.subplots(figsize=(3.4, 1.9))
for d, ls in (("uci", "-"), ("3dc", "--")):
    for m, lab in (("tuned", "Tuned on training data"), ("conformal+personal", "Conformal + personal")):
        s = sh[d][(sh[d].vote == 1) & (sh[d].method == m)].groupby("alpha").misfire.mean()
        ax.plot(s.index, s.values, color=C[m], ls=ls, marker="o", ms=2.5,
                label=f"{lab}, {d.upper()}")
ax.plot([0, .21], [0, .21], color="#52514e", lw=0.8, ls=":")
ax.set_xlim(0, .21); ax.set_xticks([0, .05, .10, .15, .20])
ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Target"); ax.set_ylabel("Mean misfire, unseen users")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(P / "figures" / "shared.pdf"); plt.close(fig)

with open(P / "numbers.tex", "a") as f:
    f.write(f"\\newcommand{{\\examplePerson}}{{{macros_pid}}}\n")
print(f"{len(macros) + 1} macros, 2 tables, 3 figures")
for k in ("shiftAdaptCI", "wilcoxP", "wilcoxOracleP", "nRuns", "nPeople", "keptVotePersTdc", "rfKeptConf"):
    print(k, macros[k])
