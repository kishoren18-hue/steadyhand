"""Figures for the drift experiments (results/drift_*.csv)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"conformal-before": "#eb6834", "conformal-after": "#1baf7a",
     "adaptive": "#2a78d6", "fixed": "#52514e", "oracle-static": "#eda100"}
LAB = {"conformal-before": "Conformal, calibrated before the change",
       "conformal-after": "Conformal, calibrated after the change",
       "adaptive": "Adaptive (ours)", "oracle-static": "Best fixed cutoff in hindsight"}
NAMES = {"ciil": "Electrode shift (21 people)", "limbpos": "Arm position (12 people)",
         "uci": "UCI, later session (36)", "3dc": "3DC, later session (22)"}
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
R, F = Path("results"), Path("figures")
F.mkdir(exist_ok=True)
data = {d: pd.read_csv(R / f"drift_{d}.csv") for d in NAMES}

# 1. worst person's misfire rate vs target, every dataset
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=False)
for ax, (d, df) in zip(axes, data.items()):
    for m in ["conformal-before", "conformal-after", "adaptive"]:
        g = df[df.method == m].groupby("alpha").misfire.max()
        ax.plot(g.index, g.values, color=C[m], lw=2, marker="o", ms=5, label=LAB[m])
    ax.plot([0, 0.22], [0, 0.22], ls="--", color=MUTED, lw=1)
    ax.set_title(NAMES[d], loc="left", fontsize=10.5, color=INK)
    ax.set_xlim(0, 0.22); ax.set_ylim(0, max(0.25, df[df.method == "conformal-before"].misfire.max() * 1.05))
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Target misfire rate")
axes[0].set_ylabel("Worst person's misfire rate")
fig.suptitle("The adaptive cutoff holds every user near the target, even after electrode shift (dashed: target)",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(F / "drift_worst_person.png", dpi=160); plt.close(fig)

# 2. how often the hand moves at each target, every dataset
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
for ax, (d, df) in zip(axes, data.items()):
    for m in ["conformal-after", "oracle-static", "adaptive"]:
        g = df[df.method == m].groupby("alpha").active.mean()
        ax.plot(g.index, g.values, color=C[m], lw=2, marker="o", ms=5, label=LAB[m],
                ls=":" if m == "oracle-static" else "-")
    ax.set_title(NAMES[d], loc="left", fontsize=10.5, color=INK)
    ax.set_xlim(0, 0.22); ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Target misfire rate")
axes[0].set_ylabel("Decisions where the hand moves")
fig.suptitle("At low targets, the adaptive cutoff (with feedback) moves the hand most",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(F / "drift_activity.png", dpi=160); plt.close(fig)

# 3. one electrode-shift user over time
tr = json.loads((R / "drift_ciil_traces.json").read_text())
df = data["ciil"]
b = df[(df.method == "conformal-before") & np.isclose(df.alpha, 0.05)].set_index("person").misfire
p = str(int(b.idxmax()))
t = tr[p]
x = np.arange(len(t["run_before"])) * 4 * 0.05
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(x, t["run_before"], color=C["conformal-before"], lw=2, label=LAB["conformal-before"])
ax.plot(x, t["run_adaptive"], color=C["adaptive"], lw=2, label=LAB["adaptive"])
ax.axhline(0.05, ls="--", color=MUTED, lw=1)
ax.text(x[-1], 0.056, "target 5%", ha="right", color=MUTED, fontsize=9)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Seconds of use after a 45-degree armband shift")
ax.set_ylabel("Misfire rate so far")
ax.set_title(f"Electrode-shift participant {p}: the adaptive cutoff recovers within seconds",
             loc="left", fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(F / "drift_one_user.png", dpi=160); plt.close(fig)
print("saved drift figures; example participant", p)
