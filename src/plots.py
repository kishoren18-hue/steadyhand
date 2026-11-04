"""Make the README figures from results/per_person_<model>.csv."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (validated palette), text in ink colors
COLORS = {"conformal+personal": "#2a78d6", "tuned": "#eb6834",
          "conformal": "#1baf7a", "personal": "#eda100"}
LABELS = {"conformal+personal": "Conformal + personal check (ours)",
          "tuned": "Tuned on training people",
          "conformal": "Conformal only",
          "personal": "Personal calibration only"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def promise_plot(df, out):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = 0.24
    ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1.2)
    ax.text(0.235, 0.12, "below the dashed line\n= promise kept", color=MUTED,
            fontsize=9, ha="right", va="top")
    for m in ["tuned", "personal", "conformal", "conformal+personal"]:
        g = df[df.method == m].groupby("alpha")["misfire"].mean()
        ax.plot(g.index, g.values, color=COLORS[m], lw=2, marker="o", ms=7,
                label=LABELS[m])
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Target misfire rate",
           ylabel="Actual misfire rate on unseen people (mean)")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_title("Which thresholds keep their promise on new users?",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def per_person_plot(df, alpha, out):
    methods = ["tuned", "personal", "conformal", "conformal+personal"]
    fixed = df[df.method == "fixed"]["misfire"].values
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rng = np.random.default_rng(0)
    short = {"tuned": "Tuned on\ntraining people", "personal": "Personal\nonly",
             "conformal": "Conformal\nonly",
             "conformal+personal": "Conformal +\npersonal (ours)"}
    cols = [("fixed", fixed, MUTED, "LibEMG default\n(fixed 0.9)")]
    for m in methods:
        v = df[(df.method == m) & np.isclose(df.alpha, alpha)]["misfire"].values
        cols.append((m, v, COLORS[m], short[m]))
    for i, (m, v, c, lab) in enumerate(cols):
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=34, color=c, edgecolor="#fcfcfb", lw=1, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.max()] * 2, color=INK, lw=1.4, zorder=4)
        ax.text(i + 0.31, v.max(), f"worst {v.max():.0%}", va="center",
                fontsize=8.5, color=INK)
    ax.axhline(alpha, ls="--", color=MUTED, lw=1.2)
    ax.text(-0.45, alpha + 0.006, f"target {alpha:.0%}", ha="left",
            color=MUTED, fontsize=9)
    ax.set_xticks(range(len(cols)), [c[3] for c in cols], fontsize=8.5)
    ax.set_xlim(-0.5, len(cols) - 0.2)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Misfire rate (one dot per unseen person)")
    ax.set_title(f"Every person, at a {alpha:.0%} target", loc="left",
                 color=INK, fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def tradeoff_plot(df, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for m in ["tuned", "conformal", "conformal+personal"]:
        g = df[df.method == m].groupby("alpha").agg(
            worst=("misfire", "max"), active=("active", "mean"))
        ax.plot(g.active, g.worst, color=COLORS[m], lw=2, marker="o", ms=7,
                label=LABELS[m])
    f = df[df.method == "fixed"]
    ax.scatter(f.active.mean(), f.misfire.max(), s=90, marker="D",
               color=MUTED, zorder=4, label="LibEMG default (fixed 0.9)")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set(xlabel="How often the hand acts (mean over people)",
           ylabel="Worst person's misfire rate")
    ax.set_title("Acting more without a dangerous worst case",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LDA")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = pd.read_csv(Path(a.out) / f"per_person_{a.model}.csv")
    figs = Path("figures"); figs.mkdir(exist_ok=True)
    promise_plot(df, figs / f"promise_{a.model}.png")
    per_person_plot(df, 0.05, figs / f"per_person_{a.model}.png")
    tradeoff_plot(df, figs / f"tradeoff_{a.model}.png")
    print("saved figures to", figs)
