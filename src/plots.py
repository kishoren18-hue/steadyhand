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
          "conformal": "#1baf7a", "personal": "#eda100", "own-conformal": "#6250d6"}
LABELS = {"conformal+personal": "Conformal + personal check (ours)",
          "tuned": "Tuned on training people",
          "conformal": "Conformal only",
          "personal": "Personal calibration only",
          "own-conformal": "User-specific model + conformal"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def promise_plot(df, out, name=""):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = 0.24
    ylim = max(lim, df.groupby(["method", "alpha"]).misfire.mean().max() * 1.08)
    ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1.2)
    ax.text(0.236, 0.012, "below the dashed line\n= promise kept", color=MUTED,
            fontsize=9, ha="right", va="bottom")
    for m in ["tuned", "personal", "conformal", "conformal+personal", "own-conformal"]:
        if m not in set(df.method):
            continue
        g = df[df.method == m].groupby("alpha")["misfire"].mean()
        ax.plot(g.index, g.values, color=COLORS[m], lw=2, marker="o", ms=7,
                label=LABELS[m])
    ax.set(xlim=(0, lim), ylim=(0, ylim), xlabel="Target misfire rate",
           ylabel="Actual misfire rate on unseen people (mean)")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_title(f"Which cutoffs keep their promise on new users?{name}",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def per_person_plot(df, alpha, out, name=""):
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
    ax.set_title(f"Every person, at a {alpha:.0%} target{name}", loc="left",
                 color=INK, fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def tradeoff_plot(df, out, name=""):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for m in ["tuned", "conformal", "conformal+personal", "own-conformal"]:
        if m not in set(df.method):
            continue
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
    ax.set_title(f"Acting more without a dangerous worst case{name}",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    figs = Path("figures"); figs.mkdir(exist_ok=True)
    names = {"uci": "UCI", "3dc": "3DC"}
    for f in sorted(Path(a.out).glob("*_per_person.csv")):
        if f.name.startswith("old_"):
            continue
        ds, model = f.name.split("_")[:2]
        df = pd.read_csv(f)
        if "vote" in df.columns:
            df = df[df.vote == 1]
        label = f" ({names.get(ds, ds)}, {model})"
        promise_plot(df, figs / f"promise_{ds}_{model}.png", label)
        per_person_plot(df, 0.05, figs / f"per_person_{ds}_{model}.png", label)
        tradeoff_plot(df, figs / f"tradeoff_{ds}_{model}.png", label)
        print("figures for", ds, model)
