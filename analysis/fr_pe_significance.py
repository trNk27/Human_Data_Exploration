"""Significance summary of the per-neuron firing-rate vs RW prediction-error table.

Reads ``results/rw/fr_vs_rw_pe_correlations_<window>.csv`` (from
``analysis.fr_pe_correlation``) and asks, for each neuron, whether its firing rate
correlates significantly (p < alpha) with the RW prediction error δ = outcome − Q
in the rewarded (G+R) and/or unrewarded (G+NR) gamble condition — and with which
sign. Each neuron falls into one of seven buckets:

  * reward only (+)/(-)     — significant in G+R only; sign of that correlation
  * no-reward only (+)/(-)  — significant in G+NR only; sign of that correlation
  * both concordant         — significant in both, SAME sign (consistent monotonic
                              coding of the signed prediction error)
  * both discordant         — significant in both, OPPOSITE signs (outcome-specific
                              / unsigned-salience-like)
  * neither

Prints overall + per-region counts AND percentages of the total neuron count, and
saves a grouped bar chart (six significant buckets per region, sign shown by hatch)
plus a tidy summary CSV.

Usage
-----
    python -m analysis.fr_pe_significance [--window reward_to_end] [--alpha 0.05]
                                          [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils import RESULTS_DIR

# Significant buckets, in plot/table order. Sign shown by hatch (solid = +/concordant).
SIG_CATS = [
    "reward only (+)", "reward only (-)",
    "no-reward only (+)", "no-reward only (-)",
    "both concordant", "both discordant",
]
ALL_CATS = SIG_CATS + ["neither"]

# (face colour, hatch) — colour = condition (orange G+R, red G+NR, purple both);
# hatch = the "inverted" case (negative r, or discordant signs).
_STYLE = {
    "reward only (+)":    ("darkorange",   None),
    "reward only (-)":    ("darkorange",   "////"),
    "no-reward only (+)": ("firebrick",    None),
    "no-reward only (-)": ("firebrick",    "////"),
    "both concordant":    ("rebeccapurple", None),
    "both discordant":    ("rebeccapurple", "////"),
}


def classify(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Add significance flags and a 7-level signed ``category`` column."""
    df = df.copy()
    sr = df["p_gamble_rewarded"].lt(alpha).fillna(False).to_numpy()
    sn = df["p_gamble_unrewarded"].lt(alpha).fillna(False).to_numpy()
    rr = df["r_gamble_rewarded"].to_numpy()
    rn = df["r_gamble_unrewarded"].to_numpy()
    same_sign = np.sign(rr) == np.sign(rn)

    df["sig_reward"] = sr
    df["sig_noreward"] = sn
    df["category"] = np.select(
        [
            sr & sn & same_sign,
            sr & sn & ~same_sign,
            sr & ~sn & (rr > 0),
            sr & ~sn & (rr < 0),
            ~sr & sn & (rn > 0),
            ~sr & sn & (rn < 0),
        ],
        ["both concordant", "both discordant",
         "reward only (+)", "reward only (-)",
         "no-reward only (+)", "no-reward only (-)"],
        default="neither",
    )
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region counts for each bucket (+ n_neurons, any_sig, and ALL row)."""
    tab = (df.groupby("region")["category"].value_counts().unstack(fill_value=0)
             .reindex(columns=ALL_CATS, fill_value=0))
    tab["n_neurons"] = tab.sum(axis=1)
    tab["any_sig"] = tab["n_neurons"] - tab["neither"]
    tab = tab.sort_values("n_neurons", ascending=False)
    tab.loc["ALL"] = tab.sum(axis=0)
    return tab


def plot(tab: pd.DataFrame, alpha: float, window: str, total: int, out_path: str):
    regions = [r for r in tab.index if r != "ALL"]
    x = np.arange(len(regions))
    w = 0.8 / len(SIG_CATS)

    fig, ax = plt.subplots(figsize=(max(12, 2.7 * len(regions) + 5), 6))
    for i, cat in enumerate(SIG_CATS):
        color, hatch = _STYLE[cat]
        vals = tab.loc[regions, cat].to_numpy()
        bars = ax.bar(x + (i - (len(SIG_CATS) - 1) / 2) * w, vals, w,
                      color=color, hatch=hatch, edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, labels=[f"{int(v)}\n{100*v/total:.1f}%" for v in vals],
                     fontsize=6.5, padding=2, linespacing=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(tab.loc[r, 'n_neurons'])})" for r in regions])
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"Neurons with firing rate ~ RW prediction error (p < {alpha})\n"
                 f"window: {window}  ·  {total} neurons total  ·  "
                 f"% labels are of the {total}-neuron total  ·  "
                 f"hatch = negative r / discordant", fontsize=10)
    legend_handles = [Patch(facecolor=_STYLE[c][0], hatch=_STYLE[c][1],
                            edgecolor="white", label=c) for c in SIG_CATS]
    ax.legend(handles=legend_handles, title="significant in", frameon=False,
              ncol=3, loc="upper right", fontsize=8)
    ax.margins(y=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    return fig


def _tidy(tab: pd.DataFrame, total: int) -> pd.DataFrame:
    """Long-form summary: region × category with count, % of total, % of region."""
    rows = []
    for region in tab.index:
        n_region = int(tab.loc[region, "n_neurons"])
        for cat in ALL_CATS:
            c = int(tab.loc[region, cat])
            rows.append({
                "region": region, "category": cat, "count": c,
                "pct_of_total": round(100 * c / total, 2),
                "pct_of_region": round(100 * c / n_region, 2) if n_region else 0.0,
            })
    return pd.DataFrame(rows)


def run(window="reward_to_end", alpha=0.05, csv=None, out=None):
    csv = csv or os.path.join(RESULTS_DIR, "rw", f"fr_vs_rw_pe_correlations_{window}.csv")
    df = pd.read_csv(csv)
    total = len(df)
    df = classify(df, alpha)
    tab = summarise(df)

    def pc(n):  # count + percent-of-total
        return f"{n} ({100*n/total:.1f}%)"

    cat_counts = df["category"].value_counts()
    n_both_c = int(cat_counts.get("both concordant", 0))
    n_both_d = int(cat_counts.get("both discordant", 0))
    n_any = int((df["category"] != "neither").sum())

    print(f"\n=== FR ~ RW prediction error significance (p < {alpha}, window={window}) ===")
    print(f"Total neurons:               {total}")
    print(f"Significant in G+R:          {pc(int(df['sig_reward'].sum()))}   [chance ≈ {alpha*total:.0f}]")
    print(f"Significant in G+NR:         {pc(int(df['sig_noreward'].sum()))}   [chance ≈ {alpha*total:.0f}]")
    print(f"Significant in ANY:          {pc(n_any)}")
    print(f"  reward only (+):           {pc(int(cat_counts.get('reward only (+)', 0)))}")
    print(f"  reward only (-):           {pc(int(cat_counts.get('reward only (-)', 0)))}")
    print(f"  no-reward only (+):        {pc(int(cat_counts.get('no-reward only (+)', 0)))}")
    print(f"  no-reward only (-):        {pc(int(cat_counts.get('no-reward only (-)', 0)))}")
    print(f"  both concordant:           {pc(n_both_c)}")
    print(f"  both discordant:           {pc(n_both_d)}")
    print(f"  (both total:               {pc(n_both_c + n_both_d)}; "
          f"chance if independent ≈ {alpha*alpha*total:.1f})")

    print("\nPer-region counts:")
    print(tab.to_string())
    print("\nPer-region % of total neurons:")
    print((100 * tab[ALL_CATS] / total).round(1).to_string())

    out = out or os.path.join(RESULTS_DIR, "rw",
                              f"fr_vs_rw_pe_significance_by_region_signed_{window}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot(tab, alpha, window, total, out)

    summ_csv = os.path.splitext(out)[0] + ".csv"
    _tidy(tab, total).to_csv(summ_csv, index=False)
    print(f"Saved summary  -> {summ_csv}")
    return df, tab


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Signed significance summary + per-region barplot of FR~RW-PE correlations")
    p.add_argument("--window", default="reward_to_end",
                   help="Window tag identifying the correlation CSV (default: reward_to_end)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default: 0.05)")
    p.add_argument("--csv", default=None, help="Override input CSV path")
    p.add_argument("--out", default=None, help="Override output PNG path")
    args = p.parse_args()
    run(window=args.window, alpha=args.alpha, csv=args.csv, out=args.out)
