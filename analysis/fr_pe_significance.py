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


_REGION_ORDER = ["MFG", "IFG", "SMG", "AG"]


def plot(tab: pd.DataFrame, alpha: float, window: str, total: int, out_path: str):
    regions = [r for r in _REGION_ORDER if r in tab.index and r != "ALL"]
    n_region = tab.loc[regions, "n_neurons"].to_numpy()
    x = np.arange(len(regions))
    w = 0.8 / len(SIG_CATS)

    fig, ax = plt.subplots(figsize=(max(12, 2.7 * len(regions) + 5), 6))
    for i, cat in enumerate(SIG_CATS):
        color, hatch = _STYLE[cat]
        counts = tab.loc[regions, cat].to_numpy()
        pcts   = 100 * counts / n_region
        bars = ax.bar(x + (i - (len(SIG_CATS) - 1) / 2) * w, pcts, w,
                      color=color, hatch=hatch, edgecolor="white", linewidth=0.6)
        ax.bar_label(bars,
                     labels=[f"{p:.1f}%\n{int(c)}/{int(n)}"
                             for p, c, n in zip(pcts, counts, n_region)],
                     fontsize=6.5, padding=2, linespacing=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(tab.loc[r, 'n_neurons'])})" for r in regions])
    ax.set_ylabel(f"% of neurons in region (p < {alpha})")
    ax.set_title(f"FR ~ RW prediction error by region & sign  (p < {alpha})\n"
                 f"window: {window}  ·  {total} neurons total  ·  "
                 f"hatch = negative r / discordant", fontsize=10)
    legend_handles = [Patch(facecolor=_STYLE[c][0], hatch=_STYLE[c][1],
                            edgecolor="white", label=c) for c in SIG_CATS]
    ax.legend(handles=legend_handles, title="significant in", frameon=False,
              ncol=3, loc="upper right", fontsize=8)
    ax.margins(y=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    return fig


def plot_simple(tab: pd.DataFrame, alpha: float, window: str, total: int,
                out_path: str):
    """Single bar per region: % neurons with ANY significant PE correlation."""
    regions = [r for r in _REGION_ORDER if r in tab.index and r != "ALL"]
    if not regions:
        return None
    n_any = tab.loc[regions, "any_sig"].to_numpy()
    n_tot = tab.loc[regions, "n_neurons"].to_numpy()
    pcts  = 100 * n_any / n_tot
    x     = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, pcts, 0.55, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(n)})" for r, n in zip(regions, n_tot)])
    ax.set_ylabel(f"% neurons FR ~ RW prediction error (p < {alpha})")
    ax.set_ylim(0, min(100, max(pcts) * 1.18))
    ax.set_title(f"FR ~ RW prediction error by region\n"
                 f"(window: {window},  {total} neurons total)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, pct, ns, nt in zip(bars, pcts, n_any, n_tot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n{int(ns)}/{int(nt)}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    return fig


_BUCKETS = [
    ("Reward only",    ["reward only (+)", "reward only (-)"],       "darkorange"),
    ("No-reward only", ["no-reward only (+)", "no-reward only (-)"], "firebrick"),
    ("Both",           ["both concordant", "both discordant"],       "rebeccapurple"),
]


def plot_unsigned(tab: pd.DataFrame, alpha: float, window: str, total: int,
                  out_path: str):
    """3-bucket grouped bar: reward only / no-reward only / both, % of region."""
    regions = [r for r in _REGION_ORDER if r in tab.index and r != "ALL"]
    if not regions:
        return None
    n_region = tab.loc[regions, "n_neurons"].to_numpy()
    x = np.arange(len(regions))
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, cats, color) in enumerate(_BUCKETS):
        counts = sum(tab.loc[regions, c].to_numpy() for c in cats)
        pcts   = 100 * counts / n_region
        bars   = ax.bar(x + (i - 1) * w, pcts, w, color=color, alpha=0.85,
                        label=label)
        for bar, pct, cnt, n in zip(bars, pcts, counts, n_region):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{pct:.1f}%\n{int(cnt)}/{int(n)}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(n)})" for r, n in zip(regions, n_region)])
    ax.set_ylabel(f"% of neurons in region (p < {alpha})")
    all_pcts = np.concatenate([
        100 * sum(tab.loc[regions, c].to_numpy() for c in cats) / n_region
        for _, cats, _ in _BUCKETS
    ])
    ax.set_ylim(0, min(100, float(all_pcts.max()) * 1.35))
    ax.set_title(f"FR ~ RW prediction error by region & condition  (p < {alpha})\n"
                 f"window: {window}  ·  {total} neurons total")
    ax.legend(fontsize=8, frameon=False)
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
    print(f"Significant in G+R:          {pc(int(df['sig_reward'].sum()))}   [chance ~= {alpha*total:.0f}]")
    print(f"Significant in G+NR:         {pc(int(df['sig_noreward'].sum()))}   [chance ~= {alpha*total:.0f}]")
    print(f"Significant in ANY:          {pc(n_any)}")
    print(f"  reward only (+):           {pc(int(cat_counts.get('reward only (+)', 0)))}")
    print(f"  reward only (-):           {pc(int(cat_counts.get('reward only (-)', 0)))}")
    print(f"  no-reward only (+):        {pc(int(cat_counts.get('no-reward only (+)', 0)))}")
    print(f"  no-reward only (-):        {pc(int(cat_counts.get('no-reward only (-)', 0)))}")
    print(f"  both concordant:           {pc(n_both_c)}")
    print(f"  both discordant:           {pc(n_both_d)}")
    print(f"  (both total:               {pc(n_both_c + n_both_d)}; "
          f"chance if independent ~= {alpha*alpha*total:.1f})")

    print("\nPer-region counts:")
    print(tab.to_string())
    print("\nPer-region % of total neurons:")
    print((100 * tab[ALL_CATS] / total).round(1).to_string())

    out = out or os.path.join(RESULTS_DIR, "rw",
                              f"fr_vs_rw_pe_significance_by_region_signed_{window}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot(tab, alpha, window, total, out)

    simple_out = os.path.join(os.path.dirname(out),
                              f"fr_vs_rw_pe_any_sig_by_region_{window}.png")
    plot_simple(tab, alpha, window, total, simple_out)

    unsigned_out = os.path.join(os.path.dirname(out),
                                f"fr_vs_rw_pe_by_condition_by_region_{window}.png")
    plot_unsigned(tab, alpha, window, total, unsigned_out)

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
