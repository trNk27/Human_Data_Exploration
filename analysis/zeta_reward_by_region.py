"""Percentage of ZETA reward-responsive neurons per region.

Pools the per-session reward-aligned ZETA CSVs in
``results/zeta_responsiveness/zeta_reward_<session>.csv`` (one row per neuron,
column ``p_zeta``), parses the region out of each neuron's label, and plots — one
bar per region — the fraction of neurons that respond significantly to reward.

This is the "general responsiveness" companion to the reward *selectivity* by
region figure (``scripts/report_neural_aggregate.py`` →
``fig_reward_selectivity_by_region.png``): that one asks "does the response differ
between G+R and G+N?", this one just asks "does the neuron respond to reward at
all?". Significance is reported both raw (p_zeta < alpha) and
Benjamini-Hochberg FDR-corrected (q); the bars use the FDR call to match the
selectivity figure's convention.

Run the ZETA reward test first if the CSVs are missing:
    python analysis/zeta_analysis.py --event reward --csv --save   (per session)

Usage
-----
    python -m analysis.zeta_reward_by_region [--alpha 0.05] [--q 0.05]
                                             [--min-n 5] [--raw] [--out PATH]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils import RESULTS_SUBDIRS
from analysis.responsive_region import parse_region, load_event


def bh_reject(pvals, q):
    """Benjamini-Hochberg FDR: boolean reject mask at level ``q``."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[np.max(np.where(below)[0])]
    return p <= cutoff


def region_table(df, alpha, q, min_n):
    """Per-region n, raw- and FDR-significant counts and percentages."""
    df = df.copy()
    df["sig_raw"] = df["p_zeta"] < alpha
    df["sig_fdr"] = bh_reject(df["p_zeta"].to_numpy(), q)

    g = df.groupby("region")
    tab = pd.DataFrame({
        "n_neurons": g.size(),
        "n_sig_raw": g["sig_raw"].sum(),
        "n_sig_fdr": g["sig_fdr"].sum(),
    })
    tab["pct_raw"] = (100 * tab["n_sig_raw"] / tab["n_neurons"]).round(1)
    tab["pct_fdr"] = (100 * tab["n_sig_fdr"] / tab["n_neurons"]).round(1)
    _ORDER = ["MFG", "IFG", "SMG", "AG"]
    tab = tab[tab["n_neurons"] >= min_n]
    tab = tab.loc[[r for r in _ORDER if r in tab.index]]
    return tab


def plot(tab, alpha, q, use_raw, total, out_path):
    pct_col = "pct_raw" if use_raw else "pct_fdr"
    n_col = "n_sig_raw" if use_raw else "n_sig_fdr"
    crit = f"raw p < {alpha}" if use_raw else f"FDR q < {q}"

    regions = list(tab.index)
    vals = tab[pct_col].to_numpy()
    x = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, vals, 0.55, color="firebrick", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(tab.loc[r, 'n_neurons'])})" for r in regions])
    ax.set_ylabel(f"% neurons reward-responsive ({crit})")
    ax.set_ylim(0, min(100, max(vals) * 1.18))
    ax.set_title("ZETA reward responsiveness by region\n"
                 f"(reward-aligned, {total} neurons pooled across 8 sessions)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, r in zip(bars, regions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{tab.loc[r, pct_col]:.1f}%\n{int(tab.loc[r, n_col])}/{int(tab.loc[r, 'n_neurons'])}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    return fig


def run(alpha=0.05, q=0.05, min_n=5, use_raw=False, out=None):
    df = load_event("reward")           # pools zeta_reward_<session>.csv, adds `region`
    if df is None:
        raise SystemExit("No zeta_reward_<session>.csv files found in "
                         f"{RESULTS_SUBDIRS['responsiveness']} — run the ZETA reward test first.")
    total = len(df)
    tab = region_table(df, alpha, q, min_n)

    n_raw = int((df["p_zeta"] < alpha).sum())
    n_fdr = int(bh_reject(df["p_zeta"].to_numpy(), q).sum())
    print(f"\n=== ZETA reward responsiveness (pooled, {total} neurons, 8 sessions) ===")
    print(f"Raw p < {alpha}:   {n_raw}  ({100*n_raw/total:.1f}%)")
    print(f"FDR q < {q}:   {n_fdr}  ({100*n_fdr/total:.1f}%)")
    print("\nPer region:")
    print(tab.to_string())

    out = out or os.path.join(RESULTS_SUBDIRS["responsiveness"],
                              "zeta_reward_responsive_by_region.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot(tab, alpha, q, use_raw, total, out)
    tab.to_csv(os.path.splitext(out)[0] + ".csv")
    print(f"Saved summary  -> {os.path.splitext(out)[0] + '.csv'}")
    return tab


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ZETA reward responsiveness % by region (bar plot)")
    p.add_argument("--alpha", type=float, default=0.05, help="Raw p_zeta threshold (default 0.05)")
    p.add_argument("--q", type=float, default=0.05, help="Benjamini-Hochberg FDR level (default 0.05)")
    p.add_argument("--min-n", type=int, default=5, help="Drop regions with fewer pooled neurons (default 5)")
    p.add_argument("--raw", action="store_true",
                   help="Plot raw p<alpha bars instead of FDR-corrected (default: FDR)")
    p.add_argument("--out", default=None, help="Override output PNG path")
    args = p.parse_args()
    run(alpha=args.alpha, q=args.q, min_n=args.min_n, use_raw=args.raw, out=args.out)
