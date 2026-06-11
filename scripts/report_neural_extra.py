"""Enrichment stats + figures for the report: selectivity-index distributions,
reward-response latencies, and single- vs multi-unit selectivity.

Run:  python scripts/report_neural_extra.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "results")
OUT = os.path.join(RESULTS, "report")


def load_all():
    frames = []
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("neuron_summary_") and f.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(RESULTS, f)))
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_all()

    # significance flags (raw, per pooled — matches report aggregate uses FDR but raw ok for distributions)
    sig_rew = df["p_zeta2_rew_outcome"] < 0.05
    sig_cho = df["p_zeta2_choice_outcome"] < 0.05

    print("=== Selectivity index (SI) summary among significant neurons ===")
    for name, col, sig in [("reward (G+R vs G+N)", "SI_rew_outcome", sig_rew),
                           ("choice (G+R vs S+R)", "SI_choice_outcome", sig_cho)]:
        v = df.loc[sig, col].dropna()
        print(f"  {name}: n={len(v)} mean={v.mean():+.3f} median={v.median():+.3f} "
              f"frac|SI|>0.3={np.mean(np.abs(v) > 0.3):.2f}  "
              f"frac>0={np.mean(v > 0):.2f}")

    print("\n=== Reward-response latency (s) among reward-responsive neurons (p_zeta_reward<.05) ===")
    rr = df[df["p_zeta_reward"] < 0.05]
    lat = rr["latency_reward_s"].dropna()
    print(f"  n={len(lat)} median={lat.median():.3f}s  IQR=[{lat.quantile(.25):.3f}, {lat.quantile(.75):.3f}]")
    po = rr["peak_onset_reward_s"].dropna()
    print(f"  peak-onset: median={po.median():.3f}s  IQR=[{po.quantile(.25):.3f}, {po.quantile(.75):.3f}]")

    print("\n=== Single- vs multi-unit outcome selectivity (raw p<.05) ===")
    for ut in ["su", "mu"]:
        sub = df[df["unit_type"] == ut]
        pr = 100 * (sub["p_zeta2_rew_outcome"] < 0.05).mean()
        pc = 100 * (sub["p_zeta2_choice_outcome"] < 0.05).mean()
        print(f"  {ut}: n={len(sub)}  reward-sel={pr:.1f}%  choice-sel={pc:.1f}%")

    # Figure: SI histograms
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, col, sig, color) in zip(axes, [
        ("Reward selectivity (G+R vs G+N)", "SI_rew_outcome", sig_rew, "firebrick"),
        ("Choice selectivity (G+R vs S+R)", "SI_choice_outcome", sig_cho, "steelblue")]):
        v = df.loc[sig, col].dropna()
        ax.hist(v, bins=40, color=color, alpha=0.8)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(v.mean(), color="orange", lw=2, label=f"mean={v.mean():+.3f}")
        ax.set_title(f"{name}\n(n={len(v)} significant)", fontsize=10)
        ax.set_xlabel("Selectivity index  (A−B)/(A+B)")
        ax.set_ylabel("neurons")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Direction of outcome / choice coding is roughly balanced across the population", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_selectivity_index.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
