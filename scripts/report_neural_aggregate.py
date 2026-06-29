"""Aggregate the per-session neuron_summary CSVs into report-level tables + figures.

Pools all neurons across the 8 sessions and quantifies:
  1. Event responsiveness: fraction of neurons with one-sample ZETA p<.05
     (raw and Benjamini-Hochberg FDR), overall and per region, per event.
  2. Outcome selectivity: fraction with two-sample ZETA p<.05 for the
     reward (G+R vs G+N) and choice (G+R vs S+R) contrasts, per region.
  3. Direction of the effect: preference breakdown (rewarded vs non-rewarded,
     gamble vs safe) among significant neurons; mean selectivity index.
  4. Overlap: are reward-responsive neurons also outcome-selective?

Writes:
  results/report/neural_responsiveness.csv
  results/report/neural_outcome.csv
  results/report/neural_region_event_pct.csv
  results/report/fig_responsiveness_by_region.png
  results/report/fig_outcome_by_region.png

Run:  python scripts/report_neural_aggregate.py
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
ALPHA = 0.05

EVENTS = ["trial_start", "cue", "cue_to_reward", "reward"]
CONTRASTS = {"rew_outcome": "Reward (G+R vs G+N)", "choice_outcome": "Choice (G+R vs S+R)"}
REGIONS = ["MFG", "IFG", "SMG", "AG"]


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns boolean array of rejections at ALPHA."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    rej = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return rej
    pp = p[idx]
    order = np.argsort(pp)
    m = len(pp)
    thresh = (np.arange(1, m + 1) / m) * ALPHA
    passed = pp[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        sig_sorted = np.zeros(m, dtype=bool)
        sig_sorted[: kmax + 1] = True
        back = np.empty(m, dtype=bool)
        back[order] = sig_sorted
        rej[idx] = back
    return rej


def load_all():
    frames = []
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("neuron_summary_") and f.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(RESULTS, f)))
    df = pd.concat(frames, ignore_index=True)
    # FDR per (session, event/contrast) — multiple comparisons live within a session test
    for ev in EVENTS:
        col = f"p_zeta_{ev}"
        if col in df.columns:
            sig = np.zeros(len(df), dtype=bool)
            for sid, g in df.groupby("session"):
                sig[g.index] = bh_fdr(g[col].values)
            df[f"sig_fdr_{ev}"] = sig
            df[f"sig_raw_{ev}"] = df[col] < ALPHA
    for ct in CONTRASTS:
        col = f"p_zeta2_{ct}"
        if col in df.columns:
            sig = np.zeros(len(df), dtype=bool)
            for sid, g in df.groupby("session"):
                sig[g.index] = bh_fdr(g[col].values)
            df[f"sig_fdr_{ct}"] = sig
            df[f"sig_raw_{ct}"] = df[col] < ALPHA
    return df


def responsiveness_table(df):
    rows = []
    n_tot = len(df)
    for ev in EVENTS:
        raw = df[f"sig_raw_{ev}"].sum()
        fdr = df[f"sig_fdr_{ev}"].sum()
        rows.append(dict(event=ev, n=n_tot,
                         n_sig_raw=int(raw), pct_raw=round(100 * raw / n_tot, 1),
                         n_sig_fdr=int(fdr), pct_fdr=round(100 * fdr / n_tot, 1)))
    return pd.DataFrame(rows)


def region_event_table(df):
    rows = []
    for region in REGIONS:
        sub = df[df["region"] == region]
        if len(sub) == 0:
            continue
        row = dict(region=region, n_neurons=len(sub))
        for ev in EVENTS:
            row[f"pct_{ev}_fdr"] = round(100 * sub[f"sig_fdr_{ev}"].sum() / len(sub), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_table(df):
    rows = []
    n_tot = len(df)
    for ct, lbl in CONTRASTS.items():
        raw = df[f"sig_raw_{ct}"].sum()
        fdr = df[f"sig_fdr_{ct}"].sum()
        sub_sig = df[df[f"sig_fdr_{ct}"]]
        pref_col = f"pref_{ct}"
        pref_counts = sub_sig[pref_col].value_counts().to_dict() if pref_col in df.columns else {}
        si_col = f"SI_{ct}"
        mean_si = round(float(sub_sig[si_col].mean()), 3) if (si_col in df.columns and len(sub_sig)) else np.nan
        rows.append(dict(contrast=ct, label=lbl, n=n_tot,
                         n_sig_raw=int(raw), pct_raw=round(100 * raw / n_tot, 1),
                         n_sig_fdr=int(fdr), pct_fdr=round(100 * fdr / n_tot, 1),
                         pref_breakdown=str(pref_counts), mean_SI_sig=mean_si))
    return pd.DataFrame(rows)


def outcome_by_region(df):
    rows = []
    for region in REGIONS:
        sub = df[df["region"] == region]
        if len(sub) == 0:
            continue
        row = dict(region=region, n_neurons=len(sub))
        for ct in CONTRASTS:
            row[f"pct_{ct}_fdr"] = round(100 * sub[f"sig_fdr_{ct}"].sum() / len(sub), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_stats(df):
    """Are reward-responsive neurons more likely to be reward-outcome selective?"""
    resp = df["sig_fdr_reward"]
    sel = df["sig_fdr_rew_outcome"]
    a = int((resp & sel).sum())
    b = int((resp & ~sel).sum())
    c = int((~resp & sel).sum())
    d = int((~resp & ~sel).sum())
    p_sel_if_resp = a / (a + b) if (a + b) else np.nan
    p_sel_if_not = c / (c + d) if (c + d) else np.nan
    return dict(reward_resp_and_outcome_sel=a, reward_resp_not_sel=b,
               not_resp_but_sel=c, neither=d,
               P_sel_given_resp=round(p_sel_if_resp, 3),
               P_sel_given_notresp=round(p_sel_if_not, 3))


def fig_responsiveness(region_tab):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(REGIONS))
    w = 0.2
    colors = ["#888", "royalblue", "seagreen", "darkorange"]
    for i, ev in enumerate(EVENTS):
        vals = [region_tab.loc[region_tab.region == r, f"pct_{ev}_fdr"].values[0]
                if r in region_tab.region.values else 0 for r in REGIONS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=ev, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(REGIONS)
    ax.set_ylabel("% neurons responsive (FDR q<.05)")
    ax.set_title("Event responsiveness by brain region (pooled across 8 sessions)")
    ax.legend(title="aligned to", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_responsiveness_by_region.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_outcome(oreg):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGIONS))
    w = 0.35
    for i, (ct, lbl) in enumerate(CONTRASTS.items()):
        vals = [oreg.loc[oreg.region == r, f"pct_{ct}_fdr"].values[0]
                if r in oreg.region.values else 0 for r in REGIONS]
        ax.bar(x + (i - 0.5) * w, vals, w, label=lbl,
               color=["firebrick", "steelblue"][i])
    ax.set_xticks(x); ax.set_xticklabels(REGIONS)
    ax.set_ylabel("% neurons outcome-selective (FDR q<.05)")
    ax.set_title("Outcome / choice selectivity by region (reward-aligned)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_outcome_by_region.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_reward_selectivity_pref(df):
    """Stacked bar: % reward-selective neurons per region, split by preference direction."""
    if "pref_rew_outcome" not in df.columns:
        print("  [skip] pref_rew_outcome column not found")
        return None

    regions_present = [r for r in REGIONS if r in df["region"].values]
    pct_rew, pct_nrew, n_tots = [], [], []
    for r in regions_present:
        sub = df[df["region"] == r]
        sig = sub[sub["sig_fdr_rew_outcome"]]
        n = len(sub)
        n_tots.append(n)
        pct_rew.append(100 * (sig["pref_rew_outcome"] == "G+R").sum() / n)
        pct_nrew.append(100 * (sig["pref_rew_outcome"] == "G+N").sum() / n)

    x = np.arange(len(regions_present))
    w = 0.55
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, pct_rew, w, label="Reward-preferring (G+R)", color="firebrick", alpha=0.85)
    ax.bar(x, pct_nrew, w, bottom=pct_rew, label="No-reward-preferring (G+N)",
           color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(regions_present)
    ax.set_ylabel("% of all neurons (FDR q<.05)")
    ax.set_title("Reward selectivity by region & preference\n(G+R vs G+N, reward-aligned)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (r, n) in enumerate(zip(pct_rew, n_tots)):
        total_bar = r + pct_nrew[i]
        ax.text(x[i], total_bar + 0.4, f"n={n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_reward_selectivity_pref_by_region.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_reward_selectivity(oreg):
    regions_present = [r for r in REGIONS if r in oreg.region.values]
    vals = [oreg.loc[oreg.region == r, "pct_rew_outcome_fdr"].values[0]
            for r in regions_present]
    ns = [oreg.loc[oreg.region == r, "n_neurons"].values[0]
          for r in regions_present]

    n_sig = [round(v * n / 100) for v, n in zip(vals, ns)]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(regions_present))
    bars = ax.bar(x, vals, 0.55, color="firebrick", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(n)})" for r, n in zip(regions_present, ns)])
    ax.set_ylabel("% neurons reward-selective (FDR q<.05)")
    ax.set_ylim(0, min(100, max(vals) * 1.18))
    ax.set_title("Reward outcome selectivity by region\n(G+R vs G+N, reward-aligned)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v, ns_i, nt in zip(bars, vals, n_sig, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}%\n{int(ns_i)}/{int(nt)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_reward_selectivity_by_region.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_all()
    print(f"Pooled neurons: {len(df)}  across {df['session'].nunique()} sessions")
    print(f"Region counts: {df['region'].value_counts().to_dict()}")
    print(f"Unit types: {df['unit_type'].value_counts().to_dict()}")

    resp = responsiveness_table(df)
    print("\n=== EVENT RESPONSIVENESS (pooled, n={}) ===".format(len(df)))
    print(resp.to_string(index=False))
    resp.to_csv(os.path.join(OUT, "neural_responsiveness.csv"), index=False)

    rtab = region_event_table(df)
    print("\n=== RESPONSIVENESS BY REGION (% FDR q<.05) ===")
    print(rtab.to_string(index=False))
    rtab.to_csv(os.path.join(OUT, "neural_region_event_pct.csv"), index=False)

    otab = outcome_table(df)
    print("\n=== OUTCOME / CHOICE SELECTIVITY (pooled) ===")
    print(otab.to_string(index=False))
    otab.to_csv(os.path.join(OUT, "neural_outcome.csv"), index=False)

    oreg = outcome_by_region(df)
    print("\n=== OUTCOME SELECTIVITY BY REGION (% FDR q<.05) ===")
    print(oreg.to_string(index=False))
    oreg.to_csv(os.path.join(OUT, "neural_outcome_by_region.csv"), index=False)

    ov = overlap_stats(df)
    print("\n=== OVERLAP: reward-responsive vs reward-outcome-selective ===")
    for k, v in ov.items():
        print(f"  {k}: {v}")

    p1 = fig_responsiveness(rtab)
    p2 = fig_outcome(oreg)
    p3 = fig_reward_selectivity(oreg)
    p4 = fig_reward_selectivity_pref(df)
    paths = [p for p in [p1, p2, p3, p4] if p]
    print("\nSaved figures -> " + "\n              -> ".join(paths))


if __name__ == "__main__":
    main()
