"""Add direction-of-effect columns to the existing zeta2 CSVs.

For each neuron, anchor a window of width 2*W around its zeta_t_s latency,
clip to [0, dur], and compute the mean firing rate per condition in that
window from the raw spike data. Selectivity index:

    SI = (r_a - r_b) / (r_a + r_b)

Reward contrast (a=G+R, b=G+N):   SI > 0 -> prefers rewarded
Choice contrast (a=G+R, b=S+R):   SI > 0 -> prefers gamble

The script appends four columns to each results/zeta_outcome/zeta2_<contrast>_<session>.csv
in place: rate_<a>, rate_<b>, SI, preference. Untested / skipped neurons
(zeta_t_s NaN) get NaN. Across all sessions it also writes a per-contrast
PNG with the SI distribution of significant neurons and a binomial test for
skew vs 50/50 to results/direction/.

Usage:
    python outcome_direction.py
    python outcome_direction.py --window-ms 200
    python outcome_direction.py --sessions 20250521 20250602 --alpha 0.01
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from session import Session
from utils import (
    RESULTS_SUBDIRS, REPO_ROOT,
)

HERE         = Path(REPO_ROOT)
OUTCOME_DIR  = Path(RESULTS_SUBDIRS["outcome"])
DIRECTION_DIR = Path(RESULTS_SUBDIRS["direction"])

DEFAULT_DUR_S     = 2.0
DEFAULT_WINDOW_MS = 300
DEFAULT_ALPHA     = 0.05

CONTRASTS = {
    "reward": {"a": "G+R", "b": "G+N",
               "pref_pos": "rewarded", "pref_neg": "non-rewarded"},
    "choice": {"a": "G+R", "b": "S+R",
               "pref_pos": "gamble",   "pref_neg": "safe"},
}


def discover_sessions():
    return sorted(
        d.name for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}', d.name)
    )


def session_data(session):
    """Load (sorted) spikes and condition event times for one session."""
    sess = Session(session, HERE)
    trains, _ = sess.spike_trains
    trains = [np.sort(s) for s in trains]
    return trains, sess.condition_event_times(event="reward")


def mean_rate(spikes, event_times, t_lo, t_hi):
    """Mean firing rate (Hz) in [event + t_lo, event + t_hi], averaged across events."""
    win = t_hi - t_lo
    if win <= 0 or len(event_times) == 0:
        return np.nan
    los = event_times + t_lo
    his = event_times + t_hi
    counts = (np.searchsorted(spikes, his, side="left")
              - np.searchsorted(spikes, los, side="left"))
    return counts.sum() / (len(event_times) * win)


def col_name(cond):
    return "rate_" + cond.replace("+", "")


def augment_csv(csv_path, cfg, trains, cond_times, dur_s, w_s):
    df = pd.read_csv(csv_path)

    cond_a, cond_b = cfg["a"], cfg["b"]
    events_a       = cond_times[cond_a]
    events_b       = cond_times[cond_b]
    rate_a_col     = col_name(cond_a)
    rate_b_col     = col_name(cond_b)

    rates_a = np.full(len(df), np.nan)
    rates_b = np.full(len(df), np.nan)
    sis     = np.full(len(df), np.nan)
    prefs   = [""] * len(df)

    for i, row in df.iterrows():
        zt = row["zeta_t_s"]
        if pd.isna(zt):
            continue
        spikes = trains[int(row["neuron_idx"])]

        t_lo = max(zt - w_s, 0.0)
        t_hi = min(zt + w_s, dur_s)
        if t_hi <= t_lo:
            continue

        r_a = mean_rate(spikes, events_a, t_lo, t_hi)
        r_b = mean_rate(spikes, events_b, t_lo, t_hi)
        rates_a[i] = r_a
        rates_b[i] = r_b
        denom = r_a + r_b
        if not (np.isnan(r_a) or np.isnan(r_b)) and denom > 0:
            si = (r_a - r_b) / denom
            sis[i] = si
            prefs[i] = (cfg["pref_pos"] if si > 0
                        else cfg["pref_neg"] if si < 0
                        else "equal")

    df[rate_a_col] = rates_a
    df[rate_b_col] = rates_b
    df["SI"]         = sis
    df["preference"] = prefs

    tmp = csv_path.with_suffix(".csv.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)
    except PermissionError as exc:
        print(f"  [LOCK] {csv_path.name} not writable ({exc}). "
              "Stats still pooled below; CSV not modified.")
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
    return df


def plot_distribution(pooled, contrast, cfg, alpha, window_ms, out_path):
    sig = pooled[(pooled["p_zeta"] < alpha) & pooled["SI"].notna()]
    n_pos = int((sig["SI"] > 0).sum())
    n_neg = int((sig["SI"] < 0).sum())
    n_total = n_pos + n_neg
    p_binom = binomtest(n_pos, n_total, 0.5).pvalue if n_total > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(-1, 1, 41)
    ax.hist(sig.loc[sig["SI"] > 0, "SI"], bins=bins, color="seagreen",  alpha=0.75,
            label=f"{cfg['pref_pos']}  (n={n_pos})")
    ax.hist(sig.loc[sig["SI"] < 0, "SI"], bins=bins, color="firebrick", alpha=0.75,
            label=f"{cfg['pref_neg']}  (n={n_neg})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(
        f"SI = (r_{cfg['a']} - r_{cfg['b']}) / (r_{cfg['a']} + r_{cfg['b']})",
        fontsize=9,
    )
    ax.set_ylabel("Significant neurons", fontsize=9)
    pct = 100 * n_pos / max(n_total, 1)
    ax.set_title(
        f"{contrast} contrast — direction at zeta_t_s +/- {window_ms} ms "
        f"(alpha={alpha})\n"
        f"prefer {cfg['pref_pos']} vs {cfg['pref_neg']}: "
        f"{n_pos}/{n_total} ({pct:.0f}%) - "
        f"binomial p vs 50/50 = {p_binom:.3g}",
        fontsize=9,
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n_pos, n_neg, p_binom


def parse_args():
    p = argparse.ArgumentParser(
        description="Add direction-of-effect columns to zeta2 CSVs and plot SI distributions.")
    p.add_argument("--sessions", nargs="+", metavar="YYYYMMDD",
                   help="Sessions to process (default: every YYYYMMDD directory).")
    p.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS,
                   help=f"Half-width of window around zeta_t_s (default: {DEFAULT_WINDOW_MS} ms).")
    p.add_argument("--dur", type=float, default=DEFAULT_DUR_S,
                   help=f"ZETA analysis window in seconds (default: {DEFAULT_DUR_S}).")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help=f"Significance threshold for the histogram (default: {DEFAULT_ALPHA}).")
    return p.parse_args()


def main():
    args     = parse_args()
    sessions = args.sessions or discover_sessions()
    w_s      = args.window_ms / 1000.0

    if not sessions:
        print("No sessions found.")
        return

    DIRECTION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sessions: {sessions}")
    print(f"Window: zeta_t_s +/- {args.window_ms} ms, clipped to [0, {args.dur}] s")
    print(f"Significance: p_zeta < {args.alpha}\n")

    pooled = {c: [] for c in CONTRASTS}

    for session in sessions:
        print(f"=== {session} ===")
        trains, cond_times = session_data(session)

        for contrast, cfg in CONTRASTS.items():
            csv_path = OUTCOME_DIR / f"zeta2_{contrast}_{session}.csv"
            if not csv_path.exists():
                print(f"  [skip] {csv_path.name} not found in {OUTCOME_DIR}")
                continue

            df = augment_csv(csv_path, cfg, trains, cond_times, args.dur, w_s)
            df = df.copy()
            df["session"] = session
            pooled[contrast].append(df)

            sig_mask = df["p_zeta"] < args.alpha
            n_sig = int(sig_mask.sum())
            n_pos = int((sig_mask & (df["SI"] > 0)).sum())
            n_neg = int((sig_mask & (df["SI"] < 0)).sum())
            print(f"  {contrast}: significant={n_sig:4d}  "
                  f"{cfg['pref_pos']}={n_pos:4d}  {cfg['pref_neg']}={n_neg:4d}")

    print()
    for contrast, cfg in CONTRASTS.items():
        if not pooled[contrast]:
            continue
        big = pd.concat(pooled[contrast], ignore_index=True)
        out_path = DIRECTION_DIR / f"direction_{contrast}.png"
        n_pos, n_neg, p_binom = plot_distribution(big, contrast, cfg,
                                                  args.alpha, args.window_ms, out_path)
        n_total = n_pos + n_neg
        pct = 100 * n_pos / max(n_total, 1)
        print(f"POOLED {contrast}: significant w/ SI = {n_total} | "
              f"{cfg['pref_pos']}={n_pos} ({pct:.0f}%), "
              f"{cfg['pref_neg']}={n_neg} ({100 - pct:.0f}%) | "
              f"binomial p vs 50/50 = {p_binom:.3g}")
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
