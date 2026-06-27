"""Per-neuron firing-rate vs RW prediction-error correlation table (all neurons, all sessions).

For every neuron in every session, computes the linear (Pearson) correlation
between per-trial firing rate and the Rescorla-Wagner (+ stickiness) reward
prediction error δ = outcome − Q, **separately for the two gamble outcomes**:

  * G+R  (gamble + rewarded)   → δ > 0   columns ``r_gamble_rewarded`` / ``p_gamble_rewarded``
  * G+NR (gamble + no reward)  → δ < 0   columns ``r_gamble_unrewarded`` / ``p_gamble_unrewarded``

One row per neuron. The result is a flat CSV you can sort/filter to find the
neurons whose firing tracks the prediction error most strongly and where they sit
(region / electrode / unit type). This is the batch, stats-only counterpart of the
per-neuron scatter in ``viewers/firing_rate_vs_rw_pe.py`` (same per-condition
``scipy.stats.linregress`` fit), aggregated across the whole dataset.

The prediction error is read from ``results/rw/trajectory_<session>.csv`` — run
``python -m analysis.rw_trajectories`` first to generate it.

Usage
-----
    python -m analysis.fr_pe_correlation [--window reward_to_end]
                                         [--sessions 20250521 ...]
                                         [--min-trials 4] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Repo root importable when run via ``python -m analysis.fr_pe_correlation``
# or ``python analysis/fr_pe_correlation.py``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from session import Session
from compute import trial_firing_rates, WINDOWS
from utils import RESULTS_DIR, load_unit_info, load_rw_trajectory_column
from analysis.hgf.data import list_sessions as available_sessions

# Map the two gamble outcomes to clear output-column stems.
_CONDITIONS = {
    "G+R": "gamble_rewarded",     # δ = 1 − Q  (positive prediction error)
    "G+N": "gamble_unrewarded",   # δ = 0 − Q  (negative prediction error)
}


def _corr(x, y, min_trials):
    """Pearson r, p, n for finite (x, y) pairs; (nan, nan, n) if too few / no variance."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < min_trials or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, np.nan, n
    res = stats.linregress(x[ok], y[ok])
    return float(res.rvalue), float(res.pvalue), n


def session_rows(session_id, window, min_trials):
    """One row per neuron for a session, or [] if the RW trajectory is missing."""
    sess = Session(session_id)
    n_trials = len(sess.trials)

    rw_pe = load_rw_trajectory_column(session_id, n_trials, "rw_pe")
    if rw_pe is None:
        print(f"  {session_id}: no RW trajectory CSV — skipping "
              f"(run `python -m analysis.rw_trajectories`)")
        return []

    trains, labels = sess.spike_trains
    info = load_unit_info(sess._data_dir)
    masks = sess.condition_masks()
    # firing rate per (trial, neuron) over the window; NaN where the window is undefined
    rates = trial_firing_rates(trains, sess.trials, sess.sampling_rate, window=window)

    rows = []
    for j in range(len(trains)):
        row = {
            "session": session_id,
            "neuron_id": j,
            "label": labels[j],
            "unit_id": info["unit_id"].iloc[j],
            "region": info["region"].iloc[j],
            "electrode": info["electrode"].iloc[j],
            "unit_type": info["unit_type"].iloc[j],
        }
        for key, stem in _CONDITIONS.items():
            m = masks[key]
            r, p, n = _corr(rw_pe[m], rates[m, j], min_trials)
            row[f"r_{stem}"] = r
            row[f"p_{stem}"] = p
            row[f"n_{stem}"] = n
        rows.append(row)
    print(f"  {session_id}: {len(rows)} neurons")
    return rows


def run(sessions=None, window="reward_to_end", min_trials=4, out=None) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    if sessions is None:
        sessions = available_sessions()
    print(f"Window: {window}  |  sessions: {sessions}")

    rows = []
    for sid in sessions:
        rows.extend(session_rows(sid, window, min_trials))
    df = pd.DataFrame(rows)

    out = out or os.path.join(RESULTS_DIR, "rw", f"fr_vs_rw_pe_correlations_{window}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} neuron rows -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Per-neuron firing-rate vs RW prediction-error correlation table")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Session IDs (default: all available)")
    p.add_argument("--window", default="reward_to_end", choices=list(WINDOWS),
                   help="Firing-rate window (default: reward_to_end — the post-outcome "
                        "period, where a reward prediction error is expressed)")
    p.add_argument("--min-trials", type=int, default=4,
                   help="Minimum finite trials per condition to report a correlation (default: 4)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: results/rw/fr_vs_rw_pe_correlations_<window>.csv)")
    args = p.parse_args()
    run(sessions=args.sessions, window=args.window, min_trials=args.min_trials, out=args.out)
