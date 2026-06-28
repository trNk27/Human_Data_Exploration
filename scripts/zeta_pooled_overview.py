"""Pooled one-sample ZETA responsiveness across ALL sessions.

For every session directory (YYYYMMDD) sitting next to the repo root, run the
one-sample ZETA responsiveness test (analysis/zeta_analysis.py) for each
behavioural event on every neuron, then POOL the significant / total counts
across sessions and draw one bar per event: the fraction of all recorded
neurons that ZETA flags as responsive.

This answers "how many neurons turn out significant with ZETA alone?" — the
deliberately low bar that almost every task-engaged neuron clears. Compare it
against the two-sample outcome contrasts (analysis/zeta_outcome.py), which is
the bar that actually selects for reward/choice coding.

Run on the machine that has the .mat data + the `humandata` env:

    python scripts/zeta_pooled_overview.py --save --csv
    python scripts/zeta_pooled_overview.py --resamp 250 --jobs 4 --save

Outputs (with --save):
    results/zeta_responsiveness/zeta_pooled_overview.png
    results/zeta_responsiveness/zeta_pooled_counts.csv   (per session x event)
"""

import argparse
import os

# Single-thread BLAS so the neuron-level process pool does not oversubscribe.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from session import Session
from utils import EVENTS, EVENT_STYLE, REPO_ROOT, RESULTS_SUBDIRS
from analysis.zeta_analysis import responding_event_times, run_zeta_all_neurons


def find_sessions():
    """All YYYYMMDD session dirs next to the repo root, sorted."""
    return sorted(
        d for d in os.listdir(REPO_ROOT)
        if len(d) == 8 and d.isdigit()
        and os.path.isdir(os.path.join(REPO_ROOT, d))
    )


def parse_args():
    p = argparse.ArgumentParser(description="Pooled ZETA responsiveness across sessions.")
    p.add_argument("--alpha",  type=float, default=0.05, help="Significance threshold.")
    p.add_argument("--dur",    type=float, default=2.0,  help="Fixed window (s) per event.")
    p.add_argument("--resamp", type=int,   default=100,  help="Jitter iterations.")
    p.add_argument("--jobs",   type=int,   default=None, help="Worker processes (1 = serial).")
    p.add_argument("--csv",    action="store_true",      help="Also write the per-session counts CSV.")
    p.add_argument("--save",   action="store_true",      help="Save the figure (else just show).")
    return p.parse_args()


def main():
    args     = parse_args()
    sessions = find_sessions()
    events   = list(EVENTS.keys())
    if not sessions:
        sys.exit(f"No YYYYMMDD session dirs found under {REPO_ROOT}. "
                 "Run this on the machine that has the .mat data.")

    print(f"Sessions ({len(sessions)}): {', '.join(sessions)}")

    pooled = {ev: {"sig": 0, "total": 0} for ev in events}
    rows   = []

    for sid in sessions:
        sess           = Session(sid)
        trains, labels = sess.spike_trains
        trials, sr     = sess.trials, sess.sampling_rate
        print(f"\n[{sid}] {len(trains)} neurons, {len(trials)} trials")

        for ev in events:
            ev_times       = responding_event_times(ev, trials, sr)
            results, _     = run_zeta_all_neurons(
                trains, labels, ev_times,
                dur_s=args.dur, n_resamp=args.resamp, n_jobs=args.jobs,
            )
            n_sig = int((results["p_zeta"] < args.alpha).sum())
            n_tot = len(results)
            pooled[ev]["sig"]   += n_sig
            pooled[ev]["total"] += n_tot
            rows.append({"session": sid, "event": ev, "n_sig": n_sig, "n_total": n_tot})
            print(f"   {ev:12s}: {n_sig:3d} / {n_tot:3d}  ({n_sig / n_tot:.0%})")

    # ---- pooled bar chart -------------------------------------------------
    fracs   = [pooled[ev]["sig"] / pooled[ev]["total"] for ev in events]
    n_sig   = [pooled[ev]["sig"]   for ev in events]
    n_tot   = [pooled[ev]["total"] for ev in events]
    xlabels = [EVENTS[ev]["label"] for ev in events]
    colors  = [EVENT_STYLE[ev]["color"] for ev in events]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(xlabels, fracs, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{s}/{t}" for s, t in zip(n_sig, n_tot)],
                 fontsize=9, padding=2)
    ax.axhline(args.alpha, color="tomato", ls="--", lw=1,
               label=f"chance = {args.alpha:.0%}")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of neurons ZETA-significant")
    ax.set_title(f"One-sample ZETA responsiveness, pooled across "
                 f"{len(sessions)} sessions\n(total {n_tot[0]} neurons, "
                 f"alpha = {args.alpha})")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_dir = RESULTS_SUBDIRS["responsiveness"]
    os.makedirs(out_dir, exist_ok=True)
    if args.csv:
        csv_path = os.path.join(out_dir, "zeta_pooled_counts.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSaved counts -> {csv_path}")
    if args.save:
        png = os.path.join(out_dir, "zeta_pooled_overview.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved figure -> {png}")
    plt.show()


if __name__ == "__main__":
    main()
