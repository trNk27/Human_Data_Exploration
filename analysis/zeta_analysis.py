"""ZETA-test analysis: test each neuron's responsiveness to behavioural events.

Tests all neurons against cue / response-window / reward / trial-start onsets
(events from utils.EVENTS). Outputs a summary table sorted by p-value and
plots IFRs for the top N most significant neurons per event.

Usage:
    python zeta_analysis.py
    python zeta_analysis.py --session 20250605 --event reward --top 10 --save
    python zeta_analysis.py --event all --dur 2.0 --alpha 0.05
    python zeta_analysis.py --jobs 4          # limit parallel worker processes

    # Variable-duration window: per trial, use [event, window_end] instead of
    # a fixed --dur. Useful for cue, where the cue->reward interval varies and
    # a fixed 2s window leaks into post-reward activity.
    python zeta_analysis.py --event cue --window-end reward --csv --save

Neurons are tested in parallel across CPU cores by default (--jobs 1 forces
serial). The CLI must therefore be run as a script, not imported.

Requires: pip install zetapy
"""

import argparse
import os

# Keep each worker process single-threaded for BLAS so the neuron-level
# process pool below does not oversubscribe the CPU. Must precede `import numpy`.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zetapy import zetatest

# Make the repo root importable when run as `python analysis/<file>.py`.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session import Session
from utils import (
    EVENTS, RESULTS_SUBDIRS,
    event_times as event_times_for,
    add_session_arg, add_save_arg, maybe_save,
)


DEFAULT_DUR_S  = 2.0
DEFAULT_RESAMP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def responding_event_times(event_name, trials, sr):
    """Event onset times in seconds, responding trials only."""
    times      = event_times_for(trials, sr, event_name)
    responding = trials["NotResponding"] == 0
    return times[responding]


def responding_event_windows(start_event, end_event, trials, sr):
    """Per-trial [start, end] times (seconds) for ZETA's variable-duration mode.

    Returns an array of shape (n_valid_trials, 2). Includes only responding
    trials where both endpoints are finite and end > start. Trials that fail
    the check are dropped and reported.
    """
    starts = event_times_for(trials, sr, start_event)
    ends   = event_times_for(trials, sr, end_event)
    responding = (trials["NotResponding"] == 0).to_numpy()
    valid = responding & np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    n_dropped = int(responding.sum() - valid.sum())
    if n_dropped:
        print(f"  Dropped {n_dropped} responding trials with invalid "
              f"[{start_event} -> {end_event}] window.")
    return np.column_stack([starts[valid], ends[valid]])


def _empty_row(i, label):
    return {"neuron_idx": i, "label": label,
            "p_zeta": 1.0, "zeta": np.nan,
            "mean_z": np.nan, "p_mean": 1.0,
            "latency_s": np.nan, "peak_onset_s": np.nan}


def _zeta_one(task):
    """Run zetatest for one neuron. Top-level so it is picklable for the process pool.

    Returns (neuron_idx, result_row, dRate-or-None).
    """
    i, spikes, label, ev_times, dur_s, n_resamp = task
    if len(spikes) < 10:
        return i, _empty_row(i, label), None

    try:
        dblP, dZETA, dRate = zetatest(
            spikes, ev_times,
            dblUseMaxDur=dur_s,
            intResampNum=n_resamp,
            boolReturnRate=True,
            boolParallel=False,
            boolPlot=False,
        )
    except Exception as exc:
        print(f"  [skip] neuron {i}: {exc}", flush=True)
        return i, _empty_row(i, label), None

    return i, {
        "neuron_idx":  i,
        "label":       label,
        "p_zeta":      dblP,
        "zeta":        dZETA["dblZETA"],
        "mean_z":      dZETA["dblMeanZ"],
        "p_mean":      dZETA["dblMeanP"],
        "latency_s":   dZETA["dblLatencyZETA"],
        "peak_onset_s": dRate.get("dblLatencyPeakOnset", np.nan),
    }, dRate


def run_zeta_all_neurons(trains, labels, ev_times, dur_s=DEFAULT_DUR_S,
                         n_resamp=DEFAULT_RESAMP, n_jobs=None):
    """Run zetatest for every neuron across n_jobs worker processes."""
    n_total = len(trains)
    tasks   = [(i, spikes, label, ev_times, dur_s, n_resamp)
               for i, (spikes, label) in enumerate(zip(trains, labels))]

    rows_by_idx = {}
    rate_by_idx = {}

    def _record(done, i, row, dRate):
        print(f"  [{done}/{n_total}] {row['label']}", flush=True)
        rows_by_idx[i] = row
        rate_by_idx[i] = dRate

    if n_jobs == 1:
        for done, task in enumerate(tasks, start=1):
            _record(done, *_zeta_one(task))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_zeta_one, task) for task in tasks]
            for done, fut in enumerate(as_completed(futures), start=1):
                _record(done, *fut.result())

    rows      = [rows_by_idx[i] for i in range(n_total)]
    rate_data = [rate_by_idx[i] for i in range(n_total)]
    df = pd.DataFrame(rows).sort_values("p_zeta").reset_index(drop=True)
    return df, rate_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ifr_grid(results_df, rate_data_list, alpha, top_n, event_name, args,
                  file_prefix=None):
    """Plot IFR for the top_n significant neurons."""
    if file_prefix is None:
        file_prefix = f"zeta_{event_name}"
    window_desc = (f"{event_name} -> {args.window_end} (per-trial)"
                   if args.window_end else f"{event_name} (fixed {args.dur}s)")
    sig = results_df[results_df["p_zeta"] < alpha].head(top_n)
    if sig.empty:
        print(f"  No significant neurons (alpha={alpha}) for event '{event_name}'.")
        return

    ncols = min(4, len(sig))
    nrows = int(np.ceil(len(sig) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (_, row) in enumerate(sig.iterrows()):
        ax    = axes_flat[ax_i]
        dRate = rate_data_list[int(row["neuron_idx"])]

        if dRate is not None and "vecT" in dRate and "vecRate" in dRate:
            ax.plot(dRate["vecT"], dRate["vecRate"], color="steelblue", lw=1.5)
        else:
            ax.text(0.5, 0.5, "no rate data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)

        ax.axvline(0, color="gray", lw=0.8, ls="--")
        if not np.isnan(row["latency_s"]):
            ax.axvline(row["latency_s"], color="tomato", lw=1.0, ls=":",
                       label=f"ZETA t={row['latency_s']*1000:.0f} ms")
        if not np.isnan(row["peak_onset_s"]):
            ax.axvline(row["peak_onset_s"], color="orange", lw=1.0, ls="--",
                       label=f"onset t={row['peak_onset_s']*1000:.0f} ms")

        short_label = row["label"].split("|")[0].strip()
        ax.set_title(f"{short_label}\np={row['p_zeta']:.2e}", fontsize=8)
        ax.set_xlabel("Time from event (s)", fontsize=7)
        ax.set_ylabel("IFR (Hz)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for ax_i in range(len(sig), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(
        f"ZETA IFR — {window_desc} | session: {args.session} | alpha={alpha}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    maybe_save(fig, args, prefix=file_prefix,
               subdir=RESULTS_SUBDIRS["responsiveness"])
    plt.show()


def plot_pvalue_overview(all_results, alpha, session):
    """Fraction of significant neurons per event."""
    events  = list(all_results.keys())
    n_total = [len(df) for df in all_results.values()]
    n_sig   = [int((df["p_zeta"] < alpha).sum()) for df in all_results.values()]
    fracs   = [s / n for s, n in zip(n_sig, n_total)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(events, fracs, color="steelblue", edgecolor="white")
    ax.bar_label(bars, labels=[f"{s}/{n}" for s, n in zip(n_sig, n_total)], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(alpha, color="tomato", ls="--", lw=1, label=f"alpha={alpha}")
    ax.set_ylabel("Fraction of significant neurons")
    ax.set_title(f"ZETA responsiveness — session {session}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ZETA responsiveness test for all neurons.")
    add_session_arg(p)
    p.add_argument("--event",  default="all",
                   choices=list(EVENTS.keys()) + ["all"],
                   help="Which event to align to (default: all).")
    p.add_argument("--dur",    type=float, default=DEFAULT_DUR_S,
                   help=f"Fixed analysis window in seconds (default: {DEFAULT_DUR_S}). "
                        "Ignored when --window-end is set.")
    p.add_argument("--window-end", default=None, choices=list(EVENTS),
                   help="Variable-duration mode: use [event, window_end] per "
                        "trial instead of a fixed --dur. Requires a specific "
                        "--event (not 'all'). Example: --event cue --window-end reward.")
    p.add_argument("--resamp", type=int,   default=DEFAULT_RESAMP,
                   help=f"Jitter iterations (default: {DEFAULT_RESAMP}).")
    p.add_argument("--alpha",  type=float, default=0.05,
                   help="Significance threshold (default: 0.05).")
    p.add_argument("--top",    type=int,   default=8,
                   help="Top N significant neurons to plot (default: 8).")
    p.add_argument("--csv",    action="store_true",
                   help=f"Write the full results table per event to {RESULTS_SUBDIRS['responsiveness']}/.")
    p.add_argument("--jobs",   type=int,   default=None,
                   help="Parallel worker processes (default: all CPU cores; 1 = serial).")
    add_save_arg(p)
    args = p.parse_args()
    if args.window_end and args.event == "all":
        p.error("--window-end requires a specific --event (not 'all').")
    if args.window_end == args.event:
        p.error("--window-end must differ from --event.")
    return args


def main():
    args   = parse_args()
    sess   = Session(args.session)
    print(f"Session: {args.session}")

    trains, labels = sess.spike_trains
    trials, sr     = sess.trials, sess.sampling_rate
    print(f"Loaded {len(trains)} neurons, {len(trials)} trials, SR={sr} Hz")
    print(f"Workers: {args.jobs or os.cpu_count()} process(es)")

    events_to_run = list(EVENTS.keys()) if args.event == "all" else [args.event]
    all_results   = {}
    suffix        = f"_to_{args.window_end}" if args.window_end else ""

    for event_name in events_to_run:
        # 1-D onsets (fixed --dur) or 2-D [onset, offset] per trial (--window-end).
        if args.window_end:
            ev_times = responding_event_windows(event_name, args.window_end, trials, sr)
            durations = ev_times[:, 1] - ev_times[:, 0]
            dur_used  = float(durations.max())
            print(f"\n--- Event: {event_name} -> {args.window_end} | "
                  f"{len(ev_times)} trials, "
                  f"window {durations.min():.2f}..{dur_used:.2f}s "
                  f"(median {np.median(durations):.2f}s) ---")
        else:
            ev_times = responding_event_times(event_name, trials, sr)
            dur_used = args.dur
            print(f"\n--- Event: {event_name} ({len(ev_times)} onsets, "
                  f"fixed {dur_used}s window) ---")

        results, rate_data = run_zeta_all_neurons(
            trains, labels, ev_times,
            dur_s=dur_used, n_resamp=args.resamp, n_jobs=args.jobs,
        )
        all_results[event_name] = results

        n_sig = int((results["p_zeta"] < args.alpha).sum())
        print(f"Significant (alpha={args.alpha}): {n_sig} / {len(results)} neurons")
        print(results[["label", "zeta", "p_zeta", "latency_s", "peak_onset_s"]].head(10).to_string(index=False))

        if args.csv:
            out_dir = RESULTS_SUBDIRS["responsiveness"]
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"zeta_{event_name}{suffix}_{args.session}.csv")
            results.to_csv(csv_path, index=False)
            print(f"Saved -> {csv_path}")

        plot_ifr_grid(results, rate_data,
                      alpha=args.alpha, top_n=args.top,
                      event_name=event_name, args=args,
                      file_prefix=f"zeta_{event_name}{suffix}")

    if len(events_to_run) > 1:
        plot_pvalue_overview(all_results, args.alpha, args.session)


if __name__ == "__main__":
    main()
