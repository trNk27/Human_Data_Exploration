"""Two-sample ZETA: does a neuron's reward-aligned response differ by outcome?

All trials are aligned to reward onset (RewardOnset_sp), responding trials
only. Three outcomes are considered:

    G+R   gamble arm, rewarded
    G+N   gamble arm, not rewarded
    S+R   safe arm,   rewarded

The one-sample zeta_analysis.py only tests whether a neuron is responsive; it
cannot say whether the response *differs* between outcomes. This script uses
the two-sample ZETA-test (zetapy.zetatest2) for two contrasts, each isolating
one factor:

    reward   G+R vs G+N   — effect of reward   (choice held = gamble)
    choice   G+R vs S+R   — effect of choice   (reward held constant)

For every neuron it runs zetatest2, writes a results table sorted by p-value,
and plots the ZETA difference curve for the top-N most significant neurons.
Neurons are tested in parallel across CPU cores (see --jobs).

Usage:
    python zeta_outcome.py
    python zeta_outcome.py --contrast reward --top 10 --save --csv
    python zeta_outcome.py --jobs 4

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
from zetapy import zetatest2

from utils import get_spike_trains, load_trials_sync, load_sr, sp_to_s, SESSION, add_save_arg, maybe_save


REWARD_COL = "RewardOnset_sp"   # outcome onset, sampling points; present on no-reward trials too

# Each contrast compares condition "a" against condition "b" (same neuron).
CONTRASTS = {
    "reward": {"a": "G+R", "b": "G+N", "desc": "effect of reward (gamble: rewarded vs not)"},
    "choice": {"a": "G+R", "b": "S+R", "desc": "effect of choice (rewarded: gamble vs safe)"},
}

DEFAULT_DUR_S  = 2.0
DEFAULT_RESAMP = 250   # zetatest2 default; raise if p-values sit near the threshold
MIN_SPIKES     = 10
MIN_TRIALS     = 10


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def condition_event_times(trials, sr):
    """Reward-onset times (seconds) split into the three outcome conditions."""
    times      = sp_to_s(trials, sr, REWARD_COL)
    responding = (trials["NotResponding"] == 0).to_numpy()
    gamble     = (trials["ChosenArm_G1S0"] == 1).to_numpy()
    rewarded   = (trials["Rewarded"] == 1).to_numpy()
    return {
        "G+R": times[responding &  gamble &  rewarded],
        "G+N": times[responding &  gamble & ~rewarded],
        "S+R": times[responding & ~gamble &  rewarded],
    }


# ---------------------------------------------------------------------------
# ZETA
# ---------------------------------------------------------------------------

def _empty_row(i, label):
    return {"neuron_idx": i, "label": label, "p_zeta": 1.0,
            "zeta": np.nan, "zeta_t_s": np.nan,
            "mean_z": np.nan, "p_mean": 1.0}


def _zeta2_one(task):
    """Run zetatest2 for one neuron. Top-level so it is picklable for the pool.

    Returns (neuron_idx, result_row, plot_data-or-None).
    """
    i, spikes, label, events_a, events_b, dur_s, n_resamp = task
    if len(spikes) < MIN_SPIKES or len(events_a) < MIN_TRIALS or len(events_b) < MIN_TRIALS:
        return i, _empty_row(i, label), None

    try:
        # Same spike train, two event subsets: tests whether the event-locked
        # response differs between condition a and condition b.
        dblP, dZETA = zetatest2(
            spikes, events_a, spikes, events_b,
            dblUseMaxDur=dur_s,
            intResampNum=n_resamp,
            boolPlot=False,
        )
    except Exception as exc:
        print(f"  [skip] neuron {i}: {exc}", flush=True)
        return i, _empty_row(i, label), None

    row = {
        "neuron_idx": i,
        "label":      label,
        "p_zeta":     dblP,
        "zeta":       dZETA["dblZETA"],
        "zeta_t_s":   dZETA["dblZetaT"],
        "mean_z":     dZETA["dblMeanZ"],
        "p_mean":     dZETA["dblMeanP"],
    }
    # Keep only the difference curve for plotting — the resampled null arrays
    # in dZETA are large and not needed downstream.
    plot_data = {"vecSpikeT": dZETA["vecSpikeT"], "vecRealDiff": dZETA["vecRealDiff"]}
    return i, row, plot_data


def run_zeta2_all_neurons(trains, labels, events_a, events_b,
                          dur_s=DEFAULT_DUR_S, n_resamp=DEFAULT_RESAMP, n_jobs=None):
    """Run zetatest2 for every neuron across n_jobs worker processes.

    n_jobs=1 runs serially; None uses every CPU core. Returns a results
    DataFrame sorted by p-value and a list of plot-data dicts indexed by
    neuron_idx (None where a neuron was skipped).
    """
    n_total = len(trains)
    tasks   = [(i, spikes, label, events_a, events_b, dur_s, n_resamp)
               for i, (spikes, label) in enumerate(zip(trains, labels))]

    rows_by_idx = {}
    plot_by_idx = {}

    def _record(done, i, row, plot_data):
        print(f"  [{done}/{n_total}] {row['label']}", flush=True)
        rows_by_idx[i] = row
        plot_by_idx[i] = plot_data

    if n_jobs == 1:
        for done, task in enumerate(tasks, start=1):
            _record(done, *_zeta2_one(task))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_zeta2_one, task) for task in tasks]
            for done, fut in enumerate(as_completed(futures), start=1):
                _record(done, *fut.result())

    rows      = [rows_by_idx[i] for i in range(n_total)]
    plot_data = [plot_by_idx[i] for i in range(n_total)]
    df = pd.DataFrame(rows).sort_values("p_zeta").reset_index(drop=True)
    return df, plot_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diff_grid(results_df, plot_data_list, alpha, top_n, contrast, args):
    """Plot the ZETA difference curve for the top_n most significant neurons.

    vecRealDiff is the temporal deviation between the two conditions' cumulative
    spike vectors; its largest excursion from zero is what ZETA scores.
    """
    cfg = CONTRASTS[contrast]
    sig = results_df[results_df["p_zeta"] < alpha].head(top_n)
    if sig.empty:
        print(f"  No significant neurons (alpha={alpha}) for contrast '{contrast}'.")
        return

    ncols = min(4, len(sig))
    nrows = int(np.ceil(len(sig) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (_, row) in enumerate(sig.iterrows()):
        ax   = axes_flat[ax_i]
        data = plot_data_list[int(row["neuron_idx"])]

        if data is not None:
            ax.plot(data["vecSpikeT"], data["vecRealDiff"], color="purple", lw=1.5)
        else:
            ax.text(0.5, 0.5, "no diff data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        if not np.isnan(row["zeta_t_s"]):
            ax.axvline(row["zeta_t_s"], color="tomato", lw=1.0, ls=":",
                       label=f"ZETA t={row['zeta_t_s']*1000:.0f} ms")

        short_label = row["label"].split("|")[0].strip()
        ax.set_title(f"{short_label}\np={row['p_zeta']:.2e}", fontsize=8)
        ax.set_xlabel("Time from reward onset (s)", fontsize=7)
        ax.set_ylabel(f"Cum. spike diff\n({cfg['a']} − {cfg['b']})", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for ax_i in range(len(sig), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(f"Two-sample ZETA — {contrast}: {cfg['a']} vs {cfg['b']} "
                 f"| session: {SESSION} | α={alpha}", fontsize=10, y=1.01)
    plt.tight_layout()
    maybe_save(fig, args, prefix=f"zeta2_{contrast}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Two-sample ZETA test for outcome differences.")
    p.add_argument("--contrast", default="all",
                   choices=list(CONTRASTS.keys()) + ["all"],
                   help="Which contrast to run (default: all).")
    p.add_argument("--dur",    type=float, default=DEFAULT_DUR_S,
                   help=f"Analysis window in seconds (default: {DEFAULT_DUR_S}).")
    p.add_argument("--resamp", type=int,   default=DEFAULT_RESAMP,
                   help=f"Jitter iterations (default: {DEFAULT_RESAMP}).")
    p.add_argument("--alpha",  type=float, default=0.05,
                   help="Significance threshold (default: 0.05).")
    p.add_argument("--top",    type=int,   default=8,
                   help="Top N significant neurons to plot (default: 8).")
    p.add_argument("--csv",    action="store_true",
                   help="Write the full results table per contrast to results/.")
    p.add_argument("--jobs",   type=int,   default=None,
                   help="Parallel worker processes (default: all CPU cores; 1 = serial).")
    add_save_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Session: {SESSION}")

    trains, labels = get_spike_trains()
    trials = load_trials_sync()
    sr     = load_sr()["SamplingRate_Hz"].iloc[0]
    print(f"Loaded {len(trains)} neurons, {len(trials)} trials, SR={sr} Hz")
    print(f"Workers: {args.jobs or os.cpu_count()} process(es)")

    cond_times = condition_event_times(trials, sr)
    print("Outcome trial counts: "
          + ", ".join(f"{name}={len(t)}" for name, t in cond_times.items()))

    contrasts_to_run = list(CONTRASTS.keys()) if args.contrast == "all" else [args.contrast]

    for contrast in contrasts_to_run:
        cfg      = CONTRASTS[contrast]
        events_a = cond_times[cfg["a"]]
        events_b = cond_times[cfg["b"]]
        print(f"\n--- Contrast: {contrast} | {cfg['a']} ({len(events_a)}) "
              f"vs {cfg['b']} ({len(events_b)}) — {cfg['desc']} ---")

        results, plot_data = run_zeta2_all_neurons(
            trains, labels, events_a, events_b,
            dur_s=args.dur, n_resamp=args.resamp, n_jobs=args.jobs,
        )

        n_sig = int((results["p_zeta"] < args.alpha).sum())
        print(f"Significant (alpha={args.alpha}): {n_sig} / {len(results)} neurons")
        print(results[["label", "zeta", "p_zeta", "zeta_t_s", "p_mean"]].head(10).to_string(index=False))

        if args.csv:
            out_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"zeta2_{contrast}_{SESSION}.csv")
            results.to_csv(csv_path, index=False)
            print(f"Saved -> {csv_path}")

        plot_diff_grid(results, plot_data,
                       alpha=args.alpha, top_n=args.top,
                       contrast=contrast, args=args)


if __name__ == "__main__":
    main()
