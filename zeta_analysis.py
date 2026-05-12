"""ZETA-test analysis: test each neuron's responsiveness to behavioural events.

Tests all neurons against cue onset, response-window onset, and reward onset.
Outputs a summary table sorted by p-value and plots IFRs for the top N
most significant neurons per event.

Usage:
    python zeta_analysis.py
    python zeta_analysis.py --event reward --top 10 --save
    python zeta_analysis.py --event all --dur 2.0 --alpha 0.05

Requires: pip install zetapy
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zetapy import zetatest, ifr as zeta_ifr

from utils import get_spike_trains, load_trials_sync, load_sr, sp_to_s, SESSION, add_save_arg, maybe_save


# ---------------------------------------------------------------------------
# Event definitions (name → Trials_Sync column, already in seconds or needs conversion)
# ---------------------------------------------------------------------------

EVENTS = {
    "cue":      {"col": "CuePresent_sp",         "unit": "sp"},
    "response": {"col": "RespWindowStart_sp",     "unit": "sp"},
    "reward":   {"col": "RewardOnset_sp",         "unit": "sp"},
    "trial_start": {"col": "TrialStart_s",        "unit": "s"},
}

DEFAULT_DUR_S = 2.0   # analysis window after each event onset (seconds)
DEFAULT_RESAMP = 100  # number of jitter iterations for null distribution


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def get_event_times(event_name, trials, sr):
    """Return event onset times in seconds, excluding non-responding trials."""
    cfg = EVENTS[event_name]
    responding = trials["NotResponding"] == 0

    if cfg["unit"] == "sp":
        times = sp_to_s(trials, sr, cfg["col"])
    else:
        times = trials[cfg["col"]].to_numpy()

    return times[responding]


def run_zeta_all_neurons(trains, labels, event_times, dur_s=DEFAULT_DUR_S, n_resamp=DEFAULT_RESAMP):
    """Run zetatest for every neuron against a single set of event times.

    Returns a DataFrame with one row per neuron.
    """
    rows = []
    for i, (spikes, label) in enumerate(zip(trains, labels)):
        if len(spikes) < 10:
            rows.append({"neuron_idx": i, "label": label,
                         "zeta": np.nan, "p_zeta": 1.0,
                         "mean_z": np.nan, "p_mean": 1.0,
                         "latency_s": np.nan})
            continue

        try:
            dblZETA, dblMeanZ, dblZETA_p, intZETAIdx, dblMeanZ_p, intPeakIdx, \
                dblLatency, vecLatencies, dZETA = zetatest(
                    spikes, event_times,
                    dblUseMaxDur=dur_s,
                    intResampNum=n_resamp,
                    boolPlot=False,
                )
        except Exception as exc:
            print(f"  [skip] neuron {i} ({label}): {exc}")
            rows.append({"neuron_idx": i, "label": label,
                         "zeta": np.nan, "p_zeta": 1.0,
                         "mean_z": np.nan, "p_mean": 1.0,
                         "latency_s": np.nan})
            continue

        rows.append({
            "neuron_idx": i,
            "label":      label,
            "zeta":       dblZETA,
            "p_zeta":     dblZETA_p,
            "mean_z":     dblMeanZ,
            "p_mean":     dblMeanZ_p,
            "latency_s":  dblLatency,
        })

    df = pd.DataFrame(rows).sort_values("p_zeta").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ifr_grid(trains, results_df, event_times, dur_s, alpha, top_n, event_name, args):
    """Plot instantaneous firing rate for the top_n significant neurons."""
    sig = results_df[results_df["p_zeta"] < alpha].head(top_n)
    if sig.empty:
        print(f"  No significant neurons (α={alpha}) for event '{event_name}'.")
        return

    ncols = min(4, len(sig))
    nrows = int(np.ceil(len(sig) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             sharey=False, squeeze=False)
    axes_flat = axes.flatten()

    t_vec = np.linspace(0, dur_s, 500)

    for ax_i, (_, row) in enumerate(sig.iterrows()):
        ax = axes_flat[ax_i]
        spikes = trains[int(row["neuron_idx"])]

        try:
            vecT, vecRate, vecSD, _ = zeta_ifr(
                spikes, event_times,
                dblUseMaxDur=dur_s,
                intSmoothSd=0,
                dblMinScale=None,
            )
            ax.plot(vecT, vecRate, color="steelblue", lw=1.5)
        except Exception as exc:
            ax.text(0.5, 0.5, f"IFR error:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)

        ax.axvline(0, color="gray", lw=0.8, ls="--")
        if not np.isnan(row["latency_s"]):
            ax.axvline(row["latency_s"], color="tomato", lw=1.0, ls=":")

        short_label = row["label"].split("|")[0].strip()
        ax.set_title(f"{short_label}\np={row['p_zeta']:.2e}", fontsize=8)
        ax.set_xlabel("Time from event (s)", fontsize=7)
        ax.set_ylabel("IFR (Hz)", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax_i in range(len(sig), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(f"ZETA — event: {event_name} | session: {SESSION} | α={alpha}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    maybe_save(fig, args, prefix=f"zeta_{event_name}")
    plt.show()


def plot_pvalue_overview(all_results, alpha):
    """Bar chart of fraction of significant neurons per event."""
    events  = list(all_results.keys())
    n_total = [len(df) for df in all_results.values()]
    n_sig   = [int((df["p_zeta"] < alpha).sum()) for df in all_results.values()]
    fracs   = [s / n for s, n in zip(n_sig, n_total)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(events, fracs, color="steelblue", edgecolor="white")
    ax.bar_label(bars, labels=[f"{s}/{n}" for s, n in zip(n_sig, n_total)], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(alpha, color="tomato", ls="--", lw=1, label=f"α={alpha}")
    ax.set_ylabel("Fraction of significant neurons")
    ax.set_title(f"ZETA responsiveness — session {SESSION}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ZETA responsiveness test for all neurons.")
    p.add_argument("--event",  default="all",
                   choices=list(EVENTS.keys()) + ["all"],
                   help="Which event to align to (default: all).")
    p.add_argument("--dur",    type=float, default=DEFAULT_DUR_S,
                   help=f"Analysis window in seconds (default: {DEFAULT_DUR_S}).")
    p.add_argument("--resamp", type=int,   default=DEFAULT_RESAMP,
                   help=f"Jitter iterations for null distribution (default: {DEFAULT_RESAMP}).")
    p.add_argument("--alpha",  type=float, default=0.05,
                   help="Significance threshold (default: 0.05).")
    p.add_argument("--top",    type=int,   default=8,
                   help="Number of top neurons to plot IFR for (default: 8).")
    add_save_arg(p)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Session: {SESSION}")
    trains, labels = get_spike_trains()
    trials = load_trials_sync()
    sr_df  = load_sr()
    sr     = sr_df["SamplingRate_Hz"].iloc[0]

    print(f"Loaded {len(trains)} neurons, {len(trials)} trials, SR={sr} Hz")

    events_to_run = list(EVENTS.keys()) if args.event == "all" else [args.event]
    all_results = {}

    for event_name in events_to_run:
        event_times = get_event_times(event_name, trials, sr)
        print(f"\n--- Event: {event_name} ({len(event_times)} onsets) ---")

        results = run_zeta_all_neurons(
            trains, labels, event_times,
            dur_s=args.dur, n_resamp=args.resamp,
        )
        all_results[event_name] = results

        n_sig = int((results["p_zeta"] < args.alpha).sum())
        print(f"Significant (α={args.alpha}): {n_sig} / {len(results)} neurons")
        print(results[["label", "zeta", "p_zeta", "latency_s"]].head(10).to_string(index=False))

        plot_ifr_grid(trains, results, event_times,
                      dur_s=args.dur, alpha=args.alpha,
                      top_n=args.top, event_name=event_name, args=args)

    if len(events_to_run) > 1:
        plot_pvalue_overview(all_results, args.alpha)


if __name__ == "__main__":
    main()
