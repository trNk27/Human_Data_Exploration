"""Demo figures for a ZETA-detected reward-responsive neuron.

Produces two PNGs for a single neuron, both aligned to reward onset:

  1. <session>/psth_n<idx>_alltrials_reward.png
        Reward-aligned PSTH (the neuron's firing response to reward), with the
        ZETA latency marked.

  2. <session>/zeta_detection_n<idx>_reward.png
        The ZETA machinery behind the detection, two stacked panels:
          top    — cumulative spike count vs the flat-rate baseline
          bottom — the deviation (observed - baseline) = the ZETA statistic,
                   against the jittered null traces, with the ZETA peak marked.

The one-sample ZETA (zetapy.zetatest) tests whether a neuron's spike timing is
non-uniform within the post-reward window; the deviation's maximum is the ZETA.

Usage:
    python scripts/zeta_reward_demo.py
    python scripts/zeta_reward_demo.py --session 20250521 --neuron 113

NOTE: zetatest defaults to boolParallel=True, which on Windows spawns child
processes that re-import this module — so we pass boolParallel=False (a single
neuron is cheap) and guard the entry point under __main__.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from zetapy import zetatest

from session import Session
from explore import Panel
from utils import RESULTS_DIR
from analysis.zeta_analysis import responding_event_times

DEFAULT_SESSION = "20250521"
DEFAULT_NEURON  = 113        # unit226 | AG ele051 (su): strong excitatory reward response
DUR_S           = 2.0        # post-reward analysis window (matches batch ZETA)
RESAMP          = 250


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=DEFAULT_SESSION)
    p.add_argument("--neuron", type=int, default=DEFAULT_NEURON,
                   help="Neuron column index (default: a known reward-responsive unit).")
    p.add_argument("--dur", type=float, default=DUR_S,
                   help=f"Post-reward window in seconds (default: {DUR_S}).")
    p.add_argument("--resamp", type=int, default=RESAMP,
                   help=f"ZETA jitter iterations (default: {RESAMP}).")
    return p.parse_args()


def plot_psth(sess, idx, dZETA, args):
    """Reward-aligned PSTH for the neuron, with the ZETA latency marked."""
    panel = sess.neuron(idx).psth(align="reward", pre_ms=500,
                                  post_ms=int(args.dur * 1000),
                                  bin_ms=50, sigma_ms=50)
    ax = panel.ax
    lat_ms = dZETA["dblLatencyZETA"] * 1000
    ax.axvline(lat_ms, color="purple", lw=1.2, ls=":",
               label=f"ZETA latency = {lat_ms:.0f} ms")
    ax.legend(fontsize=6, loc="upper right")
    panel.fig.suptitle(
        f"Reward-responsive neuron {idx}  ·  session {sess.id}  ·  "
        f"ZETA p = {dZETA['dblZetaP']:.1e}", fontsize=9)
    return panel


def plot_zeta_detection(sess, idx, label, dZETA, args):
    """Cumulative spike count + the ZETA deviation/detection, two stacked panels."""
    t        = dZETA["vecSpikeT"]
    frac     = dZETA["vecRealFrac"]          # observed cumulative (fraction of spikes)
    baseline = dZETA["vecRealFracLinear"]    # flat-rate (uniform) expectation
    dev      = dZETA["vecRealDeviation"]      # observed - baseline (the ZETA statistic)
    z_t      = dZETA["dblLatencyZETA"]
    z_dev    = dZETA["dblZETADeviation"]
    rand_t   = dZETA["cellRandTime"]
    rand_dev = dZETA["cellRandDeviation"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    # --- top: cumulative spike count vs flat-rate baseline ---
    ax_top.plot(t, frac, color="steelblue", lw=1.6, label="Observed cumulative")
    ax_top.plot(t, baseline, color="gray", lw=1.2, ls="--",
                label="Flat-rate baseline")
    ax_top.axvline(z_t, color="purple", lw=1.0, ls=":")
    ax_top.set_ylabel("Cumulative fraction of spikes")
    ax_top.set_title("Cumulative spike count", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper left")
    ax_top.spines[["top", "right"]].set_visible(False)

    # --- bottom: deviation = ZETA detection, against the jittered null ---
    n_null = min(len(rand_t), 50)
    for i in range(n_null):
        ax_bot.plot(rand_t[i], rand_dev[i], color=[0.8, 0.8, 0.8], lw=0.5,
                    zorder=1, label="Jittered null" if i == 0 else None)
    ax_bot.axhline(0, color="black", lw=0.6)
    ax_bot.plot(t, dev, color="firebrick", lw=1.6, zorder=3,
                label="Observed deviation")
    ax_bot.plot(z_t, z_dev, "o", color="purple", ms=9, zorder=4,
                label=f"ZETA = {dZETA['dblZETA']:.2f}  (t = {z_t*1000:.0f} ms)")
    ax_bot.set_xlabel("Time after reward onset (s)")
    ax_bot.set_ylabel("Spiking-density deviation")
    ax_bot.set_title("ZETA detection (deviation from baseline)", fontsize=10)
    ax_bot.legend(fontsize=8, loc="upper left")
    ax_bot.spines[["top", "right"]].set_visible(False)

    short = label.split("|")[0].strip()
    fig.suptitle(
        f"ZETA reward detection — neuron {idx} ({short})  ·  session {sess.id}\n"
        f"ZETA = {dZETA['dblZETA']:.2f},  p = {dZETA['dblZetaP']:.2e}  "
        f"(window 0–{args.dur:.0f}s post-reward, {args.resamp} resamples)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return Panel(fig, (ax_top, ax_bot),
                 autoname=(sess.id, ["zeta_detection", f"n{idx}", "reward"]))


def main():
    args = parse_args()
    sess = Session(args.session)
    trains, labels = sess.spike_trains
    idx = args.neuron
    label = labels[idx]
    print(f"Session {args.session} | neuron {idx}: {label}")

    ev_times = responding_event_times("reward", sess.trials, sess.sampling_rate)
    print(f"Responding reward onsets: {len(ev_times)}")

    dblP, dZETA, dRate = zetatest(trains[idx], ev_times, dblUseMaxDur=args.dur,
                                  intResampNum=args.resamp, boolReturnRate=True,
                                  boolPlot=False, boolParallel=False)
    print(f"ZETA = {dZETA['dblZETA']:.3f}, p = {dblP:.3e}, "
          f"latency = {dZETA['dblLatencyZETA']*1000:.0f} ms, "
          f"deviation = {dZETA['dblZETADeviation']:+.3f} "
          f"({'excitation' if dZETA['dblZETADeviation'] > 0 else 'inhibition'})")

    out_dir = os.path.join(RESULTS_DIR, "figures", sess.id)
    os.makedirs(out_dir, exist_ok=True)

    psth_panel = plot_psth(sess, idx, dZETA, args)
    psth_panel.save()

    zeta_panel = plot_zeta_detection(sess, idx, label, dZETA, args)
    zeta_panel.save()


if __name__ == "__main__":
    main()
