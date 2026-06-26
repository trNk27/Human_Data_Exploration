"""Demo figures for a ZETA2-detected outcome-difference neuron.

The two-sample analogue of scripts/zeta_reward_demo.py. Where the one-sample
ZETA asks "does this neuron respond to reward at all?", the two-sample ZETA2
(zetapy.zetatest2) asks "does the reward-aligned response *differ* between two
outcomes?". Both conditions are aligned to reward onset; the test compares their
cumulative spike-fraction curves.

Contrasts (same as analysis/zeta_outcome.py, conditions from utils.CONDITIONS):

    reward   G+R vs G+N   — effect of reward (choice held = gamble)
    choice   G+R vs S+R   — effect of choice (reward held constant)

Produces two PNGs for a single neuron:

  1. <session>/psth_n<idx>_<a>_vs_<b>_reward.png
        Reward-aligned PSTH overlaying the two contrasted conditions, so you can
        see the firing difference the test operates on. ZETA2 latency marked.

  2. <session>/zeta2_detection_n<idx>_<contrast>_reward.png
        The ZETA2 machinery, two stacked panels:
          top    — the two conditions' cumulative spike fractions
          bottom — their mean-centred difference (A - B) = the ZETA2 statistic,
                   against the jittered null traces, with the ZETA2 peak marked.

Usage:
    python scripts/zeta2_outcome_demo.py
    python scripts/zeta2_outcome_demo.py --session 20250521 --neuron 113 --contrast reward

NOTE: the spike zetatest2 runs single-process internally (boolParallel=False),
so it does not hit the Windows fork-bomb trap that one-sample zetatest can; we
still guard the entry point under __main__ out of habit.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from zetapy import zetatest2

from session import Session
from explore import Panel
from compute import compute_psth
from utils import RESULTS_DIR, CONDITIONS

DEFAULT_SESSION = "20250521"
DEFAULT_NEURON  = 113        # unit226 | AG ele051 (su): strong excitatory reward response
DEFAULT_CONTRAST = "reward"
DUR_S           = 2.0        # post-reward analysis window (matches batch ZETA)
RESAMP          = 250

# Each contrast compares condition "a" against condition "b" (same neuron).
CONTRASTS = {
    "reward": {"a": "G+R", "b": "G+N", "desc": "effect of reward (gamble: rewarded vs not)"},
    "choice": {"a": "G+R", "b": "S+R", "desc": "effect of choice (rewarded: gamble vs safe)"},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=DEFAULT_SESSION)
    p.add_argument("--neuron", type=int, default=DEFAULT_NEURON,
                   help="Neuron column index (default: a known reward-responsive unit).")
    p.add_argument("--contrast", default=DEFAULT_CONTRAST, choices=list(CONTRASTS),
                   help=f"Which two conditions to compare (default: {DEFAULT_CONTRAST}).")
    p.add_argument("--dur", type=float, default=DUR_S,
                   help=f"Post-reward window in seconds (default: {DUR_S}).")
    p.add_argument("--resamp", type=int, default=RESAMP,
                   help=f"ZETA2 jitter iterations (default: {RESAMP}).")
    return p.parse_args()


def plot_psth(sess, idx, label, cfg, dZETA, args):
    """Reward-aligned PSTH overlaying the two contrasted conditions."""
    pre_ms, post_ms, bin_ms, sigma_ms = 500, int(args.dur * 1000), 50, 50
    train = sess.spike_trains[0][idx]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for cond in (cfg["a"], cfg["b"]):
        align_times = sess.aligned_event_times("reward", cond)
        n_tr = int(np.sum(np.isfinite(align_times)))
        centres, rate = compute_psth(train, align_times, pre_ms, post_ms, bin_ms)
        rate = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
        ax.plot(centres, rate, color=CONDITIONS[cond]["color"], lw=1.6,
                label=f"{CONDITIONS[cond]['label']} (n={n_tr})")

    ax.axvline(0, color="red", lw=1.0, ls="--", label="Reward (align)")
    lat_ms = dZETA["dblZetaT"] * 1000
    ax.axvline(lat_ms, color="purple", lw=1.2, ls=":",
               label=f"ZETA2 latency = {lat_ms:.0f} ms")
    ax.set_xlabel("Time rel. to reward onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    short = label.split("|")[0].strip()
    fig.suptitle(
        f"Outcome difference — neuron {idx} ({short})  ·  session {sess.id}  ·  "
        f"{cfg['a']} vs {cfg['b']}  ·  ZETA2 p = {dZETA['dblZetaP']:.1e}", fontsize=9)
    return Panel(fig, ax,
                 autoname=(sess.id, ["psth", f"n{idx}", f"{cfg['a']}_vs_{cfg['b']}", "reward"]))


def plot_zeta2_detection(sess, idx, label, cfg, dZETA, args):
    """Two conditions' cumulative fractions + the ZETA2 difference/detection."""
    t        = dZETA["vecSpikeT"]
    frac1    = dZETA["vecRealFrac1"]     # cumulative spikes/trial, condition A
    frac2    = dZETA["vecRealFrac2"]     # cumulative spikes/trial, condition B
    diff     = dZETA["vecRealDiff"]      # mean-centred (A - B) = the ZETA2 statistic
    z_t      = dZETA["dblZetaT"]
    z_dev    = dZETA["dblZETADeviation"]
    rand_t   = dZETA["cellRandTime"]
    rand_d   = dZETA["cellRandDiff"]
    col_a    = CONDITIONS[cfg["a"]]["color"]
    col_b    = CONDITIONS[cfg["b"]]["color"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    # --- top: the two cumulative spike-fraction curves ---
    ax_top.plot(t, frac1, color=col_a, lw=1.6, label=f"{cfg['a']} cumulative")
    ax_top.plot(t, frac2, color=col_b, lw=1.6, label=f"{cfg['b']} cumulative")
    ax_top.axvline(z_t, color="purple", lw=1.0, ls=":")
    ax_top.set_ylabel("Cumulative spikes per trial")
    ax_top.set_title("Cumulative spike count per condition", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper left")
    ax_top.spines[["top", "right"]].set_visible(False)

    # --- bottom: the difference = ZETA2 detection, against the jittered null ---
    if rand_t is not None and rand_d is not None:
        n_null = min(len(rand_t), 50)
        for i in range(n_null):
            ax_bot.plot(rand_t[i], rand_d[i], color=[0.8, 0.8, 0.8], lw=0.5,
                        zorder=1, label="Jittered null" if i == 0 else None)
    ax_bot.axhline(0, color="black", lw=0.6)
    ax_bot.plot(t, diff, color="firebrick", lw=1.6, zorder=3,
                label=f"Observed difference ({cfg['a']} - {cfg['b']})")
    ax_bot.plot(z_t, z_dev, "o", color="purple", ms=9, zorder=4,
                label=f"ZETA2 = {dZETA['dblZETA']:.2f}  (t = {z_t*1000:.0f} ms)")
    ax_bot.set_xlabel("Time after reward onset (s)")
    ax_bot.set_ylabel("Difference in cumulative spikes/trial")
    ax_bot.set_title("ZETA2 detection (difference between conditions)", fontsize=10)
    ax_bot.legend(fontsize=8, loc="upper left")
    ax_bot.spines[["top", "right"]].set_visible(False)

    short = label.split("|")[0].strip()
    fig.suptitle(
        f"ZETA2 outcome detection — neuron {idx} ({short})  ·  session {sess.id}\n"
        f"{cfg['a']} vs {cfg['b']}  ·  ZETA2 = {dZETA['dblZETA']:.2f},  "
        f"p = {dZETA['dblZetaP']:.2e}  "
        f"(window 0-{args.dur:.0f}s post-reward, {args.resamp} resamples)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return Panel(fig, (ax_top, ax_bot),
                 autoname=(sess.id, ["zeta2_detection", f"n{idx}", f"{cfg['a']}_vs_{cfg['b']}"]))


def main():
    args = parse_args()
    cfg  = CONTRASTS[args.contrast]
    sess = Session(args.session)
    trains, labels = sess.spike_trains
    idx = args.neuron
    label = labels[idx]
    print(f"Session {args.session} | neuron {idx}: {label}")
    print(f"Contrast '{args.contrast}': {cfg['a']} vs {cfg['b']} - {cfg['desc']}")

    cond_times = sess.condition_event_times(event="reward")
    events_a = cond_times[cfg["a"]]
    events_b = cond_times[cfg["b"]]
    print(f"Reward onsets: {cfg['a']}={len(events_a)}, {cfg['b']}={len(events_b)}")

    dblP, dZETA = zetatest2(trains[idx], events_a, trains[idx], events_b,
                            dblUseMaxDur=args.dur, intResampNum=args.resamp,
                            boolPlot=False)
    print(f"ZETA2 = {dZETA['dblZETA']:.3f}, p = {dblP:.3e}, "
          f"latency = {dZETA['dblZetaT']*1000:.0f} ms, "
          f"peak diff = {dZETA['dblZETADeviation']:+.3f} "
          f"({'>0: ' + cfg['a'] + ' leads' if dZETA['dblZETADeviation'] > 0 else '<0: ' + cfg['b'] + ' leads'})")

    out_dir = os.path.join(RESULTS_DIR, "figures", sess.id)
    os.makedirs(out_dir, exist_ok=True)

    plot_psth(sess, idx, label, cfg, dZETA, args).save()
    plot_zeta2_detection(sess, idx, label, cfg, dZETA, args).save()


if __name__ == "__main__":
    main()
