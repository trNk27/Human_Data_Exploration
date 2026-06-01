"""Population firing-rate heatmap: z-scored activity across all neurons in a session.

Aligns every neuron's spikes to trial_start, bins across all responding trials,
z-scores each neuron's time course, and renders a neurons x time heatmap.
Vertical lines mark the median cue and reward onsets relative to trial_start.

By default neurons are sorted by the time of their peak z-score (classic
"sorted population" visualization). Pass --sort-by region to group by brain
area instead.

Usage:
    python population_heatmap.py --session 20250521 --save
    python population_heatmap.py --bin-ms 25 --smooth-ms 100
    python population_heatmap.py --sort-by region --vmax 3 --save
"""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Make the repo root importable when run as `python analysis/<file>.py`.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute import compute_psth
from session import Session
from utils import (
    EVENT_STYLE,
    add_save_arg, add_session_arg, maybe_save,
)

_REGION_RE = re.compile(r"\|\s*(\S+)\s+ele")


def parse_region(label):
    m = _REGION_RE.search(label)
    return m.group(1) if m else "?"


def build_population_matrix(trains, labels, trial_starts, post_ms, bin_ms,
                            smooth_ms, min_spikes):
    """Return (matrix, centres_ms, kept_mask).

    matrix: (n_kept, n_bins) z-scored firing rate.
    centres_ms: 1-D array of bin centre times in ms from trial_start.
    kept_mask: boolean array len(trains) — True where neuron was retained.
    """
    kept_rows   = []
    kept_mask   = np.zeros(len(trains), dtype=bool)

    for i, (spikes, label) in enumerate(zip(trains, labels)):
        if len(spikes) < min_spikes:
            continue
        _, rate = compute_psth(spikes, trial_starts, pre_ms=0,
                               post_ms=post_ms, bin_ms=bin_ms)
        if smooth_ms > 0:
            sigma = smooth_ms / bin_ms
            rate  = gaussian_filter1d(rate, sigma=sigma)
        std = rate.std()
        z = (rate - rate.mean()) / std if std > 0 else np.zeros_like(rate)
        kept_rows.append(z)
        kept_mask[i] = True

    # Recompute centres from a dummy compute_psth call (same params, no spikes).
    dummy = np.array([0.0])
    centres_ms, _ = compute_psth(dummy, np.array([0.0]),
                                 pre_ms=0, post_ms=post_ms, bin_ms=bin_ms)

    return np.array(kept_rows), centres_ms, kept_mask


def sort_order(matrix, labels, sort_by):
    """Return row indices that sort the matrix according to sort_by."""
    n = len(labels)
    if sort_by == "peak_time":
        return np.argsort(np.argmax(matrix, axis=1))
    if sort_by == "region":
        regions = [parse_region(l) for l in labels]
        return np.argsort(regions, kind="stable")
    return np.arange(n)


def plot_heatmap(matrix, centres_ms, labels, cue_lag_ms, reward_lag_ms,
                 sort_by, vmax, session, args):
    order   = sort_order(matrix, labels, sort_by)
    sorted_m = matrix[order]
    sorted_l = [labels[i] for i in order]
    n, n_bins = sorted_m.shape

    fig_h = max(4.0, 0.025 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ext = [centres_ms[0], centres_ms[-1], n, 0]
    im  = ax.imshow(sorted_m, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")

    ax.axvline(cue_lag_ms,    label=f"Cue (median {cue_lag_ms:.0f} ms)",
               **{k: v for k, v in EVENT_STYLE["cue"].items()},    lw=1.5)
    ax.axvline(reward_lag_ms, label=f"Reward (median {reward_lag_ms:.0f} ms)",
               **{k: v for k, v in EVENT_STYLE["reward"].items()}, lw=1.5)

    ax.set_xlabel("Time from trial start (ms)", fontsize=10)
    ax.set_ylabel("Neuron (sorted by peak)" if sort_by == "peak_time"
                  else f"Neuron (sorted by {sort_by})", fontsize=10)
    ax.set_title(f"Population activity — session {session}  "
                 f"({n} neurons, {n_bins} bins x {int(round((centres_ms[1]-centres_ms[0])))} ms)",
                 fontsize=10)

    if sort_by == "region":
        regions = [parse_region(l) for l in sorted_l]
        boundaries = [0] + [i+1 for i in range(n-1)
                            if regions[i] != regions[i+1]] + [n]
        for b in boundaries[1:-1]:
            ax.axhline(b, color="black", lw=0.5, alpha=0.5)
        mids = [(boundaries[j] + boundaries[j+1]) / 2
                for j in range(len(boundaries)-1)]
        unique_r = [regions[boundaries[j]] for j in range(len(boundaries)-1)]
        ax.set_yticks(mids)
        ax.set_yticklabels(unique_r, fontsize=8)
    else:
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, label="z-score", shrink=0.6, pad=0.02)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    maybe_save(fig, args, prefix=f"population_heatmap_{sort_by}",
               subdir=None)   # saves to cwd or --save path; user can redirect
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_arg(p)
    p.add_argument("--bin-ms", type=float, default=50,
                   help="Time bin width in ms (default 50).")
    p.add_argument("--smooth-ms", type=float, default=0,
                   help="Gaussian smoothing sigma in ms applied before z-scoring "
                        "(default 0 = off). Try 100 for noisy data.")
    p.add_argument("--post-ms", type=float, default=None,
                   help="X-axis upper bound in ms from trial_start "
                        "(default: 90th percentile of trial duration).")
    p.add_argument("--sort-by", default="peak_time",
                   choices=["peak_time", "region", "none"],
                   help="How to order neurons on the y-axis (default: peak_time).")
    p.add_argument("--vmax", type=float, default=2.5,
                   help="Z-score colour clamp (default 2.5).")
    p.add_argument("--min-spikes", type=int, default=20,
                   help="Skip neurons with fewer than this many spikes (default 20).")
    add_save_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    sess = Session(args.session)
    print(f"Session: {args.session}")

    trains, labels  = sess.spike_trains
    print(f"Loaded {len(trains)} neurons, {len(sess.trials)} trials, SR={sess.sampling_rate} Hz")

    trial_starts    = sess.event_times("trial_start")[sess.responding_mask]
    cue_times       = sess.event_times("cue")[sess.responding_mask]
    reward_times    = sess.event_times("reward")[sess.responding_mask]
    trial_durations = sess.trials["TrialDuration_s"].to_numpy()[sess.responding_mask]

    post_ms = args.post_ms if args.post_ms is not None \
              else float(np.percentile(trial_durations, 90) * 1000)
    print(f"Trial window: 0 -> {post_ms:.0f} ms  |  {sess.responding_mask.sum()} responding trials")

    cue_lag_ms    = float(np.median(cue_times    - trial_starts) * 1000)
    reward_lag_ms = float(np.median(reward_times - trial_starts) * 1000)
    print(f"Median cue lag: {cue_lag_ms:.0f} ms  |  "
          f"Median reward lag: {reward_lag_ms:.0f} ms")

    print(f"Computing z-scored PSTHs "
          f"(bin={args.bin_ms} ms, smooth={args.smooth_ms} ms) ...")
    matrix, centres_ms, kept = build_population_matrix(
        trains, labels, trial_starts,
        post_ms=post_ms, bin_ms=args.bin_ms,
        smooth_ms=args.smooth_ms, min_spikes=args.min_spikes,
    )
    kept_labels = [labels[i] for i in range(len(labels)) if kept[i]]
    print(f"Plotting {len(kept_labels)} neurons (skipped "
          f"{len(trains) - kept.sum()} with < {args.min_spikes} spikes)")

    plot_heatmap(matrix, centres_ms, kept_labels,
                 cue_lag_ms, reward_lag_ms,
                 sort_by=args.sort_by, vmax=args.vmax,
                 session=args.session, args=args)


if __name__ == "__main__":
    main()
