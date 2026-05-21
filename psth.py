"""Peristimulus time histogram (PSTH).

Aligns each neuron's spike train to a behavioural event, bins spikes across
trials, and plots firing rate in Hz. Vertical lines mark the mean timing of
other key events relative to the alignment point.

All time parameters (bin sizes, windows) are in milliseconds. Internal spike
and event times remain in seconds, as stored in the data.
"""

import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils import (
    EVENTS, EVENT_STYLE, CONDITIONS,
    get_spike_trains, load_trials_sync, load_sr,
    event_times as event_times_for, condition_masks,
    select_neurons, add_session_arg, add_selection_args,
    add_save_arg, maybe_save, handle_list, session_data_dir,
)


def compute_psth(spike_times, event_times_s, pre_ms, post_ms, bin_ms):
    """Return (bin_centres_ms, firing_rate_hz) for spikes aligned to events."""
    bin_s  = bin_ms  / 1000
    pre_s  = pre_ms  / 1000
    post_s = post_ms / 1000

    edges  = np.arange(-pre_s, post_s + bin_s / 2, bin_s)
    counts = np.zeros(len(edges) - 1, dtype=np.float64)
    n_valid = 0

    for t_ev in event_times_s:
        if not np.isfinite(t_ev):
            continue
        aligned = spike_times - t_ev
        in_win  = aligned[(aligned >= -pre_s) & (aligned < post_s)]
        counts += np.histogram(in_win, bins=edges)[0]
        n_valid += 1

    centres_s = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / (n_valid * bin_s) if n_valid > 0 else counts
    return centres_s * 1000, rate


def plot_psth(neuron_indices=None, area=None, event="cue",
              pre_ms=500, post_ms=1000, bin_ms=50, sigma_ms=None,
              by_condition=False, data_dir=None, session=None):
    """Plot one PSTH subplot per neuron, optionally split by (arm, reward) condition."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {list(EVENTS)}")

    trains, labels = get_spike_trains(data_dir=data_dir)
    trains, labels = select_neurons(trains, labels, indices=neuron_indices, area=area)

    trials = load_trials_sync(data_dir=data_dir)
    sr     = load_sr(data_dir=data_dir)["SamplingRate_Hz"].iloc[0]

    align_times = event_times_for(trials, sr, event)
    responding  = trials["NotResponding"].to_numpy() != 1
    align_times = np.where(responding, align_times, np.nan)

    cond_masks  = None
    cond_counts = None
    if by_condition:
        cond_masks  = condition_masks(trials)
        cond_counts = {name: int(np.sum(m)) for name, m in cond_masks.items()}

    # Mean timing of other events relative to the alignment point (in ms).
    markers = {}
    for name in EVENTS:
        if name == event:
            continue
        rel = event_times_for(trials, sr, name) - align_times
        if not np.any(np.isfinite(rel)):
            continue
        mean_rel_ms = float(np.nanmean(rel)) * 1000
        if -pre_ms <= mean_rel_ms <= post_ms:
            markers[name] = mean_rel_ms

    n     = len(trains)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]

        if by_condition:
            for name, cfg in CONDITIONS.items():
                cond_align = align_times[cond_masks[name]]
                if cond_align.size == 0:
                    continue
                centres, rate = compute_psth(train, cond_align, pre_ms, post_ms, bin_ms)
                if sigma_ms is not None:
                    rate = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
                    ax.plot(centres, rate, color=cfg["color"], linewidth=1.2,
                            label=f"{cfg['label']} (n={cond_counts[name]})")
                else:
                    ax.step(centres, rate, where="mid", color=cfg["color"],
                            linewidth=1.0,
                            label=f"{cfg['label']} (n={cond_counts[name]})")
        else:
            centres, rate = compute_psth(train, align_times, pre_ms, post_ms, bin_ms)
            ax.bar(centres, rate, width=bin_ms, color="steelblue",
                   edgecolor="none", alpha=0.6, label="_nolegend_")
            if sigma_ms is not None:
                smoothed = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
                ax.plot(centres, smoothed, color="navy", linewidth=1.2, label="_nolegend_")

        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   label=f"{EVENTS[event]['label']} (align)")
        for name, t_rel_ms in markers.items():
            ax.axvline(t_rel_ms, linewidth=0.8,
                       label=EVENTS[name]["label"], **EVENT_STYLE[name])

        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
        ax.set_ylabel("Firing rate (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    smooth_str = f", smoothed sigma={sigma_ms:.0f} ms" if sigma_ms is not None else ""
    cond_str   = "  |  split by (arm, reward)" if by_condition else ""
    fig.suptitle(
        f"PSTH — session {session}  |  aligned to: {event}{cond_str}"
        f"  (pre={pre_ms}ms, post={post_ms}ms, bin={bin_ms}ms{smooth_str})",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peristimulus time histogram (PSTH)")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--event", type=str, default="cue", choices=list(EVENTS),
                        help="Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",   type=float, default=500,  help="ms before event (default: 500)")
    parser.add_argument("--post",  type=float, default=1000, help="ms after event (default: 1000)")
    parser.add_argument("--bin",   type=float, default=50,   help="Bin width in ms (default: 50)")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian smoothing SD in ms (default: off)")
    parser.add_argument("--by-condition", action="store_true",
                        help="Overlay one curve per (arm, reward) condition")
    add_save_arg(parser)
    args = parser.parse_args()

    data_dir = session_data_dir(args.session)
    if handle_list(args, data_dir=data_dir):
        raise SystemExit

    fig, _ = plot_psth(
        neuron_indices=args.neurons, area=args.area,
        event=args.event, pre_ms=args.pre, post_ms=args.post,
        bin_ms=args.bin, sigma_ms=args.sigma,
        by_condition=args.by_condition,
        data_dir=data_dir, session=args.session,
    )
    maybe_save(fig, args, prefix="psth")
    plt.show()
