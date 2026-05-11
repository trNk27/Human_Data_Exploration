"""Spike raster plots from STMtx.mat.

Two modes:
  plot_raster         — full-recording raster, one row per neuron.
  plot_aligned_raster — trial-by-trial raster aligned to a behavioural event,
                        one subplot per neuron, one row per trial.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from utils import SESSION, get_spike_trains, load_trials_sync, load_sr, sp_to_s, MAX_NEURONS, add_save_arg, maybe_save
from psth import EVENTS, EVENT_STYLE, CONDITIONS


# ---------------------------------------------------------------------------
# Full-recording raster
# ---------------------------------------------------------------------------

def plot_raster(t_start=None, t_end=None, neuron_indices=None, area=None):
    trains, labels = get_spike_trains()

    if neuron_indices is not None:
        trains = [trains[i] for i in neuron_indices]
        labels = [labels[i] for i in neuron_indices]

    if area is not None:
        mask = [area.lower() in lbl.lower() for lbl in labels]
        trains = [t for t, m in zip(trains, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    if not trains:
        raise ValueError("No neurons match the given selection.")

    if t_start is not None or t_end is not None:
        lo = -np.inf if t_start is None else t_start
        hi =  np.inf if t_end   is None else t_end
        trains = [s[(s >= lo) & (s <= hi)] for s in trains]

    fig, ax = plt.subplots(figsize=(12, max(3, 0.25 * len(trains))))

    # eventplot draws one tick per spike, one row per neuron.
    ax.eventplot(trains, colors="black", linelengths=0.8, linewidths=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-0.5, len(trains) - 0.5)
    if t_start is not None or t_end is not None:
        ax.set_xlim(t_start, t_end)
    ax.set_title(f"Spike raster — session {SESSION}  ({len(trains)} units)")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Aligned raster
# ---------------------------------------------------------------------------

def compute_aligned_raster(spike_times, event_times_s, pre_ms, post_ms):
    """Return a list of 1-D arrays (ms relative to event), one per trial.

    Non-finite event times (non-responding trials) yield empty arrays.
    """
    pre_s  = pre_ms  / 1000
    post_s = post_ms / 1000
    result = []
    for t_ev in event_times_s:
        if not np.isfinite(t_ev):
            result.append(np.array([]))
            continue
        aligned = spike_times - t_ev
        result.append(aligned[(aligned >= -pre_s) & (aligned < post_s)] * 1000)
    return result


def plot_aligned_raster(neuron_indices=None, area=None, event="cue",
                        pre_ms=500, post_ms=1000, by_condition=False):
    """Plot a trial-by-trial raster aligned to a behavioural event.

    One subplot per neuron. Each row is a trial; tick marks are the spike
    times relative to the alignment event. Vertical lines show the mean
    timing of other key events. With by_condition=True, tick colours
    distinguish the (arm × reward) conditions defined in psth.CONDITIONS.
    """
    if event not in EVENTS:
        raise ValueError(f"event must be one of {list(EVENTS)}")

    trains, labels = get_spike_trains()

    if neuron_indices is not None:
        trains = [trains[i] for i in neuron_indices]
        labels = [labels[i] for i in neuron_indices]

    if area is not None:
        mask   = [area.lower() in lbl.lower() for lbl in labels]
        trains = [t for t, m in zip(trains, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    if not trains:
        raise ValueError("No neurons match the given selection.")

    if len(trains) > MAX_NEURONS:
        raise ValueError(
            f"{len(trains)} neurons selected — limit is {MAX_NEURONS}. "
            "Use --neurons or --area to narrow the selection."
        )

    trials = load_trials_sync()
    sr     = load_sr()["SamplingRate_Hz"].iloc[0]

    align_times = sp_to_s(trials, sr, EVENTS[event])
    responding  = trials["NotResponding"].to_numpy() != 1
    align_times = np.where(responding, align_times, np.nan)

    # One colour per trial; default black, overwritten per condition.
    trial_colors = ["black"] * len(align_times)
    if by_condition:
        arm_col = trials["ChosenArm_G1S0"].to_numpy()
        rew_col = trials["Rewarded"].to_numpy()
        for name, cfg in CONDITIONS.items():
            mask = (arm_col == cfg["arm"]) & (rew_col == cfg["rewarded"])
            for i, m in enumerate(mask):
                if m:
                    trial_colors[i] = cfg["color"]

    # Mean timing of other events relative to alignment point.
    markers = {}
    for name, col in EVENTS.items():
        if name == event:
            continue
        rel = sp_to_s(trials, sr, col) - align_times
        if not np.any(np.isfinite(rel)):
            continue
        mean_rel_ms = float(np.nanmean(rel)) * 1000
        if -pre_ms <= mean_rel_ms <= post_ms:
            markers[name] = mean_rel_ms

    n        = len(trains)
    n_trials = len(align_times)
    ncols    = min(n, 4)
    nrows    = math.ceil(n / ncols)
    row_h    = max(3, 0.08 * n_trials)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, row_h * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]

        spikes_per_trial = compute_aligned_raster(train, align_times, pre_ms, post_ms)
        ax.eventplot(spikes_per_trial, colors=trial_colors,
                     linelengths=0.8, linewidths=0.5)

        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   label=f"{EVENT_STYLE[event]['label']} (align)")
        for name, t_rel_ms in markers.items():
            ax.axvline(t_rel_ms, linewidth=0.8, **EVENT_STYLE[name])

        ax.set_xlim(-pre_ms, post_ms)
        ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
        ax.set_ylabel("Trial", fontsize=7)
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    cond_str = "  |  coloured by (arm × reward)" if by_condition else ""
    fig.suptitle(
        f"Aligned raster — session {SESSION}  |  aligned to: {event}{cond_str}"
        f"  (pre={pre_ms} ms, post={post_ms} ms)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spike raster plot")
    parser.add_argument("--aligned",  action="store_true",
                        help="Show trial-by-trial aligned raster instead of full recording")
    parser.add_argument("t_start",    nargs="?",  type=float, default=None,
                        help="(full raster) Start time in seconds")
    parser.add_argument("t_end",      nargs="?",  type=float, default=None,
                        help="(full raster) End time in seconds")
    parser.add_argument("--event",    type=str,   default="cue", choices=list(EVENTS),
                        help="(aligned) Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",      type=float, default=500,
                        help="(aligned) ms before event (default: 500)")
    parser.add_argument("--post",     type=float, default=1000,
                        help="(aligned) ms after event (default: 1000)")
    parser.add_argument("--by-condition", action="store_true",
                        help="(aligned) Colour trials by (arm × reward) condition")
    parser.add_argument("--neurons",  nargs="+",  type=int,   default=None,
                        help="Neuron indices to show, e.g. --neurons 0 1 5")
    parser.add_argument("--area",     type=str,   default=None,
                        help="Show only neurons whose label contains this string")
    parser.add_argument("--list",     action="store_true",
                        help="Print all neuron indices and labels, then exit")
    add_save_arg(parser)
    args = parser.parse_args()

    if args.list:
        _, labels = get_spike_trains()
        for i, lbl in enumerate(labels):
            print(f"{i:4d}  {lbl}")
    elif args.aligned:
        fig, _ = plot_aligned_raster(
            neuron_indices=args.neurons,
            area=args.area,
            event=args.event,
            pre_ms=args.pre,
            post_ms=args.post,
            by_condition=args.by_condition,
        )
        maybe_save(fig, args, prefix="aligned_raster")
        plt.show()
    else:
        fig, _ = plot_raster(args.t_start, args.t_end,
                             neuron_indices=args.neurons, area=args.area)
        maybe_save(fig, args, prefix="raster")
        plt.show()
