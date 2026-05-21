"""Spike raster plots from STMtx.mat.

Two modes:
  plot_raster         — full-recording raster, one row per neuron.
  plot_aligned_raster — trial-by-trial raster aligned to a behavioural event,
                        one subplot per neuron, one row per trial.
"""

import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    EVENTS, EVENT_STYLE, CONDITIONS,
    get_spike_trains, load_trials_sync, load_sr,
    event_times as event_times_for, condition_masks,
    select_neurons, add_session_arg, add_selection_args,
    add_save_arg, maybe_save, handle_list, session_data_dir,
)


# ---------------------------------------------------------------------------
# Full-recording raster
# ---------------------------------------------------------------------------

def plot_raster(t_start=None, t_end=None, neuron_indices=None, area=None,
                data_dir=None, session=None):
    trains, labels = get_spike_trains(data_dir=data_dir)
    # Full-recording raster doesn't make per-neuron subplots, so the MAX_NEURONS
    # cap doesn't apply.
    trains, labels = select_neurons(trains, labels,
                                    indices=neuron_indices, area=area,
                                    enforce_cap=False)

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
    ax.set_title(f"Spike raster — session {session}  ({len(trains)} units)")

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
                        pre_ms=500, post_ms=1000, by_condition=False,
                        data_dir=None, session=None):
    """Plot a trial-by-trial raster aligned to a behavioural event."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {list(EVENTS)}")

    trains, labels = get_spike_trains(data_dir=data_dir)
    trains, labels = select_neurons(trains, labels, indices=neuron_indices, area=area)

    trials = load_trials_sync(data_dir=data_dir)
    sr     = load_sr(data_dir=data_dir)["SamplingRate_Hz"].iloc[0]

    align_times = event_times_for(trials, sr, event)
    responding  = trials["NotResponding"].to_numpy() != 1
    align_times = np.where(responding, align_times, np.nan)

    # Mean timing of other events relative to alignment point.
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

    cond = None
    if by_condition:
        cond = condition_masks(trials)

    n     = len(trains)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 2 * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]

        if by_condition:
            for name, cfg in CONDITIONS.items():
                cond_align  = align_times[cond[name]]
                cond_spikes = compute_aligned_raster(train, cond_align, pre_ms, post_ms)
                flat        = np.concatenate(cond_spikes) if cond_spikes else np.array([])
                if len(flat):
                    ax.eventplot([flat], lineoffsets=[0], colors=[cfg["color"]],
                                 linelengths=0.8, linewidths=0.5, alpha=0.1,
                                 label=cfg["label"])
        else:
            spikes_per_trial = compute_aligned_raster(train, align_times, pre_ms, post_ms)
            flat = np.concatenate(spikes_per_trial) if spikes_per_trial else np.array([])
            if len(flat):
                ax.eventplot([flat], lineoffsets=[0], colors=["black"],
                             linelengths=0.8, linewidths=0.5, alpha=0.1)

        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   label=f"{EVENTS[event]['label']} (align)")
        for name, t_rel_ms in markers.items():
            ax.axvline(t_rel_ms, linewidth=0.8,
                       label=EVENTS[name]["label"], **EVENT_STYLE[name])

        ax.set_xlim(-pre_ms, post_ms)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    cond_str = "  |  coloured by (arm x reward)" if by_condition else ""
    fig.suptitle(
        f"Aligned raster — session {session}  |  aligned to: {event}{cond_str}"
        f"  (pre={pre_ms} ms, post={post_ms} ms)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike raster plot")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--aligned",  action="store_true",
                        help="Trial-by-trial aligned raster (default: full-recording)")
    parser.add_argument("t_start",    nargs="?", type=float, default=None,
                        help="(full raster) Start time in seconds")
    parser.add_argument("t_end",      nargs="?", type=float, default=None,
                        help="(full raster) End time in seconds")
    parser.add_argument("--event",    type=str,  default="cue", choices=list(EVENTS),
                        help="(aligned) Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",      type=float, default=500,
                        help="(aligned) ms before event (default: 500)")
    parser.add_argument("--post",     type=float, default=1000,
                        help="(aligned) ms after event (default: 1000)")
    parser.add_argument("--by-condition", action="store_true",
                        help="(aligned) Colour trials by (arm, reward) condition")
    add_save_arg(parser)
    args = parser.parse_args()

    data_dir = session_data_dir(args.session)
    if handle_list(args, data_dir=data_dir):
        raise SystemExit

    if args.aligned:
        fig, _ = plot_aligned_raster(
            neuron_indices=args.neurons, area=args.area,
            event=args.event, pre_ms=args.pre, post_ms=args.post,
            by_condition=args.by_condition,
            data_dir=data_dir, session=args.session,
        )
        maybe_save(fig, args, prefix="aligned_raster")
    else:
        fig, _ = plot_raster(
            args.t_start, args.t_end,
            neuron_indices=args.neurons, area=args.area,
            data_dir=data_dir, session=args.session,
        )
        maybe_save(fig, args, prefix="raster")
    plt.show()
