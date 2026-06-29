"""Aligned raster plot for a single neuron — one row per trial.

    # all trials, plain black
    python -m viewers.raster_plot --session 20250714 --neuron 7

    # colour trials by condition
    python -m viewers.raster_plot --session 20250714 --neuron 7 --by-condition

    # only gamble trials, coloured
    python -m viewers.raster_plot --session 20250714 --neuron 7 --condition G+R G+N --by-condition

    # separate subplot per condition (stacked, shared x)
    python -m viewers.raster_plot --session 20250714 --neuron 7 --split

    # only gamble conditions, split
    python -m viewers.raster_plot --session 20250714 --neuron 7 --condition G+R G+N --split --save
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from session import Session
from compute import compute_aligned_raster
from utils import (
    EVENTS, CONDITIONS, EVENT_STYLE,
    add_session_arg, add_save_arg, maybe_save,
)


def _draw_panel(ax, spikes, trial_indices, color, align, markers, pre_ms, post_ms,
                title=None, show_xlabel=True):
    """Render one raster panel — each entry in trial_indices is one row."""
    for row, tidx in enumerate(trial_indices):
        sp = spikes[tidx]
        if len(sp):
            ax.eventplot([sp], lineoffsets=[row], colors=[color],
                         linelengths=0.8, linewidths=0.5)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
               label=f"{EVENTS[align]['label']} (align)")
    for name, t_ms in markers.items():
        ax.axvline(t_ms, linewidth=0.8, label=EVENTS[name]["label"],
                   **EVENT_STYLE[name])

    n = len(trial_indices)
    ax.set_xlim(-pre_ms, post_ms)
    ax.set_ylim(-0.5, max(n - 0.5, 0.5))
    ax.set_ylabel("Trial")
    if show_xlabel:
        ax.set_xlabel("Time rel. to event (ms)")
    if title:
        ax.set_title(title, fontsize=20)
    ax.legend(fontsize=20, loc="upper left")


def plot_neuron_raster(sess, neuron_idx, align="cue", pre_ms=500, post_ms=1000,
                       by_condition=False, conditions=None, split=False):
    """Aligned raster for one neuron.

    conditions : list of condition keys to include (e.g. ["G+R","G+N"]).
                 None → all responding trials.
    split      : if True, one subplot per condition (shared x-axis).
    by_condition: colour all trials in a single panel by their condition.
    """
    trains, labels = sess.spike_trains
    train  = trains[neuron_idx]
    label  = labels[neuron_idx]

    align_times = sess.event_times(align)
    spikes      = compute_aligned_raster(train, align_times, pre_ms, post_ms)
    markers     = sess.marker_times_ms(align, pre_ms, post_ms)

    cond_masks  = sess.condition_masks()
    active_conds = list(conditions) if conditions else list(CONDITIONS)

    # ---------- split view: one subplot per condition ----------
    if split:
        n = len(active_conds)
        row_counts = [int(cond_masks[c].sum()) for c in active_conds]
        heights    = [max(1, rc) for rc in row_counts]
        fig, axes  = plt.subplots(
            n, 1,
            figsize=(10, min(sum(h * 0.12 for h in heights) + 1.2 * n, 20)),
            gridspec_kw={"height_ratios": heights},
            sharex=True,
        )
        if n == 1:
            axes = [axes]

        for ax, cond_key in zip(axes, active_conds):
            cfg     = CONDITIONS[cond_key]
            tidxs   = np.where(cond_masks[cond_key])[0]
            is_last = (cond_key == active_conds[-1])
            _draw_panel(ax, spikes, tidxs, cfg["color"], align, markers,
                        pre_ms, post_ms,
                        title=cfg["label"],
                        show_xlabel=is_last)

        fig.suptitle(
            f"Raster — {label}  (session {sess.id},  align: {EVENTS[align]['label']})",
            fontsize=0,
        )
        fig.tight_layout()
        return fig, axes

    # ---------- single-panel view ----------
    if conditions:
        # filter to selected conditions only
        keep = np.zeros(len(align_times), dtype=bool)
        for c in active_conds:
            keep |= cond_masks[c]
        resp_indices = np.where(keep & np.isfinite(align_times))[0]
    else:
        resp_indices = np.where(np.isfinite(align_times))[0]

    if by_condition:
        trial_color: dict[int, str] = {}
        for cond_name in active_conds:
            for idx in np.where(cond_masks[cond_name])[0]:
                trial_color[idx] = CONDITIONS[cond_name]["color"]

    n_rows = len(resp_indices)
    fig, ax = plt.subplots(figsize=(10, min(max(3, n_rows * 0.12), 20)))

    for row, tidx in enumerate(resp_indices):
        sp    = spikes[tidx]
        color = (trial_color.get(tidx, "gray") if by_condition else "black")
        if len(sp):
            ax.eventplot([sp], lineoffsets=[row], colors=[color],
                         linelengths=0.8, linewidths=0.5)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
               label=f"{EVENTS[align]['label']} (align)")
    for name, t_ms in markers.items():
        ax.axvline(t_ms, linewidth=0.8, label=EVENTS[name]["label"],
                   **EVENT_STYLE[name])

    if by_condition:
        patches = [mpatches.Patch(color=CONDITIONS[c]["color"],
                                  label=CONDITIONS[c]["label"])
                   for c in active_conds]
        ax.legend(handles=patches, fontsize=15, loc="upper right")
    else:
        ax.legend(fontsize=15, loc="upper right")

    ax.set_xlim(-pre_ms, post_ms)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlabel("Time rel. to event (ms)", fontsize = 15)
    ax.set_ylabel("Trial", fontsize = 20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)




    ax.set_title(
        f"Raster — {label}  (session {sess.id},  align: {EVENTS[align]['label']})"
    )
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aligned raster — one row per trial")
    add_session_arg(parser)
    parser.add_argument("--neuron", type=int, required=True,
                        help="Neuron index (integer column id in STMtx.mat)")
    parser.add_argument("--event", type=str, default="cue", choices=list(EVENTS),
                        help="Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",  type=float, default=500,
                        help="ms before event (default: 500)")
    parser.add_argument("--post", type=float, default=1000,
                        help="ms after event (default: 1000)")
    parser.add_argument("--condition", nargs="+", choices=list(CONDITIONS),
                        metavar="COND", default=None,
                        help="Filter to one or more conditions: G+R  G+N  S+R "
                             "(default: all responding trials)")
    parser.add_argument("--by-condition", action="store_true",
                        help="Colour trials by outcome condition in a single panel")
    parser.add_argument("--split", action="store_true",
                        help="Separate subplot per condition (stacked, shared x-axis)")
    add_save_arg(parser)
    args = parser.parse_args()

    sess = Session(args.session)
    result = plot_neuron_raster(
        sess, args.neuron,
        align=args.event,
        pre_ms=args.pre,
        post_ms=args.post,
        by_condition=args.by_condition,
        conditions=args.condition,
        split=args.split,
    )
    fig = result[0]
    maybe_save(fig, args, prefix=f"raster_n{args.neuron}")
    plt.show()
