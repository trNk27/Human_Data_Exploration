"""Spike raster CLI.

Two modes:
  plot_raster  — full-recording raster, one row per neuron (kept here because it
                 is *not* a per-neuron grid; the population is the figure).
  aligned      — trial-by-trial aligned raster via `explore.grid(sess, "raster")`.
                 For a single neuron:  Session(...).neuron(7).raster(...).save()

    python -m viewers.raster_plot --session 20250714 0 100        # full, 0-100 s
    python -m viewers.raster_plot --aligned --event reward --by-condition --save
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import matplotlib.pyplot as plt

from session import Session
from explore import grid
from utils import (
    EVENTS, select_neurons,
    add_session_arg, add_selection_args, add_save_arg, maybe_save,
    handle_list, session_data_dir,
)


def plot_raster(sess, t_start=None, t_end=None, neuron_indices=None, area=None):
    """Full-recording raster: one row of spike ticks per neuron."""
    trains, labels = sess.spike_trains
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
    ax.eventplot(trains, colors="black", linelengths=0.8, linewidths=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-0.5, len(trains) - 0.5)
    if t_start is not None or t_end is not None:
        ax.set_xlim(t_start, t_end)
    ax.set_title(f"Spike raster — session {sess.id}  ({len(trains)} units)")
    fig.tight_layout()
    return fig, ax


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

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess = Session(args.session)
    if args.aligned:
        panel = grid(
            sess, "raster",
            neurons=args.neurons, area=args.area,
            align=args.event, pre_ms=args.pre, post_ms=args.post,
            condition="all" if args.by_condition else None,
        )
        maybe_save(panel.fig, args, prefix="aligned_raster")
    else:
        fig, _ = plot_raster(
            sess, args.t_start, args.t_end,
            neuron_indices=args.neurons, area=args.area,
        )
        maybe_save(fig, args, prefix="raster")
    plt.show()
