"""Simple raster plot of spike times from STMtx.mat.

Each column of STMtx is one neuron; values are spike times in seconds,
bottom-padded with NaN. We strip the NaNs per column and draw one row
per neuron with matplotlib's eventplot.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import SESSION, get_spike_trains, add_save_arg, maybe_save


def plot_raster(t_start=None, t_end=None, neuron_indices=None, area=None):
    trains, labels = get_spike_trains()

    # Filter by explicit neuron indices.
    if neuron_indices is not None:
        trains = [trains[i] for i in neuron_indices]
        labels = [labels[i] for i in neuron_indices]

    # Filter by area substring (case-insensitive match against label).
    if area is not None:
        mask = [area.lower() in lbl.lower() for lbl in labels]
        trains = [t for t, m in zip(trains, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    if not trains:
        raise ValueError("No neurons match the given selection.")

    # Optional time window — keep only spikes inside [t_start, t_end].
    if t_start is not None or t_end is not None:
        lo = -np.inf if t_start is None else t_start
        hi =  np.inf if t_end   is None else t_end
        trains = [s[(s >= lo) & (s <= hi)] for s in trains]

    fig, ax = plt.subplots(figsize=(12, max(3, 0.25 * len(trains))))


    # +++ Fast method (but a little bad)

    # Flatten (times, neuron_idx) and draw with a single ax.plot using "|" markers.
    
    # eventplot creates one LineCollection segment per spike — at ~10M+ spikes
    # per session that takes ~minute. One Line2D with N points is orders faster.
    #times   = np.concatenate(trains) if trains else np.array([])
    #neurons = np.repeat(np.arange(len(trains)), [len(t) for t in trains])
    #ax.plot(times, neurons, "|", color="black", markersize=3, markeredgewidth=0.5)


    # +++ Slow method (but a little more accurate)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spike raster plot")
    parser.add_argument("t_start",   nargs="?",  type=float, default=None, help="Start time (s)")
    parser.add_argument("t_end",     nargs="?",  type=float, default=None, help="End time (s)")
    parser.add_argument("--neurons", nargs="+",  type=int,   default=None, help="Neuron indices to show, e.g. --neurons 0 1 5")
    parser.add_argument("--area",                type=str,   default=None, help="Show only neurons whose label contains this string, e.g. --area MFG")
    parser.add_argument("--list",    action="store_true",                  help="Print all neuron indices and labels, then exit")
    add_save_arg(parser)
    args = parser.parse_args()

    if args.list:
        _, labels = get_spike_trains()
        for i, lbl in enumerate(labels):
            print(f"{i:4d}  {lbl}")
    else:
        fig, _ = plot_raster(args.t_start, args.t_end, neuron_indices=args.neurons, area=args.area)
        maybe_save(fig, args, prefix="raster")
        plt.show()
