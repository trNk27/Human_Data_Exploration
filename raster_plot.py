"""Simple raster plot of spike times from STMtx.mat.

Each column of STMtx is one neuron; values are spike times in seconds,
bottom-padded with NaN. We strip the NaNs per column and draw one row
per neuron with matplotlib's eventplot.
"""

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from file_explorer import SESSION, DATA_DIR, load_stmtx, load_sr


def get_spike_trains():
    """Return a list of 1-D arrays (spike times in seconds), one per neuron."""
    df = load_stmtx()                       # (samples, units), NaN-padded
    trains = [df[col].dropna().to_numpy() for col in df.columns]
    return trains, list(df.columns)


def plot_raster(t_start=None, t_end=None):
    trains, labels = get_spike_trains()

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
    parser.add_argument("t_start", nargs="?", type=float, default=None, help="Start time (s)")
    parser.add_argument("t_end",   nargs="?", type=float, default=None, help="End time (s)")
    args = parser.parse_args()
    plot_raster(args.t_start, args.t_end)
    plt.show()
