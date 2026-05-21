"""Autocorrelogram for each neuron in STMtx.mat.

For each neuron, bins all spike-time differences within +/- lag_ms into a
histogram (zero-lag excluded). Subplots are arranged in a grid.
"""

import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_spike_trains, select_neurons,
    add_session_arg, add_selection_args, add_save_arg, maybe_save,
    handle_list, session_data_dir,
)


def compute_acg(spike_times, lag_ms=200, bin_ms=1):
    """Return (bin_centres_ms, counts) for the autocorrelogram.

    Discretises the spike train at bin_ms resolution, then uses FFT-based
    circular autocorrelation — O(N log N) in recording length, independent
    of spike count. Zero-lag bin is set to 0 (self-coincidences excluded).
    """
    lag_bins = int(round(lag_ms / bin_ms))
    dt       = bin_ms / 1000.0
    centres  = np.arange(-lag_bins, lag_bins + 1) * bin_ms

    spike_times = np.sort(spike_times[spike_times >= 0])
    if len(spike_times) < 2:
        return centres, np.zeros(len(centres), dtype=np.int64)

    # Discretise spikes to bin_ms resolution.
    n   = int(np.ceil(spike_times[-1] / dt)) + 2
    idx = np.clip(np.round(spike_times / dt).astype(int), 0, n - 1)
    train = np.bincount(idx, minlength=n).astype(np.float64)

    # FFT autocorrelation (pad to next power of 2 for speed)
    fft_len = int(2 ** np.ceil(np.log2(2 * n)))
    F   = np.fft.rfft(train, n=fft_len)
    acf = np.fft.irfft(F * F.conj()).real

    # Vectorised readout of +/- lag_bins lags.
    lags   = np.arange(-lag_bins, lag_bins + 1)
    lookup = np.where(lags >= 0, lags, fft_len + lags)
    counts = np.round(acf[lookup]).astype(np.int64)
    counts[lag_bins] = 0   # exclude self-coincidences

    return centres, counts


def plot_acg(neuron_indices=None, area=None, lag_ms=200, bin_ms=1,
             data_dir=None, session=None):
    trains, labels = get_spike_trains(data_dir=data_dir)
    trains, labels = select_neurons(trains, labels, indices=neuron_indices, area=area)

    n      = len(trains)
    ncols  = min(n, 4)
    nrows  = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]
        centres, counts = compute_acg(train, lag_ms=lag_ms, bin_ms=bin_ms)
        ax.bar(centres, counts, width=bin_ms, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Lag (ms)", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Autocorrelograms — session {session}  (lag +/-{lag_ms} ms, bin {bin_ms} ms)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocorrelogram")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--lag", type=float, default=200, help="Max lag in ms (default 200)")
    parser.add_argument("--bin", type=float, default=1,   help="Bin size in ms (default 1)")
    add_save_arg(parser)
    args = parser.parse_args()

    data_dir = session_data_dir(args.session)
    if handle_list(args, data_dir=data_dir):
        raise SystemExit

    fig, _ = plot_acg(
        neuron_indices=args.neurons, area=args.area,
        lag_ms=args.lag, bin_ms=args.bin,
        data_dir=data_dir, session=args.session,
    )
    maybe_save(fig, args, prefix="acg")
    plt.show()
