"""Peristimulus time histogram (PSTH).

Aligns each neuron's spike train to a behavioural event, bins spikes across
trials, and plots firing rate in Hz. Vertical lines mark the mean timing of
other key events relative to the alignment point.

Alignment events:
  cue       — cue presentation (CuePresent_sp)
  response  — response-window start (RespWindowStart_sp)
  reward    — reward onset (RewardOnset_sp)
  start     — trial start (TrialStart_sp)

All time parameters (bin sizes, windows) are in milliseconds.
Internal spike/event times remain in seconds as stored in the data.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils import SESSION, load_trials_sync, load_sr, get_spike_trains, sp_to_s, MAX_NEURONS

# Sampling-point columns in Trials_Sync that need dividing by SR.
EVENTS = {
    "cue":      "CuePresent_sp",
    "response": "RespWindowStart_sp",
    "reward":   "RewardOnset_sp",
    "start":    "TrialStart_sp",
}

EVENT_STYLE = {
    "cue":      dict(color="royalblue", linestyle="--", label="Cue"),
    "response": dict(color="green",     linestyle=":",  label="Resp. window"),
    "reward":   dict(color="darkorange",linestyle="-.", label="Reward"),
    "start":    dict(color="gray",      linestyle="--", label="Trial start"),
}


def compute_psth(spike_times, event_times_s, pre_ms, post_ms, bin_ms):
    """Return (bin_centres_ms, firing_rate_hz) for spikes aligned to events.

    spike_times   : 1-D array of spike times in seconds
    event_times_s : 1-D array of event times in seconds, one per trial
    pre_ms        : ms before each event
    post_ms       : ms after each event
    bin_ms        : histogram bin width in ms
    """
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
    return centres_s * 1000, rate  # centres in ms


def plot_psth(neuron_indices=None, area=None, event="cue",
              pre_ms=500, post_ms=1000, bin_ms=50, sigma_ms=None):
    """Plot one PSTH subplot per neuron.

    Parameters
    ----------
    neuron_indices : list[int], optional  — restrict to these neuron indices
    area           : str, optional        — case-insensitive label substring filter
    event          : str                  — alignment event key (see EVENTS)
    pre_ms         : float                — ms before event
    post_ms        : float                — ms after event
    bin_ms         : float                — histogram bin width in ms
    sigma_ms       : float, optional      — Gaussian smoothing kernel SD in ms;
                                            None disables smoothing overlay
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

    # Mean timing of other events relative to the alignment point (in ms).
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

    n     = len(trains)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]
        centres, rate = compute_psth(train, align_times, pre_ms, post_ms, bin_ms)

        ax.bar(centres, rate, width=bin_ms, color="steelblue",
               edgecolor="none", alpha=0.6, label="_nolegend_")

        if sigma_ms is not None:
            sigma_bins = sigma_ms / bin_ms
            smoothed = gaussian_filter1d(rate, sigma=sigma_bins)
            ax.plot(centres, smoothed, color="navy", linewidth=1.2, label="_nolegend_")

        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   label=f"{EVENT_STYLE[event]['label']} (align)")

        for name, t_rel_ms in markers.items():
            style = EVENT_STYLE[name]
            ax.axvline(t_rel_ms, linewidth=0.8, **style)

        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
        ax.set_ylabel("Firing rate (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    smooth_str = f", smoothed σ={sigma_ms:.0f} ms" if sigma_ms is not None else ""
    fig.suptitle(
        f"PSTH — session {SESSION}  |  aligned to: {event}"
        f"  (pre={pre_ms}ms, post={post_ms}ms, bin={bin_ms}ms{smooth_str})",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Peristimulus time histogram (PSTH)")
    parser.add_argument("--event",   type=str,   default="cue",
                        choices=list(EVENTS),
                        help="Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",     type=float, default=500,
                        help="ms before event (default: 500)")
    parser.add_argument("--post",    type=float, default=1000,
                        help="ms after event (default: 1000)")
    parser.add_argument("--bin",     type=float, default=50,
                        help="Bin width in ms (default: 50)")
    parser.add_argument("--sigma",   type=float, default=None,
                        help="Gaussian smoothing SD in ms (default: off)")
    parser.add_argument("--neurons", nargs="+",  type=int, default=None,
                        help="Neuron indices to show, e.g. --neurons 0 1 5")
    parser.add_argument("--area",    type=str,   default=None,
                        help="Show only neurons whose label contains this string")
    parser.add_argument("--list",    action="store_true",
                        help="Print all neuron indices and labels, then exit")
    args = parser.parse_args()

    if args.list:
        _, labels = get_spike_trains()
        for i, lbl in enumerate(labels):
            print(f"{i:4d}  {lbl}")
    else:
        plot_psth(
            neuron_indices=args.neurons,
            area=args.area,
            event=args.event,
            pre_ms=args.pre,
            post_ms=args.post,
            bin_ms=args.bin,
            sigma_ms=args.sigma,
        )
        plt.show()
