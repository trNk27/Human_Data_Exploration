"""Peristimulus time histogram (PSTH).

Aligns each neuron's spike train to a behavioural event, bins spikes across
trials, and plots firing rate in Hz. Vertical lines mark the mean timing of
other key events relative to the alignment point.

Alignment events:
  cue       — cue presentation (CuePresent_sp)
  response  — response-window start (RespWindowStart_sp)
  reward    — reward onset (RewardOnset_sp)
  start     — trial start (TrialStart_sp)
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

def compute_psth(spike_times, event_times_s, pre_s, post_s, bin_s):
    """Return (bin_centres_s, firing_rate_hz) for spikes aligned to events.

    spike_times   : 1-D array of spike times in seconds
    event_times_s : 1-D array of event times in seconds, one per trial
    """
    edges = np.arange(-pre_s, post_s + bin_s / 2, bin_s)
    counts = np.zeros(len(edges) - 1, dtype=np.float64)
    n_valid = 0

    for t_ev in event_times_s:
        if not np.isfinite(t_ev):
            continue
        aligned = spike_times - t_ev
        in_win = aligned[(aligned >= -pre_s) & (aligned < post_s)]
        counts += np.histogram(in_win, bins=edges)[0]
        n_valid += 1

    centres = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / (n_valid * bin_s) if n_valid > 0 else counts
    return centres, rate


def plot_psth(neuron_indices=None, area=None, event="cue",
              pre_s=0.5, post_s=1.0, bin_s=0.05, sigma_ms=None):
    """Plot one PSTH subplot per neuron.

    Parameters
    ----------
    neuron_indices : list[int], optional  — restrict to these neuron indices
    area           : str, optional        — case-insensitive label substring filter
    event          : str                  — alignment event key (see EVENTS)
    pre_s          : float                — seconds before event
    post_s         : float                — seconds after event
    bin_s          : float                — histogram bin width in seconds
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

    # Mean timing of other events relative to the alignment point.
    markers = {}
    for name, col in EVENTS.items():
        if name == event:
            continue
        rel = sp_to_s(trials, sr, col) - align_times
        if not np.any(np.isfinite(rel)):
            continue
        mean_rel = float(np.nanmean(rel))
        if -pre_s <= mean_rel <= post_s:
            markers[name] = mean_rel

    n     = len(trains)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, (train, label) in enumerate(zip(trains, labels)):
        ax = axes[idx // ncols][idx % ncols]
        centres, rate = compute_psth(train, align_times, pre_s, post_s, bin_s)

        ax.bar(centres, rate, width=bin_s, color="steelblue",
               edgecolor="none", alpha=0.6, label="_nolegend_")

        if sigma_ms is not None:
            sigma_bins = (sigma_ms / 1000.0) / bin_s
            smoothed = gaussian_filter1d(rate, sigma=sigma_bins)
            ax.plot(centres, smoothed, color="navy", linewidth=1.2, label="_nolegend_")

        # t = 0: alignment event
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   label=f"{EVENT_STYLE[event]['label']} (align)")

        for name, t_rel in markers.items():
            style = EVENT_STYLE[name]
            ax.axvline(t_rel, linewidth=0.8, **style)

        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Time rel. to event (s)", fontsize=7)
        ax.set_ylabel("Firing rate (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    smooth_str = f", smoothed σ={sigma_ms:.0f} ms" if sigma_ms is not None else ""
    fig.suptitle(
        f"PSTH — session {SESSION}  |  aligned to: {event}"
        f"  (pre={pre_s}s, post={post_s}s, bin={bin_s*1000:.0f}ms{smooth_str})",
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
    parser.add_argument("--pre",     type=float, default=0.5,
                        help="Seconds before event (default: 0.5)")
    parser.add_argument("--post",    type=float, default=1.0,
                        help="Seconds after event (default: 1.0)")
    parser.add_argument("--bin",     type=float, default=0.05,
                        help="Bin width in seconds (default: 0.05)")
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
            pre_s=args.pre,
            post_s=args.post,
            bin_s=args.bin,
            sigma_ms=args.sigma,
        )
        plt.show()
