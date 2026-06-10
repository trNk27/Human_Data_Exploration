"""Pure computation for the Human Data analysis — no matplotlib, no plotting.

Single stable home for the numeric kernels shared by the viewers, the
single-neuron `explore` layer, and the aggregate analysis scripts:

  - compute_psth            peristimulus firing-rate histogram
  - compute_aligned_raster  per-trial spike times relative to an event
  - compute_acg             autocorrelogram (FFT-based)
  - perceived_probability   rolling gamble reward rate
  - trial_firing_rates      mean firing rate per trial, per neuron
  - binned_stats            mean ± SEM of y in equal-width bins over [0, 1]

All time *parameters* are in milliseconds (`bin_ms`, `pre_ms`, `post_ms`,
`lag_ms`); absolute spike/event times stay in seconds, as stored in the data.
"""

from __future__ import annotations

import numpy as np

WINDOWS = ("trial", "cue_to_reward", "reward_to_end")

WINDOW_LABELS = {
    "trial":         "Full trial (start → end)",
    "cue_to_reward": "Cue → Reward onset",
    "reward_to_end": "Reward onset → Trial end",
}


# ---------------------------------------------------------------------------
# PSTH
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Autocorrelogram
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Firing rate vs perceived probability
# ---------------------------------------------------------------------------

def perceived_probability(trials, responding_mask: np.ndarray,
                          history: int = 10) -> np.ndarray:
    """Rolling gamble reward rate over the last `history` gamble trials.

    perceived_probability[t] is the fraction of rewarded outcomes in the last
    `history` *gamble* (arm == 1) responding trials strictly before t.
    Only defined for gamble responding trials; all other trials get NaN.
    """
    n         = len(trials)
    rewarded  = trials["Rewarded"].to_numpy()
    arm       = trials["ChosenArm_G1S0"].to_numpy()
    perc_prob = np.full(n, np.nan)
    past: list[float] = []

    for t in range(n):
        if not responding_mask[t] or arm[t] != 1:
            continue
        if past:
            perc_prob[t] = float(np.mean(past[-history:]))
        past.append(float(rewarded[t]))

    return perc_prob


def trial_firing_rates(trains: list, trials, sr: int,
                       window: str = "trial",
                       trial_mask: np.ndarray | None = None) -> np.ndarray:
    """Mean firing rate (Hz) per trial for each neuron.

    Parameters
    ----------
    trains      : list of 1-D spike-time arrays (seconds), already neuron-selected
    trials      : Trials_Sync DataFrame
    sr          : sampling rate (Hz)
    window      : one of WINDOWS
    trial_mask  : optional boolean array (n_trials,); rates are NaN where False

    Returns
    -------
    rates : (n_trials, n_neurons) float — NaN where the window is undefined.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")

    n_trials  = len(trials)
    n_neurons = len(trains)

    if window == "trial":
        t0_all = trials["TrialStart_sp"].to_numpy()  / sr
        t1_all = trials["TrialEnd_sp"].to_numpy()    / sr
        valid  = np.ones(n_trials, dtype=bool)
    elif window == "cue_to_reward":
        t0_all = trials["CuePresent_sp"].to_numpy()  / sr
        t1_all = trials["RewardOnset_sp"].to_numpy() / sr
        valid  = np.ones(n_trials, dtype=bool)
    else:  # reward_to_end
        t0_all = trials["RewardOnset_sp"].to_numpy() / sr
        t1_all = trials["TrialEnd_sp"].to_numpy()    / sr
        valid  = np.ones(n_trials, dtype=bool)

    if trial_mask is not None:
        valid = valid & trial_mask

    rates = np.full((n_trials, n_neurons), np.nan)

    for n_idx, spikes in enumerate(trains):
        spikes_s = np.sort(spikes)
        for t in range(n_trials):
            if not valid[t]:
                continue
            t0, t1 = t0_all[t], t1_all[t]
            dur = t1 - t0
            if dur <= 0 or not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            # searchsorted is faster than boolean indexing for long spike trains
            lo = int(np.searchsorted(spikes_s, t0, side="left"))
            hi = int(np.searchsorted(spikes_s, t1, side="left"))
            rates[t, n_idx] = (hi - lo) / dur

    return rates


def binned_stats(x: np.ndarray, y: np.ndarray,
                 n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and SEM of y in n_bins equal-width bins over [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    cx, mn, se = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        vals = y[mask]
        cx.append(0.5 * (lo + hi))
        mn.append(float(np.mean(vals)))
        se.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    return np.array(cx), np.array(mn), np.array(se)
