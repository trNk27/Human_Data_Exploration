"""firing_rate_vs_perc_p.py — Neuron firing rate vs perceived reward probability.

Perceived probability at trial t is the fraction of rewards in the last
`history` responding trials *before* t (rolling window — what the subject
could plausibly have learned from).

Three time windows for the firing rate:
  trial          : TrialStart → TrialEnd
  cue_to_reward  : CuePresent → RewardOnset   (rewarded trials only)
  reward_to_end  : RewardOnset → TrialEnd      (rewarded trials only)

One subplot per neuron: per-trial scatter, binned mean ± SEM, linear regression.

Usage
-----
    python firing_rate_vs_perc_p.py
    python firing_rate_vs_perc_p.py --window cue_to_reward --history 10 --bins 8
    python firing_rate_vs_perc_p.py --neurons 0 1 5 --area ACC --save
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from session import Session
from utils import (
    select_neurons,
    add_session_arg, add_selection_args, add_save_arg, maybe_save,
    handle_list, session_data_dir,
)

WINDOWS = ("trial", "cue_to_reward", "reward_to_end")

WINDOW_LABELS = {
    "trial":         "Full trial (start → end)",
    "cue_to_reward": "Cue → Reward onset  (rewarded trials only)",
    "reward_to_end": "Reward onset → Trial end  (rewarded trials only)",
}


# ---------------------------------------------------------------------------
# Core computations
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
        valid  = trials["Rewarded"].to_numpy() == 1
    else:  # reward_to_end
        t0_all = trials["RewardOnset_sp"].to_numpy() / sr
        t1_all = trials["TrialEnd_sp"].to_numpy()    / sr
        valid  = trials["Rewarded"].to_numpy() == 1

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


def _binned_stats(x: np.ndarray, y: np.ndarray,
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_fr_vs_perc_p(
    sess: Session,
    neuron_indices=None,
    area: str | None = None,
    window: str = "trial",
    history: int = 10,
    n_bins: int = 8,
):
    """Return (fig, axes) — one subplot per neuron.

    Parameters
    ----------
    sess            : Session
    neuron_indices  : list[int] | None
    area            : case-insensitive label substring filter
    window          : time window key (one of WINDOWS)
    history         : past-trial window for perceived probability
    n_bins          : number of probability bins for mean ± SEM overlay
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")

    trains, labels = sess.spike_trains
    trains, labels = select_neurons(trains, labels,
                                    indices=neuron_indices, area=area)

    gamble_mask = (
        sess.responding_mask
        & (sess.trials["ChosenArm_G1S0"].to_numpy() == 1)
    )
    perc_prob = perceived_probability(sess.trials, sess.responding_mask,
                                      history=history)
    rates     = trial_firing_rates(trains, sess.trials, sess.sampling_rate,
                                   window=window, trial_mask=gamble_mask)

    n     = len(trains)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for idx in range(n):
        ax  = axes[idx // ncols][idx % ncols]
        x   = perc_prob
        y   = rates[:, idx]
        ok  = np.isfinite(x) & np.isfinite(y)

        if ok.sum() >= 4:
            xv, yv = x[ok], y[ok]

            # Scatter — all valid trials
            ax.scatter(xv, yv, s=9, alpha=0.25, color="steelblue",
                       linewidths=0, rasterized=True, zorder=2)

            # Binned mean ± SEM
            cx, mn, se = _binned_stats(xv, yv, n_bins)
            if cx.size > 0:
                ax.errorbar(cx, mn, yerr=se, fmt="o-", color="navy",
                            markersize=4, linewidth=1.4, capsize=3,
                            label="Bin mean ± SEM", zorder=5)

            # Linear regression
            slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
            x_line = np.array([0.0, 1.0])
            p_str  = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
            ax.plot(x_line, intercept + slope * x_line,
                    color="firebrick", linewidth=1.2, linestyle="--", zorder=4,
                    label=f"r = {r_val:.2f},  p = {p_str}")
            ax.legend(fontsize=5.5, loc="upper right", framealpha=0.8)
        else:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")

        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Perceived P(reward)", fontsize=7)
        ax.set_ylabel("Firing rate (Hz)",    fontsize=7)
        ax.set_title(labels[idx], fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Firing rate vs perceived P(reward)  [gamble trials only] — session {sess.id}\n"
        f"Window: {WINDOW_LABELS[window]}   |   "
        f"History: last {history} gamble trials   |   {n_bins} bins",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument(
        "--window", default="trial", choices=list(WINDOWS),
        help="Time window for firing rate (default: trial)",
    )
    parser.add_argument(
        "--history", type=int, default=10, metavar="N",
        help="Past-trial window for perceived probability (default: 10)",
    )
    parser.add_argument(
        "--bins", type=int, default=8, metavar="N",
        help="Number of probability bins for mean overlay (default: 8)",
    )
    add_save_arg(parser)
    args = parser.parse_args()

    data_dir = session_data_dir(args.session)
    if handle_list(args, data_dir=data_dir):
        raise SystemExit

    sess = Session(args.session)
    fig, _ = plot_fr_vs_perc_p(
        sess,
        neuron_indices=args.neurons,
        area=args.area,
        window=args.window,
        history=args.history,
        n_bins=args.bins,
    )
    maybe_save(fig, args, prefix="fr_vs_perc_p")
    plt.show()
