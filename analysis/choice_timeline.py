"""choice_timeline.py — Participant decisions + Rescorla-Wagner model comparison.

Two modes share one RW + softmax engine (see run_rw_model):
  shadow   — the model *observes* the participant: it predicts each choice but
             learns from the participant's real choice & reward. Match rate is
             genuine prediction accuracy. (default; used by fit_rw.py)
  simulate — the model *free-runs*: it samples its own choices from P(Gamble)
             and its own rewards from the trial's reward probabilities. It
             drifts from the participant, so match rate is a behavioural
             comparison of two independent agents, not a fit metric.

Layout (top → bottom)
---------------------
  shadow mode
    [Participant]
      Gamble row : orange ticks — bright = rewarded, muted = unrewarded
      Safe   row : green  ticks — bright = rewarded, muted = unrewarded
    [RW model — prediction row]
      Correct : black ticks  (model prediction == participant choice)
      Wrong   : gray  ticks
    [P(Gamble)]
      The model's trial-by-trial P(Gamble) trace (orange fill above .5, green below)
    [Alignment]
      Rolling match-rate line (model prediction == participant choice)
      Fill: blue above chance, red below chance

  simulate mode
    [Participant]
      Gamble row / Safe row  (same colour scheme)
    [RW Model  (α, β, φ)]
      Gamble row / Safe row  drawn from the model's own choices & sampled rewards
      P(Gamble) overlaid on model Gamble row
    [Alignment]
      Rolling match-rate line

Usage
-----
    python choice_timeline.py
    python choice_timeline.py --session 20250602
    python choice_timeline.py --alpha 0.15 --beta 4 --phi 1.0 --window 20
    python choice_timeline.py --mode simulate --phi 1.0 --seed 42
    python choice_timeline.py --save
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# Make the repo root importable when run as `python analysis/<file>.py`.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session import Session
from utils import CONDITIONS, add_session_arg, add_save_arg, maybe_save

# ---------------------------------------------------------------------------
# Global font size — change this one number to scale all text in the figure.
# Each fontsize below is BASE_FONT + a fixed offset that preserves the original
# visual hierarchy (original sizes: 6, 6.5, 7, 7.5, 8, 9, 10, 11).
# ---------------------------------------------------------------------------
BASE_FONT = 10


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def desaturate(color, factor: float = 0.25) -> tuple:
    """Blend *color* toward its own perceived grey.

    factor = 0 → grey,  factor = 1 → original colour.
    """
    r, g, b, *a = mcolors.to_rgba(color)
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return (
        grey + factor * (r - grey),
        grey + factor * (g - grey),
        grey + factor * (b - grey),
        *(a if a else (1.0,)),
    )


# ---------------------------------------------------------------------------
# Rescorla-Wagner + softmax model
# ---------------------------------------------------------------------------

def run_rw_model(
    trials,
    alpha: float = 0.1,
    beta: float = 5.0,
    phi: float = 0.0,
    mode: str = "shadow",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trial-by-trial Rescorla-Wagner model with softmax + perseveration.

    One RW + softmax engine, two modes that differ only in where each trial's
    choice and reward come from:

    mode="shadow"  (default) — *observer*. The model watches the participant:
        its predicted choice is argmax P(Gamble), but its Q-values are always
        updated from the *participant's* actual choice and reward. Match rate
        against the participant is genuine prediction accuracy, and this is the
        mode used for likelihood fitting (fit_rw.py).

    mode="simulate" — *generative*. The model makes its own choices by sampling
        P(Gamble), samples its own reward from the trial's reward probabilities,
        and updates Q from its own choice and reward. It drifts away from the
        participant's trajectory, so the match rate is a behavioural comparison
        of two independent agents — NOT a goodness-of-fit measure.

    The perseveration term C_t uses the previous choice of whichever agent
    drives the updates: the participant (shadow) or the model itself (simulate).

        P(Gamble) = σ( β·(Q_G − Q_S) + φ·C_t )
        C_t = +1 if that previous choice was Gamble, -1 if Safe, 0 on trial 1.
    φ > 0 → sticky (repeat last choice); φ < 0 → alternating.

    Parameters
    ----------
    trials : pd.DataFrame  (Trials_Sync columns)
    alpha  : learning rate  ∈ (0, 1)
    beta   : inverse temperature (higher → more deterministic)
    phi    : perseveration strength (0 = no perseveration bias)
    mode   : "shadow" (observe participant) or "simulate" (free-run)
    rng    : np.random.Generator for "simulate" (created if None)

    Returns
    -------
    model_choice  : (n_trials,) float  1=Gamble, 0=Safe, NaN=non-responding
    p_gamble      : (n_trials,) float  P(Gamble) at each trial,   NaN=non-responding
    q_gamble      : (n_trials,) float  Q(Gamble) before update,   NaN=non-responding
    q_safe        : (n_trials,) float  Q(Safe)   before update,   NaN=non-responding
    model_rewards : (n_trials,) float  reward on the model row — the participant's
                    reward (shadow) or the model's sampled reward (simulate)
    """
    if mode not in ("shadow", "simulate"):
        raise ValueError(f"mode must be 'shadow' or 'simulate', got {mode!r}")
    if mode == "simulate" and rng is None:
        rng = np.random.default_rng()

    n          = len(trials)
    arm        = trials["ChosenArm_G1S0"].to_numpy()       # participant's choice
    rew        = trials["Rewarded"].to_numpy()
    responding = (trials["NotResponding"] == 0).to_numpy()
    if mode == "simulate":
        p_big   = trials["P_BigReward_Gamble"].to_numpy()  # reward prob if Gamble
        p_small = trials["P_SmallReward_Safe"].to_numpy()  # reward prob if Safe

    q = {1: 0.5, 0: 0.5}   # Q-values: gamble=1, safe=0
    last_choice = None       # previous responding choice of the driving agent

    model_choice  = np.full(n, np.nan)
    p_gamble      = np.full(n, np.nan)
    q_gamble      = np.full(n, np.nan)
    q_safe        = np.full(n, np.nan)
    model_rewards = np.full(n, np.nan)

    for t in range(n):
        if not responding[t]:
            continue

        # Snapshot current Q values (state *before* this trial)
        q_gamble[t] = q[1]
        q_safe[t]   = q[0]

        # Perseveration signal: +1 = last was Gamble, -1 = last was Safe, 0 = first trial
        if last_choice is None:
            c_t = 0.0
        else:
            c_t = 1.0 if last_choice == 1 else -1.0

        # Softmax: P(Gamble) = σ( β·(Q_G − Q_S) + φ·C_t )
        p_g = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0]) - phi * c_t))
        p_gamble[t] = p_g

        if mode == "shadow":
            # Predict, but learn from the participant's real choice & reward.
            model_choice[t] = 1.0 if p_g >= 0.5 else 0.0
            chosen          = int(arm[t])
            reward          = float(rew[t])
        else:  # simulate — make and learn from the model's own choice & reward
            chosen          = int(rng.random() < p_g)
            reward_prob     = p_big[t] if chosen == 1 else p_small[t]
            reward          = float(rng.random() < reward_prob)
            model_choice[t] = float(chosen)

        model_rewards[t] = reward
        q[chosen] += alpha * (reward - q[chosen])
        last_choice = chosen

    return model_choice, p_gamble, q_gamble, q_safe, model_rewards


# ---------------------------------------------------------------------------
# Rolling alignment
# ---------------------------------------------------------------------------

def rolling_match(
    participant_arm: np.ndarray,
    model_choice: np.ndarray,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial and rolling-window match rate (model == participant).

    Only responding (non-NaN model_choice) trials contribute.

    Returns
    -------
    roll_rate : (n,) float  rolling window match fraction, NaN elsewhere
    match     : (n,) float  per-trial 1/0 match, NaN for non-responding
    """
    n     = len(participant_arm)
    match = np.full(n, np.nan)
    rate  = np.full(n, np.nan)

    valid = ~np.isnan(model_choice)
    match[valid] = (participant_arm[valid] == model_choice[valid]).astype(float)

    valid_idx  = np.where(valid)[0]
    match_vals = match[valid_idx]

    for k, t in enumerate(valid_idx):
        lo       = max(0, k - window + 1)
        rate[t]  = np.mean(match_vals[lo : k + 1])

    return rate, match


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_choice_timeline(
    sess: Session,
    alpha: float = 0.1,
    beta: float = 5.0,
    phi: float = 0.0,
    window: int = 20,
    mode: str = "shadow",
    rng: np.random.Generator | None = None,
    n_trials: int | None = None,
):
    """Build and return the figure.

    shadow mode   — 4 rows: [RW Prediction] [Participant Gamble] [Participant Safe] [Alignment]
      The prediction row shows per-trial correct (coloured by arm) / wrong (gray) ticks
      with P(Gamble) overlaid.  The model is an observer here; duplicating its predicted
      arm rows would confuse reward colouring (model_rewards is the participant's reward).

    simulate mode — 5 rows: [Participant Gamble] [Participant Safe]
                             [Model Gamble]       [Model Safe]       [Alignment]
      The model free-runs with its own choices and rewards, so arm-by-arm rows make sense.
    """
    trials     = sess.trials
    responding = sess.responding_mask
    if n_trials is not None:
        trials     = trials.iloc[:n_trials]
        responding = responding[:n_trials]
    n_trials   = len(trials)
    arm        = trials["ChosenArm_G1S0"].to_numpy()
    rew        = trials["Rewarded"].to_numpy()
    idx        = np.arange(n_trials)
    is_sim     = (mode == "simulate")

    # ---- Run model ----------------------------------------------------
    model_choice, p_gamble, _, _, model_rewards = run_rw_model(
        trials, alpha=alpha, beta=beta, phi=phi, mode=mode, rng=rng,
    )
    roll_rate, match_arr = rolling_match(arm, model_choice, window=window)
    overall_acc = float(np.nanmean(match_arr))

    # ---- Colours ------------------------------------------------------
    col_gr    = CONDITIONS["G+R"]["color"]       # darkorange
    col_gn    = desaturate(col_gr, factor=0.28)  # muted orange
    col_sr    = CONDITIONS["S+R"]["color"]       # seagreen
    col_sn    = desaturate(col_sr, factor=0.28)  # muted green
    col_wrong = "#c0c0c0"                        # gray — wrong predictions (shadow only)

    # ---- Participant masks --------------------------------------------
    p_gr = responding & (arm == 1) & (rew == 1)
    p_gn = responding & (arm == 1) & (rew == 0)
    p_sr = responding & (arm == 0) & (rew == 1)
    p_sn = responding & (arm == 0) & (rew == 0)

    # ---- Tick-row drawing helper -------------------------------------
    def tick_row(ax, rows, ylabel):
        for trial_indices, color, label in rows:
            if trial_indices.size == 0:
                continue
            ax.eventplot(
                trial_indices,
                orientation="horizontal",
                lineoffsets=0.5, linelengths=0.75, linewidths=1.4,
                colors=color, label=label,
            )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(ylabel, fontsize=BASE_FONT + 3, labelpad=6,
                      rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=BASE_FONT + 0.5, framealpha=0.8,
                  handlelength=1.0, ncol=2, borderpad=0.4)

    # ---- P(Gamble) overlay helper ------------------------------------
    def overlay_pgamble(ax):
        ax_r = ax.twinx()
        ax_r.plot(idx, p_gamble, color="black", lw=0.9, alpha=0.35, label="P(Gamble)")
        ax_r.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.2)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0, 0.5, 1])
        ax_r.set_yticklabels(["0", ".5", "1"], fontsize=BASE_FONT)
        ax_r.set_ylabel("P(G)", fontsize=BASE_FONT + 1, labelpad=2)
        ax_r.spines[["top", "left"]].set_visible(False)

    # ---- Dedicated P(Gamble) row -------------------------------------
    def pgamble_row(ax, ylabel="P(Gamble)"):
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.plot(idx, p_gamble, color="black", lw=1.3, zorder=3)
        ax.fill_between(idx, 0.5, p_gamble, where=(p_gamble >= 0.5),
                        interpolate=True, color=col_gr, alpha=0.18, zorder=1)
        ax.fill_between(idx, p_gamble, 0.5, where=(p_gamble < 0.5),
                        interpolate=True, color=col_sr, alpha=0.18, zorder=1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=BASE_FONT)
        ax.set_ylabel(ylabel, fontsize=BASE_FONT + 3, labelpad=6,
                      rotation=0, ha="right", va="center")
        ax.spines[["top", "right"]].set_visible(False)

    # ---- Figure layout ------------------------------------------------
    fig = plt.figure(figsize=(15, 8))

    if is_sim:
        # 5-row layout
        gs    = gridspec.GridSpec(5, 1, figure=fig,
                                  height_ratios=[1, 1, 1, 1, 2.5], hspace=0.08)
        ax_pg = fig.add_subplot(gs[0])
        ax_ps = fig.add_subplot(gs[1], sharex=ax_pg)
        ax_mg = fig.add_subplot(gs[2], sharex=ax_pg)
        ax_ms = fig.add_subplot(gs[3], sharex=ax_pg)
        ax_al = fig.add_subplot(gs[4], sharex=ax_pg)
        top_ax = ax_pg

        # Backgrounds
        for ax in (ax_pg, ax_ps):
            ax.set_facecolor("#fafafa")
        for ax in (ax_mg, ax_ms):
            ax.set_facecolor("#f0f4ff")

        # Participant rows
        tick_row(ax_pg, [
            (idx[p_gr], col_gr, f"Rewarded  (n={p_gr.sum()})"),
            (idx[p_gn], col_gn, f"Unrewarded (n={p_gn.sum()})"),
        ], ylabel="Gamble")
        p_safe_rows = [(idx[p_sr], col_sr, f"Rewarded  (n={p_sr.sum()})")]
        if p_sn.sum():
            p_safe_rows.append((idx[p_sn], col_sn, f"Unrewarded (n={p_sn.sum()})"))
        tick_row(ax_ps, p_safe_rows, ylabel="Safe")

        # Model rows (simulate: model_rewards is model's own sampled reward)
        model_resp = ~np.isnan(model_choice)
        m_gr = model_resp & (model_choice == 1) & (model_rewards == 1)
        m_gn = model_resp & (model_choice == 1) & (model_rewards == 0)
        m_sr = model_resp & (model_choice == 0) & (model_rewards == 1)
        m_sn = model_resp & (model_choice == 0) & (model_rewards == 0)
        tick_row(ax_mg, [
            (idx[m_gr], col_gr, f"Chose G, rewarded  (n={m_gr.sum()})"),
            (idx[m_gn], col_gn, f"Chose G, unrewarded (n={m_gn.sum()})"),
        ], ylabel="Gamble")
        m_safe_rows = [(idx[m_sr], col_sr, f"Chose S, rewarded  (n={m_sr.sum()})")]
        if m_sn.sum():
            m_safe_rows.append((idx[m_sn], col_sn, f"Chose S, unrewarded (n={m_sn.sum()})"))
        tick_row(ax_ms, m_safe_rows, ylabel="Safe")

        overlay_pgamble(ax_mg)

        # Section labels
        ax_pg.text(-0.05, 0, "Participant", transform=ax_pg.transAxes,
                   fontsize=BASE_FONT + 2, color="dimgray", fontweight="bold",
                   ha="right", va="bottom", linespacing=1.4)
        ax_mg.text(-0.05, 0, "RW model",   transform=ax_mg.transAxes,
                   fontsize=BASE_FONT + 2, color="dimgray", fontweight="bold",
                   ha="right", va="bottom", linespacing=1.4)

        tick_axes = (ax_pg, ax_ps, ax_mg, ax_ms)

    else:
        # 5-row layout — shadow: participant first, then the model observer below.
        #   participant Gamble / Safe → prediction (black=correct, gray=wrong) → P(Gamble)
        gs    = gridspec.GridSpec(5, 1, figure=fig,
                                  height_ratios=[1, 1, 1, 1, 2.5], hspace=0.08)
        ax_pg = fig.add_subplot(gs[0])              # Participant – Gamble
        ax_ps = fig.add_subplot(gs[1], sharex=ax_pg) # Participant – Safe
        ax_pr = fig.add_subplot(gs[2], sharex=ax_pg) # Prediction (correct/wrong)
        ax_pp = fig.add_subplot(gs[3], sharex=ax_pg) # P(Gamble) trace
        ax_al = fig.add_subplot(gs[4], sharex=ax_pg) # Alignment
        top_ax = ax_pg

        # Backgrounds
        for ax in (ax_pg, ax_ps):
            ax.set_facecolor("#fafafa")
        for ax in (ax_pr, ax_pp):
            ax.set_facecolor("#f0f4ff")

        # Participant rows
        tick_row(ax_pg, [
            (idx[p_gr], col_gr, f"Rewarded  (n={p_gr.sum()})"),
            (idx[p_gn], col_gn, f"Unrewarded (n={p_gn.sum()})"),
        ], ylabel="Gamble")
        p_safe_rows = [(idx[p_sr], col_sr, f"Rewarded  (n={p_sr.sum()})")]
        if p_sn.sum():
            p_safe_rows.append((idx[p_sn], col_sn, f"Unrewarded (n={p_sn.sum()})"))
        tick_row(ax_ps, p_safe_rows, ylabel="Safe")

        # Prediction row: black = correct, gray = wrong
        model_resp  = ~np.isnan(model_choice)
        pred_correct = model_resp & (model_choice == arm)
        pred_wrong   = model_resp & (model_choice != arm)
        tick_row(ax_pr, [
            (idx[pred_correct], "black",     f"Correct (n={pred_correct.sum()})"),
            (idx[pred_wrong],   col_wrong,   f"Wrong   (n={pred_wrong.sum()})"),
        ], ylabel="Prediction")

        # Dedicated P(Gamble) row
        pgamble_row(ax_pp, ylabel="P(Gamble)")

        # Section labels
        ax_pg.text(-0.05, 0, "Participant", transform=ax_pg.transAxes,
                   fontsize=BASE_FONT + 2, color="dimgray", fontweight="bold",
                   ha="right", va="bottom", linespacing=1.4)
        ax_pr.text(-0.05, 0, "RW model",   transform=ax_pr.transAxes,
                   fontsize=BASE_FONT + 2, color="dimgray", fontweight="bold",
                   ha="right", va="bottom", linespacing=1.4)

        tick_axes = (ax_pg, ax_ps, ax_pr, ax_pp)

    # ---- Title --------------------------------------------------------
    top_ax.set_title(
        f"Decision sequence ({mode}) — session {sess.id}",
        fontsize=BASE_FONT + 5, pad=8,
    )

    # ---- Alignment row -----------------------------------------------
    ax_al.axhline(0.5, color="gray", lw=0.9, ls="--", alpha=0.6,
                  label="Chance (50 %)")
    ax_al.plot(idx, roll_rate, color="steelblue", lw=1.6, zorder=3,
               label=f"Rolling {window}-trial match")
    acc_word = "match" if is_sim else "accuracy"
    ax_al.axhline(overall_acc, color="steelblue", lw=1.1, ls=":",
                  alpha=0.85, label=f"Overall {acc_word}: {overall_acc:.1%}")
    ax_al.fill_between(idx, 0.5, roll_rate,
                       where=(roll_rate >= 0.5), interpolate=True,
                       color="steelblue", alpha=0.15, zorder=1)
    ax_al.fill_between(idx, roll_rate, 0.5,
                       where=(roll_rate < 0.5), interpolate=True,
                       color="firebrick", alpha=0.15, zorder=1)
    ax_al.set_ylim(0, 1.05)
    ax_al.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_al.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=BASE_FONT + 2)
    ax_al.set_ylabel("Match\nrate", fontsize=BASE_FONT + 3, labelpad=6,
                     rotation=0, ha="right", va="center")
    ax_al.set_xlabel("Trial number", fontsize=BASE_FONT + 4)
    ax_al.set_xlim(-1, n_trials)
    ax_al.spines[["top", "right"]].set_visible(False)
    ax_al.legend(loc="upper left", fontsize=BASE_FONT + 1.5, framealpha=0.85,
                 borderpad=0.5)
    ax_al.text(-0.055, 0.0, "Alignment",
               transform=ax_al.transAxes,
               fontsize=BASE_FONT + 2, color="dimgray", fontweight="bold",
               ha="right", va="bottom")

    for ax in tick_axes:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Choice timeline + Rescorla-Wagner model comparison.",
    )
    add_session_arg(parser)
    add_save_arg(parser)
    parser.add_argument(
        "--alpha", type=float, default=0.1, metavar="A",
        help="RW learning rate, 0 < A < 1  (default: 0.1)",
    )
    parser.add_argument(
        "--beta", type=float, default=5.0, metavar="B",
        help="Softmax inverse temperature, B > 0  (default: 5.0)",
    )
    parser.add_argument(
        "--phi", type=float, default=0.0, metavar="P",
        help="Perseveration strength (0 = none, >0 = sticky)  (default: 0.0)",
    )
    parser.add_argument(
        "--window", type=int, default=20, metavar="W",
        help="Rolling window size for alignment metric  (default: 20)",
    )
    parser.add_argument(
        "--mode", choices=["shadow", "simulate"], default="shadow",
        help="shadow = observe participant & predict their choices (default); "
             "simulate = free-run, sampling the model's own choices and rewards",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="RNG seed for --mode simulate (reproducibility)",
    )
    parser.add_argument(
        "--trials", type=int, default=None, metavar="N",
        help="Number of trials to display (default: all trials)",
    )
    args = parser.parse_args()

    sess = Session(args.session)
    rng  = np.random.default_rng(args.seed)
    fig  = plot_choice_timeline(
        sess, alpha=args.alpha, beta=args.beta, phi=args.phi,
        window=args.window, mode=args.mode, rng=rng, n_trials=args.trials,
    )
    maybe_save(fig, args, prefix=f"choice_timeline_{args.mode}")
    plt.show()


if __name__ == "__main__":
    main()
