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
  [Participant]
    Gamble row : orange ticks — bright = rewarded, muted = unrewarded
    Safe   row : green  ticks — bright = rewarded, muted = unrewarded

  [RW Model  (α, β, φ) — shadow | simulate]
    Gamble row : same colour scheme, drawn from the model's predicted/own choice
    Safe   row :   "
    P(Gamble)  : thin black line overlaid on the model gamble row

  [Alignment]
    Rolling match-rate line (model choice == participant choice)
    Fill: blue above chance, red below chance

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
):
    """Build and return the 5-row figure (shadow or simulate mode)."""
    trials     = sess.trials
    responding = sess.responding_mask
    arm        = trials["ChosenArm_G1S0"].to_numpy()
    rew        = trials["Rewarded"].to_numpy()
    n_trials   = len(trials)
    idx        = np.arange(n_trials)
    is_sim     = (mode == "simulate")

    # ---- Run model ----------------------------------------------------
    model_choice, p_gamble, _, _, model_rewards = run_rw_model(
        trials, alpha=alpha, beta=beta, phi=phi, mode=mode, rng=rng,
    )
    roll_rate, match_arr = rolling_match(arm, model_choice, window=window)
    overall_acc = float(np.nanmean(match_arr))

    # ---- Colours ------------------------------------------------------
    col_gr = CONDITIONS["G+R"]["color"]         # darkorange (gamble rewarded)
    col_gn = desaturate(col_gr, factor=0.28)    # muted orange  (gamble unrewarded)
    col_sr = CONDITIONS["S+R"]["color"]         # seagreen      (safe rewarded)
    col_sn = desaturate(col_sr, factor=0.28)    # muted green   (safe unrewarded)

    # ---- Participant masks --------------------------------------------
    p_gr = responding & (arm == 1) & (rew == 1)
    p_gn = responding & (arm == 1) & (rew == 0)
    p_sr = responding & (arm == 0) & (rew == 1)
    p_sn = responding & (arm == 0) & (rew == 0)

    # ---- Model masks (model's choice × its row's outcome) ------------
    # In shadow mode model_rewards is the participant's reward; in simulate
    # mode it is the model's own sampled reward.
    model_resp = ~np.isnan(model_choice)
    m_gr = model_resp & (model_choice == 1) & (model_rewards == 1)
    m_gn = model_resp & (model_choice == 1) & (model_rewards == 0)
    m_sr = model_resp & (model_choice == 0) & (model_rewards == 1)
    m_sn = model_resp & (model_choice == 0) & (model_rewards == 0)

    # ---- Figure layout ------------------------------------------------
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[1, 1, 1, 1, 2.5],
        hspace=0.08,
    )
    ax_pg = fig.add_subplot(gs[0])   # Participant – Gamble
    ax_ps = fig.add_subplot(gs[1])   # Participant – Safe
    ax_mg = fig.add_subplot(gs[2])   # Model – Gamble
    ax_ms = fig.add_subplot(gs[3])   # Model – Safe
    ax_al = fig.add_subplot(gs[4])   # Alignment

    for ax in (ax_ps, ax_mg, ax_ms, ax_al):
        ax.sharex(ax_pg)

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
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6,
                      rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.8,
                  handlelength=1.0, ncol=2, borderpad=0.4)

    # ---- Section background bands -------------------------------------
    for ax in (ax_pg, ax_ps):
        ax.set_facecolor("#fafafa")
    for ax in (ax_mg, ax_ms):
        ax.set_facecolor("#f0f4ff")

    # ---- Draw participant rows ----------------------------------------
    tick_row(ax_pg, [
        (idx[p_gr], col_gr, f"Rewarded  (n={p_gr.sum()})"),
        (idx[p_gn], col_gn, f"Unrewarded (n={p_gn.sum()})"),
    ], ylabel="Gamble")

    p_safe_rows = [(idx[p_sr], col_sr, f"Rewarded  (n={p_sr.sum()})")]
    if p_sn.sum():
        p_safe_rows.append((idx[p_sn], col_sn, f"Unrewarded (n={p_sn.sum()})"))
    tick_row(ax_ps, p_safe_rows, ylabel="Safe")

    # ---- Draw model rows ---------------------------------------------
    verb = "Chose" if is_sim else "Predicts"
    tick_row(ax_mg, [
        (idx[m_gr], col_gr, f"{verb} G, rewarded  (n={m_gr.sum()})"),
        (idx[m_gn], col_gn, f"{verb} G, unrewarded (n={m_gn.sum()})"),
    ], ylabel="Gamble")

    m_safe_rows = [(idx[m_sr], col_sr, f"{verb} S, rewarded  (n={m_sr.sum()})")]
    if m_sn.sum():
        m_safe_rows.append((idx[m_sn], col_sn, f"{verb} S, unrewarded (n={m_sn.sum()})"))
    tick_row(ax_ms, m_safe_rows, ylabel="Safe")

    # Overlay P(Gamble) line on model gamble row
    ax_mg_r = ax_mg.twinx()
    ax_mg_r.plot(idx, p_gamble, color="black", lw=0.9, alpha=0.35,
                 label="P(Gamble)")
    ax_mg_r.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.2)
    ax_mg_r.set_ylim(0, 1)
    ax_mg_r.set_yticks([0, 0.5, 1])
    ax_mg_r.set_yticklabels(["0", ".5", "1"], fontsize=6)
    ax_mg_r.set_ylabel("P(G)", fontsize=7, labelpad=2)
    ax_mg_r.spines[["top", "left"]].set_visible(False)

    # ---- Section labels (left margin) --------------------------------
    for ax, label in [
        (ax_pg, "Participant"),
        (ax_mg, f"RW model ({mode})\nα={alpha}, β={beta}, φ={phi}"),
    ]:
        ax.text(
            -0.055, 0.0, label,
            transform=ax.transAxes,
            fontsize=8, color="dimgray", fontweight="bold",
            ha="right", va="bottom", linespacing=1.4,
        )

    # ---- Title --------------------------------------------------------
    ax_pg.set_title(
        f"Decision sequence ({mode}) — session {sess.id}",
        fontsize=11, pad=8,
    )

    # ---- Alignment row -----------------------------------------------
    ax_al.axhline(0.5, color="gray", lw=0.9, ls="--", alpha=0.6,
                  label="Chance (50 %)")
    ax_al.plot(idx, roll_rate, color="steelblue", lw=1.6, zorder=3,
               label=f"Rolling {window}-trial match")
    acc_word = "match" if is_sim else "accuracy"
    ax_al.axhline(overall_acc, color="steelblue", lw=1.1, ls=":",
                  alpha=0.85, label=f"Overall {acc_word}: {overall_acc:.1%}")

    # Fill: above chance = blue, below = red
    ax_al.fill_between(
        idx, 0.5, roll_rate,
        where=(roll_rate >= 0.5), interpolate=True,
        color="steelblue", alpha=0.15, zorder=1,
    )
    ax_al.fill_between(
        idx, roll_rate, 0.5,
        where=(roll_rate < 0.5), interpolate=True,
        color="firebrick", alpha=0.15, zorder=1,
    )

    ax_al.set_ylim(0, 1.05)
    ax_al.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_al.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8)
    ax_al.set_ylabel("Match\nrate", fontsize=9, labelpad=6,
                     rotation=0, ha="right", va="center")
    ax_al.set_xlabel("Trial number", fontsize=10)
    ax_al.set_xlim(-1, n_trials)
    ax_al.spines[["top", "right"]].set_visible(False)
    ax_al.legend(loc="upper left", fontsize=7.5, framealpha=0.85,
                 borderpad=0.5)
    ax_al.text(
        -0.055, 0.0, "Alignment",
        transform=ax_al.transAxes,
        fontsize=8, color="dimgray", fontweight="bold",
        ha="right", va="bottom",
    )

    # Hide x-tick labels on all tick rows
    for ax in (ax_pg, ax_ps, ax_mg, ax_ms):
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
    args = parser.parse_args()

    sess = Session(args.session)
    rng  = np.random.default_rng(args.seed)
    fig  = plot_choice_timeline(
        sess, alpha=args.alpha, beta=args.beta, phi=args.phi,
        window=args.window, mode=args.mode, rng=rng,
    )
    maybe_save(fig, args, prefix=f"choice_timeline_{args.mode}")
    plt.show()


if __name__ == "__main__":
    main()
