"""choice_timeline.py — Participant decisions + Rescorla-Wagner model comparison.

Layout (top → bottom)
---------------------
  [Participant]
    Gamble row : orange ticks — bright = rewarded, muted = unrewarded
    Safe   row : green  ticks — bright = rewarded, muted = unrewarded

  [RW Model  (α, β)]
    Gamble row : same colour scheme, but drawn from the model's predicted choice
    Safe   row :   "
    P(Gamble)  : thin black line overlaid on the model gamble row

  [Alignment]
    Rolling match-rate line (model choice == participant choice)
    Fill: blue above chance, red below chance

Usage
-----
    python choice_timeline.py
    python choice_timeline.py --session 20250602
    python choice_timeline.py --alpha 0.15 --beta 4 --window 20
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trial-by-trial Rescorla-Wagner model with softmax decision rule.

    The model observes the *participant's* actual choices and outcomes to
    update its Q values. Predictions are generated *before* each outcome.

    Parameters
    ----------
    trials : pd.DataFrame  (Trials_Sync columns)
    alpha  : learning rate  ∈ (0, 1)
    beta   : inverse temperature (higher → more deterministic)

    Returns
    -------
    model_choice : (n_trials,) float  1=Gamble, 0=Safe, NaN=non-responding
    p_gamble     : (n_trials,) float  P(Gamble) at each trial,   NaN=non-responding
    q_gamble     : (n_trials,) float  Q(Gamble) before update,   NaN=non-responding
    q_safe       : (n_trials,) float  Q(Safe)   before update,   NaN=non-responding
    """
    n          = len(trials)
    arm        = trials["ChosenArm_G1S0"].to_numpy()       # participant's choice
    rew        = trials["Rewarded"].to_numpy()
    responding = (trials["NotResponding"] == 0).to_numpy()

    q = {1: 0.5, 0: 0.5}   # Q-values: gamble=1, safe=0

    model_choice = np.full(n, np.nan)
    p_gamble     = np.full(n, np.nan)
    q_gamble     = np.full(n, np.nan)
    q_safe       = np.full(n, np.nan)

    for t in range(n):
        if not responding[t]:
            continue

        # Snapshot current Q values (state *before* this trial)
        q_gamble[t] = q[1]
        q_safe[t]   = q[0]

        # Softmax: P(Gamble) = σ( β · (Q_G − Q_S) )
        p_g = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0])))
        p_gamble[t]     = p_g
        model_choice[t] = 1.0 if p_g >= 0.5 else 0.0

        # Update: participant's chosen arm, participant's observed reward
        chosen = int(arm[t])
        q[chosen] += alpha * (float(rew[t]) - q[chosen])

    return model_choice, p_gamble, q_gamble, q_safe


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
    window: int = 20,
):
    """Build and return the 5-row figure."""
    trials     = sess.trials
    responding = sess.responding_mask
    arm        = trials["ChosenArm_G1S0"].to_numpy()
    rew        = trials["Rewarded"].to_numpy()
    n_trials   = len(trials)
    idx        = np.arange(n_trials)

    # ---- Run model ----------------------------------------------------
    model_choice, p_gamble, _, _ = run_rw_model(
        trials, alpha=alpha, beta=beta,
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

    # ---- Model masks (model's predicted choice × actual outcome) -----
    model_resp = ~np.isnan(model_choice)
    m_gr = model_resp & (model_choice == 1) & (rew == 1)
    m_gn = model_resp & (model_choice == 1) & (rew == 0)
    m_sr = model_resp & (model_choice == 0) & (rew == 1)
    m_sn = model_resp & (model_choice == 0) & (rew == 0)

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
    tick_row(ax_mg, [
        (idx[m_gr], col_gr, f"Predicts G, rewarded  (n={m_gr.sum()})"),
        (idx[m_gn], col_gn, f"Predicts G, unrewarded (n={m_gn.sum()})"),
    ], ylabel="Gamble")

    m_safe_rows = [(idx[m_sr], col_sr, f"Predicts S, rewarded  (n={m_sr.sum()})")]
    if m_sn.sum():
        m_safe_rows.append((idx[m_sn], col_sn, f"Predicts S, unrewarded (n={m_sn.sum()})"))
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
        (ax_mg, f"RW model\nα={alpha}, β={beta}"),
    ]:
        ax.text(
            -0.055, 0.0, label,
            transform=ax.transAxes,
            fontsize=8, color="dimgray", fontweight="bold",
            ha="right", va="bottom", linespacing=1.4,
        )

    # ---- Title --------------------------------------------------------
    ax_pg.set_title(
        f"Decision sequence — session {sess.id}",
        fontsize=11, pad=8,
    )

    # ---- Alignment row -----------------------------------------------
    ax_al.axhline(0.5, color="gray", lw=0.9, ls="--", alpha=0.6,
                  label="Chance (50 %)")
    ax_al.plot(idx, roll_rate, color="steelblue", lw=1.6, zorder=3,
               label=f"Rolling {window}-trial match")
    ax_al.axhline(overall_acc, color="steelblue", lw=1.1, ls=":",
                  alpha=0.85, label=f"Overall accuracy: {overall_acc:.1%}")

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
        "--window", type=int, default=20, metavar="W",
        help="Rolling window size for alignment metric  (default: 20)",
    )
    args = parser.parse_args()

    sess = Session(args.session)
    fig  = plot_choice_timeline(
        sess, alpha=args.alpha, beta=args.beta, window=args.window,
    )
    maybe_save(fig, args, prefix="choice_timeline")
    plt.show()


if __name__ == "__main__":
    main()
