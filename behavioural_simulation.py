"""behavioural_simulation.py — Generative behavioural models for the Gamble / Safe task.

Each model makes its own choices and samples rewards from the trial's reward
probabilities.  It does NOT shadow the participant (unlike the observer model
in choice_timeline.py).

Models
------
  rw    : Rescorla-Wagner Q-learning + softmax
  ck    : Choice Kernel + softmax
  rw_ck : Rescorla-Wagner + Choice Kernel combined

Usage
-----
    python behavioural_simulation.py
    python behavioural_simulation.py --model ck --alpha_c 0.3 --beta_c 3
    python behavioural_simulation.py --model rw_ck --alpha 0.1 --beta 3 --alpha_c 0.3 --beta_c 2
    python behavioural_simulation.py --session 20250602 --seed 42 --save
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from session import Session
from utils import CONDITIONS, add_session_arg, add_save_arg, maybe_save


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def desaturate(color, factor: float = 0.25) -> tuple:
    r, g, b, *a = mcolors.to_rgba(color)
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return (
        grey + factor * (r - grey),
        grey + factor * (g - grey),
        grey + factor * (b - grey),
        *(a if a else (1.0,)),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    choices:    np.ndarray  # (n_trials,) 1=Gamble, 0=Safe, NaN=non-responding
    rewards:    np.ndarray  # (n_trials,) 1=rewarded, 0=not, NaN=non-responding
    p_gamble:   np.ndarray  # (n_trials,) P(Gamble) before choice, NaN=non-responding
    model_name: str
    params:     dict


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------

def simulate_rw(
    trials,
    alpha: float = 0.1,
    beta:  float = 5.0,
    rng:   np.random.Generator | None = None,
) -> SimResult:
    """Rescorla-Wagner Q-learning + softmax.

    Parameters
    ----------
    alpha : learning rate ∈ (0, 1)
    beta  : softmax inverse temperature > 0
    """
    if rng is None:
        rng = np.random.default_rng()

    n          = len(trials)
    responding = (trials["NotResponding"] == 0).to_numpy()
    p_big      = trials["P_BigReward_Gamble"].to_numpy()
    p_small    = trials["P_SmallReward_Safe"].to_numpy()

    choices  = np.full(n, np.nan)
    rewards  = np.full(n, np.nan)
    p_gamble = np.full(n, np.nan)

    q = {1: 0.5, 0: 0.5}

    for t in range(n):
        if not responding[t]:
            continue

        p_g = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0])))
        p_gamble[t] = p_g
        choice = int(rng.random() < p_g)
        choices[t] = choice

        reward_prob = p_big[t] if choice == 1 else p_small[t]
        reward = int(rng.random() < reward_prob)
        rewards[t] = reward

        q[choice] += alpha * (reward - q[choice])

    return SimResult(choices, rewards, p_gamble, "rw", {"alpha": alpha, "beta": beta})


def simulate_ck(
    trials,
    alpha_c: float = 0.3,
    beta_c:  float = 3.0,
    rng:     np.random.Generator | None = None,
) -> SimResult:
    """Choice Kernel + softmax.

    Captures perseverative tendencies via a decaying trace of recent choices.
    Rewards are sampled but do not drive updates — only the choice history does.

    Parameters
    ----------
    alpha_c : kernel learning rate ∈ (0, 1)
    beta_c  : softmax weight on the choice kernel > 0
    """
    if rng is None:
        rng = np.random.default_rng()

    n          = len(trials)
    responding = (trials["NotResponding"] == 0).to_numpy()
    p_big      = trials["P_BigReward_Gamble"].to_numpy()
    p_small    = trials["P_SmallReward_Safe"].to_numpy()

    choices  = np.full(n, np.nan)
    rewards  = np.full(n, np.nan)
    p_gamble = np.full(n, np.nan)

    c = {1: 0.0, 0: 0.0}

    for t in range(n):
        if not responding[t]:
            continue

        p_g = 1.0 / (1.0 + np.exp(-beta_c * (c[1] - c[0])))
        p_gamble[t] = p_g
        choice = int(rng.random() < p_g)
        choices[t] = choice

        reward_prob = p_big[t] if choice == 1 else p_small[t]
        rewards[t] = int(rng.random() < reward_prob)

        unchosen = 1 - choice
        c[choice]   += alpha_c * (1.0 - c[choice])
        c[unchosen] += alpha_c * (0.0 - c[unchosen])

    return SimResult(choices, rewards, p_gamble, "ck", {"alpha_c": alpha_c, "beta_c": beta_c})


def simulate_rw_ck(
    trials,
    alpha:   float = 0.1,
    beta:    float = 3.0,
    alpha_c: float = 0.3,
    beta_c:  float = 2.0,
    rng:     np.random.Generator | None = None,
) -> SimResult:
    """Rescorla-Wagner + Choice Kernel combined.

    Decision value = β·(Q_G − Q_S) + β_c·(C_G − C_S).
    Q-values are updated by reward prediction errors; the choice kernel
    is updated by the choice history.

    Parameters
    ----------
    alpha   : Q-value learning rate ∈ (0, 1)
    beta    : softmax weight on Q-values > 0
    alpha_c : kernel learning rate ∈ (0, 1)
    beta_c  : softmax weight on the choice kernel > 0
    """
    if rng is None:
        rng = np.random.default_rng()

    n          = len(trials)
    responding = (trials["NotResponding"] == 0).to_numpy()
    p_big      = trials["P_BigReward_Gamble"].to_numpy()
    p_small    = trials["P_SmallReward_Safe"].to_numpy()

    choices  = np.full(n, np.nan)
    rewards  = np.full(n, np.nan)
    p_gamble = np.full(n, np.nan)

    q = {1: 0.5, 0: 0.5}
    c = {1: 0.0, 0: 0.0}

    for t in range(n):
        if not responding[t]:
            continue

        decision_value = beta * (q[1] - q[0]) + beta_c * (c[1] - c[0])
        p_g = 1.0 / (1.0 + np.exp(-decision_value))
        p_gamble[t] = p_g
        choice = int(rng.random() < p_g)
        choices[t] = choice

        reward_prob = p_big[t] if choice == 1 else p_small[t]
        reward = int(rng.random() < reward_prob)
        rewards[t] = reward

        unchosen = 1 - choice
        q[choice]   += alpha   * (reward - q[choice])
        c[choice]   += alpha_c * (1.0    - c[choice])
        c[unchosen] += alpha_c * (0.0    - c[unchosen])

    return SimResult(
        choices, rewards, p_gamble, "rw_ck",
        {"alpha": alpha, "beta": beta, "alpha_c": alpha_c, "beta_c": beta_c},
    )


MODELS = {
    "rw":    simulate_rw,
    "ck":    simulate_ck,
    "rw_ck": simulate_rw_ck,
}


# ---------------------------------------------------------------------------
# Rolling alignment
# ---------------------------------------------------------------------------

def rolling_match(
    participant_arm: np.ndarray,
    sim_choices:     np.ndarray,
    window:          int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling match rate between simulation and participant choices."""
    n     = len(participant_arm)
    match = np.full(n, np.nan)
    rate  = np.full(n, np.nan)

    valid = ~np.isnan(sim_choices)
    match[valid] = (participant_arm[valid] == sim_choices[valid]).astype(float)

    valid_idx  = np.where(valid)[0]
    match_vals = match[valid_idx]

    for k, t in enumerate(valid_idx):
        lo      = max(0, k - window + 1)
        rate[t] = np.mean(match_vals[lo : k + 1])

    return rate, match


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_simulation(
    sess:   Session,
    result: SimResult,
    window: int = 20,
):
    """5-row figure: participant / simulated choices / P(Gamble) / alignment."""
    trials     = sess.trials
    responding = sess.responding_mask
    p_arm      = trials["ChosenArm_G1S0"].to_numpy()
    p_rew      = trials["Rewarded"].to_numpy()
    n_trials   = len(trials)
    idx        = np.arange(n_trials)

    roll_rate, match_arr = rolling_match(p_arm, result.choices, window=window)
    overall_acc = float(np.nanmean(match_arr))

    col_gr = CONDITIONS["G+R"]["color"]
    col_gn = desaturate(col_gr, factor=0.28)
    col_sr = CONDITIONS["S+R"]["color"]
    col_sn = desaturate(col_sr, factor=0.28)

    # Participant masks
    p_gr = responding & (p_arm == 1) & (p_rew == 1)
    p_gn = responding & (p_arm == 1) & (p_rew == 0)
    p_sr = responding & (p_arm == 0) & (p_rew == 1)
    p_sn = responding & (p_arm == 0) & (p_rew == 0)

    # Simulation masks
    sim_resp = ~np.isnan(result.choices)
    s_gr = sim_resp & (result.choices == 1) & (result.rewards == 1)
    s_gn = sim_resp & (result.choices == 1) & (result.rewards == 0)
    s_sr = sim_resp & (result.choices == 0) & (result.rewards == 1)
    s_sn = sim_resp & (result.choices == 0) & (result.rewards == 0)

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[1, 1, 1, 1, 2.5], hspace=0.08)
    ax_pg = fig.add_subplot(gs[0])
    ax_ps = fig.add_subplot(gs[1])
    ax_sg = fig.add_subplot(gs[2])
    ax_ss = fig.add_subplot(gs[3])
    ax_al = fig.add_subplot(gs[4])

    for ax in (ax_ps, ax_sg, ax_ss, ax_al):
        ax.sharex(ax_pg)

    def tick_row(ax, rows, ylabel):
        for trial_indices, color, label in rows:
            if trial_indices.size == 0:
                continue
            ax.eventplot(trial_indices, orientation="horizontal",
                         lineoffsets=0.5, linelengths=0.75, linewidths=1.4,
                         colors=color, label=label)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6,
                      rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.8,
                  handlelength=1.0, ncol=2, borderpad=0.4)

    for ax in (ax_pg, ax_ps):
        ax.set_facecolor("#fafafa")
    for ax in (ax_sg, ax_ss):
        ax.set_facecolor("#f0f4ff")

    # Participant rows
    tick_row(ax_pg, [
        (idx[p_gr], col_gr, f"Rewarded (n={p_gr.sum()})"),
        (idx[p_gn], col_gn, f"Unrewarded (n={p_gn.sum()})"),
    ], ylabel="Gamble")

    p_safe_rows = [(idx[p_sr], col_sr, f"Rewarded (n={p_sr.sum()})")]
    if p_sn.sum():
        p_safe_rows.append((idx[p_sn], col_sn, f"Unrewarded (n={p_sn.sum()})"))
    tick_row(ax_ps, p_safe_rows, ylabel="Safe")

    # Simulation rows
    tick_row(ax_sg, [
        (idx[s_gr], col_gr, f"Rewarded (n={s_gr.sum()})"),
        (idx[s_gn], col_gn, f"Unrewarded (n={s_gn.sum()})"),
    ], ylabel="Gamble")

    s_safe_rows = [(idx[s_sr], col_sr, f"Rewarded (n={s_sr.sum()})")]
    if s_sn.sum():
        s_safe_rows.append((idx[s_sn], col_sn, f"Unrewarded (n={s_sn.sum()})"))
    tick_row(ax_ss, s_safe_rows, ylabel="Safe")

    # P(Gamble) overlay
    ax_sg_r = ax_sg.twinx()
    ax_sg_r.plot(idx, result.p_gamble, color="black", lw=0.9, alpha=0.35,
                 label="P(Gamble)")
    ax_sg_r.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.2)
    ax_sg_r.set_ylim(0, 1)
    ax_sg_r.set_yticks([0, 0.5, 1])
    ax_sg_r.set_yticklabels(["0", ".5", "1"], fontsize=6)
    ax_sg_r.set_ylabel("P(G)", fontsize=7, labelpad=2)
    ax_sg_r.spines[["top", "left"]].set_visible(False)

    # Section labels
    param_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
    for ax, label in [
        (ax_pg, "Participant"),
        (ax_sg, f"{result.model_name.upper()}\n{param_str}"),
    ]:
        ax.text(-0.055, 0.0, label,
                transform=ax.transAxes,
                fontsize=8, color="dimgray", fontweight="bold",
                ha="right", va="bottom", linespacing=1.4)

    ax_pg.set_title(
        f"Behavioural simulation — {result.model_name.upper()} — session {sess.id}",
        fontsize=11, pad=8,
    )

    # Alignment row
    ax_al.axhline(0.5, color="gray", lw=0.9, ls="--", alpha=0.6,
                  label="Chance (50 %)")
    ax_al.plot(idx, roll_rate, color="steelblue", lw=1.6, zorder=3,
               label=f"Rolling {window}-trial match")
    ax_al.axhline(overall_acc, color="steelblue", lw=1.1, ls=":",
                  alpha=0.85, label=f"Overall accuracy: {overall_acc:.1%}")

    ax_al.fill_between(idx, 0.5, roll_rate,
                       where=(roll_rate >= 0.5), interpolate=True,
                       color="steelblue", alpha=0.15, zorder=1)
    ax_al.fill_between(idx, roll_rate, 0.5,
                       where=(roll_rate < 0.5), interpolate=True,
                       color="firebrick", alpha=0.15, zorder=1)

    ax_al.set_ylim(0, 1.05)
    ax_al.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_al.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8)
    ax_al.set_ylabel("Match\nrate", fontsize=9, labelpad=6,
                     rotation=0, ha="right", va="center")
    ax_al.set_xlabel("Trial number", fontsize=10)
    ax_al.set_xlim(-1, n_trials)
    ax_al.spines[["top", "right"]].set_visible(False)
    ax_al.legend(loc="upper left", fontsize=7.5, framealpha=0.85, borderpad=0.5)
    ax_al.text(-0.055, 0.0, "Alignment",
               transform=ax_al.transAxes,
               fontsize=8, color="dimgray", fontweight="bold",
               ha="right", va="bottom")

    for ax in (ax_pg, ax_ps, ax_sg, ax_ss):
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generative behavioural simulation for the Gamble / Safe task.",
    )
    add_session_arg(parser)
    add_save_arg(parser)
    parser.add_argument(
        "--model", choices=list(MODELS), default="rw",
        help="Model to simulate (default: rw)",
    )
    parser.add_argument("--alpha",   type=float, default=0.1,  metavar="A",
                        help="RW learning rate (default: 0.1)")
    parser.add_argument("--beta",    type=float, default=5.0,  metavar="B",
                        help="Softmax inverse temperature for Q-values (default: 5.0)")
    parser.add_argument("--alpha_c", type=float, default=0.3,  metavar="AC",
                        help="Choice kernel learning rate (default: 0.3)")
    parser.add_argument("--beta_c",  type=float, default=3.0,  metavar="BC",
                        help="Softmax weight on choice kernel (default: 3.0)")
    parser.add_argument("--window",  type=int,   default=20,   metavar="W",
                        help="Rolling window size for alignment (default: 20)")
    parser.add_argument("--seed",    type=int,   default=None, metavar="S",
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng  = np.random.default_rng(args.seed)
    sess = Session(args.session)

    model_fn = MODELS[args.model]
    if args.model == "rw":
        result = model_fn(sess.trials, alpha=args.alpha, beta=args.beta, rng=rng)
    elif args.model == "ck":
        result = model_fn(sess.trials, alpha_c=args.alpha_c, beta_c=args.beta_c, rng=rng)
    else:  # rw_ck
        result = model_fn(sess.trials, alpha=args.alpha, beta=args.beta,
                          alpha_c=args.alpha_c, beta_c=args.beta_c, rng=rng)

    fig = plot_simulation(sess, result, window=args.window)
    maybe_save(fig, args, prefix=f"sim_{args.model}")
    plt.show()


if __name__ == "__main__":
    main()
