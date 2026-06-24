"""behaviour_overview.py — Raw choice behaviour with true reward probability.

Three horizontal bands per session share one axes:

  [Safe strip, above]   — vertical tick lines for each safe choice
                          seagreen = rewarded, muted = unrewarded

  [Probability band]    — P(big reward | Gamble) as a block-level step line
                          (orange) with a dashed reference at y=1.0 for the
                          safe arm (always certain).

  [Gamble strip, below] — vertical tick lines for each gamble choice
                          darkorange = rewarded, muted = unrewarded

  [Non-responding]      — grey tick marks at the very bottom

  Vertical dashed lines mark positive block boundaries.

Usage
-----
    python -m analysis.behaviour_overview                    # all sessions, 4×2 grid
    python -m analysis.behaviour_overview --session 20250714 # single session
    python -m analysis.behaviour_overview --save             # auto-save PNG
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session import Session
from utils import REPO_ROOT, RESULTS_DIR, SESSION, add_session_arg, add_save_arg


FNTSIZE = 10

# ---------------------------------------------------------------------------
# Colours
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


COL_G_REW  = "darkorange"
COL_G_NREW = desaturate("darkorange", 0.30)
COL_S_REW  = "seagreen"
COL_S_NREW = desaturate("seagreen",   0.30)
COL_NONR   = "#bbbbbb"

# ---------------------------------------------------------------------------
# Y-axis layout constants
# The probability axis runs from P_MIN to P_MAX (0–1).
# Strips are carved out above and below by padding the axes range.
# ---------------------------------------------------------------------------

P_MIN, P_MAX   = 0.0, 1.0      # true probability range

SAFE_STRIP_LO  =  1.08         # bottom of safe-choice strip
SAFE_STRIP_HI  =  1.30         # top of safe-choice strip
SAFE_TICK_Y    =  1.19         # midline for safe tick marks

GAMBLE_STRIP_LO = -0.30        # bottom of gamble-choice strip
GAMBLE_STRIP_HI = -0.08        # top of gamble-choice strip
GAMBLE_TICK_Y   = -0.19        # midline for gamble tick marks

NONR_Y         = -0.40         # non-responding tick midline
NONR_STRIP_LO  = -0.47

YMIN = NONR_STRIP_LO - 0.02
YMAX = SAFE_STRIP_HI + 0.04

TICK_HALF  = 0.07              # half-height of a choice tick
NONR_HALF  = 0.04


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_boundaries(trials) -> np.ndarray:
    block = trials["Block"].to_numpy()
    valid = block > 0
    bounds = []
    for t in range(1, len(block)):
        if valid[t] and valid[t - 1] and block[t] != block[t - 1]:
            bounds.append(t - 0.5)
    return np.array(bounds)


def _ticks(ax, x_positions, y_mid, half, color, alpha=0.85, lw=1.2, label=None):
    """Draw vertical tick lines centred on y_mid."""
    if len(x_positions) == 0:
        return
    ax.vlines(x_positions,
              ymin=y_mid - half, ymax=y_mid + half,
              colors=color, linewidth=lw, alpha=alpha,
              label=label)


# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------

def draw_session(ax: plt.Axes, sess: Session) -> None:
    trials = sess.trials
    n      = len(trials)
    idx    = np.arange(n)

    responding  = sess.responding_mask
    valid_block = (trials["Block"].to_numpy() > 0) & responding

    arm = trials["ChosenArm_G1S0"].to_numpy()
    rew = trials["Rewarded"].to_numpy()
    p_g = trials["P_BigReward_Gamble"].to_numpy()

    gr   = valid_block & (arm == 1) & (rew == 1)   # gamble, rewarded
    gn   = valid_block & (arm == 1) & (rew == 0)   # gamble, unrewarded
    sr   = valid_block & (arm == 0) & (rew == 1)   # safe,   rewarded
    sn   = valid_block & (arm == 0) & (rew == 0)   # safe,   unrewarded (rare)
    nonr = ~responding

    # ---- Background strip shading ----------------------------------------
    ax.axhspan(SAFE_STRIP_LO,   SAFE_STRIP_HI,   color="#f0f9f4", zorder=0)
    ax.axhspan(GAMBLE_STRIP_LO, GAMBLE_STRIP_HI, color="#fff5ea", zorder=0)
    ax.axhspan(NONR_STRIP_LO,   GAMBLE_STRIP_LO, color="#f5f5f5", zorder=0)

    # ---- Block boundaries ------------------------------------------------
    for x in _block_boundaries(trials):
        ax.axvline(x, color="#cccccc", lw=0.7, ls="--", zorder=1)

    # ---- Probability band ------------------------------------------------
    # Safe arm reference (always certain)
    ax.axhline(1.0, color=COL_S_REW, lw=1.0, ls="--", alpha=0.45, zorder=2)

    # Gamble probability step line + fill
    ax.step(idx, p_g, where="post",
            color=COL_G_REW, lw=1.6, alpha=0.6, zorder=3)
    ax.fill_between(idx, 0, p_g, step="post",
                    color=COL_G_REW, alpha=0.07, zorder=2)

    # ---- Safe-choice ticks (strip ABOVE probability band) ----------------
    _ticks(ax, idx[sr], SAFE_TICK_Y, TICK_HALF,
           color=COL_S_REW,  label=f"Safe — rewarded   (n={sr.sum()})")
    _ticks(ax, idx[sn], SAFE_TICK_Y, TICK_HALF,
           color=COL_S_NREW, label=f"Safe — unrewarded (n={sn.sum()})" if sn.any() else None)

    # ---- Gamble-choice ticks (strip BELOW probability band) --------------
    _ticks(ax, idx[gr], GAMBLE_TICK_Y, TICK_HALF,
           color=COL_G_REW,  label=f"Gamble — rewarded   (n={gr.sum()})")
    _ticks(ax, idx[gn], GAMBLE_TICK_Y, TICK_HALF,
           color=COL_G_NREW, label=f"Gamble — unrewarded (n={gn.sum()})")

    # ---- Non-responding marks --------------------------------------------
    if nonr.any():
        _ticks(ax, idx[nonr], NONR_Y, NONR_HALF,
               color=COL_NONR, alpha=0.5, lw=0.8,
               label=f"No response (n={nonr.sum()})")

    # ---- Strip labels (right margin) -------------------------------------
    ax.text(1.002, (SAFE_STRIP_LO + SAFE_STRIP_HI) / 2, "Safe",
            transform=ax.get_yaxis_transform(),
            fontsize=FNTSIZE, color="seagreen", va="center")
    ax.text(1.002, (GAMBLE_STRIP_LO + GAMBLE_STRIP_HI) / 2, "Gamble",
            transform=ax.get_yaxis_transform(),
            fontsize=FNTSIZE, color="darkorange", va="center")

    # ---- Summary annotation ----------------------------------------------
    n_valid  = valid_block.sum()
    pct_g    = 100 * gr.sum() / n_valid if n_valid else 0
    pct_g_all = 100 * (gr.sum() + gn.sum()) / n_valid if n_valid else 0
    ax.text(0.165, GAMBLE_STRIP_LO - 0.1,
            f"Gamble {pct_g_all:.0f} %  |  n_trials={n}",
            transform=ax.get_yaxis_transform(),
            fontsize=FNTSIZE-0.5, color="dimgray", ha="right", va="bottom")

    # ---- Axes formatting -------------------------------------------------
    ax.set_xlim(-1, n)
    ax.set_ylim(YMIN, YMAX)

    # Only show y ticks in the probability band
    ax.set_yticks([0.0, 0.1, 0.2, 0.4, 0.8, 1.0])
    ax.set_yticklabels(["0", ".1", ".2", ".4", ".8", "1"], fontsize = FNTSIZE)
    ax.set_ylabel("P(reward | Gamble)", fontsize=FNTSIZE+1, labelpad=4)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Session {sess.id}", fontsize=FNTSIZE+2, pad=4)


# ---------------------------------------------------------------------------
# Single-session figure
# ---------------------------------------------------------------------------

def plot_single_session(sess: Session) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    draw_session(ax, sess)
    ax.set_xlabel("Trial number", fontsize=FNTSIZE+2)
    ax.legend(loc="best", fontsize=FNTSIZE-3, framealpha=0.88,
              ncol=2, handlelength=1.4, borderpad=0.5,
              bbox_to_anchor=(1.0, 0.0))
    #fig.suptitle(f"Choice behaviour — session {sess.id}", fontsize=FNTSIZE+4, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# All-sessions grid
# ---------------------------------------------------------------------------

SESSIONS_DEFAULT = [
    "20250521", "20250602", "20250605", "20250703",
    "20250707", "20250709", "20250710", "20250714",
]


def plot_all_sessions(sessions: list[str] | None = None) -> plt.Figure:
    if sessions is None:
        sessions = SESSIONS_DEFAULT

    n_cols = 2
    n_rows = (len(sessions) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, n_rows * 3.6))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.50, wspace=0.22)

    legend_done = False
    for k, sid in enumerate(sessions):
        row, col = divmod(k, n_cols)
        ax = fig.add_subplot(gs[row, col])
        try:
            draw_session(ax, Session(sid))
        except Exception as exc:
            ax.text(0.5, 0.5, f"No data\n({exc})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=FNTSIZE+1, color="gray")
            ax.set_title(f"Session {sid}", fontsize=FNTSIZE+2)
            continue

        ax.set_xlabel("Trial", fontsize=FNTSIZE+1)

        if not legend_done:
            ax.legend(loc="lower right", fontsize=FNTSIZE-1, framealpha=0.85,
                      ncol=2, handlelength=1.0, borderpad=0.4)
            legend_done = True

    fig.suptitle("Choice behaviour — all sessions", fontsize=FNTSIZE+6, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_sessions() -> list[str]:
    import re
    return sorted(
        name for name in os.listdir(REPO_ROOT)
        if re.fullmatch(r"\d{8}", name)
        and os.path.isdir(os.path.join(REPO_ROOT, name))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_session_arg(parser)
    add_save_arg(parser)
    parser.add_argument(
        "--all", action="store_true",
        help="Plot all sessions in a 4×2 overview figure (ignores --session)",
    )
    args = parser.parse_args()

    if args.all:
        sessions = _discover_sessions() or SESSIONS_DEFAULT
        fig = plot_all_sessions(sessions)
        prefix = "behaviour_overview_all"
        out_dir = os.path.join(RESULTS_DIR, "figures")
    else:
        fig = plot_single_session(Session(args.session))
        prefix = f"behaviour_overview_{args.session}"
        out_dir = os.path.join(RESULTS_DIR, "figures", args.session)

    if args.save is not None:
        if args.save:
            path = args.save
        else:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{prefix}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {path}")

    plt.show()


if __name__ == "__main__":
    main()
