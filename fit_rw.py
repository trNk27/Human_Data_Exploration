"""fit_rw.py — Find the best Rescorla-Wagner parameters for a session.

Strategy
--------
1. Grid search over (alpha, beta) → landscape heatmap + coarse best point.
2. Nelder-Mead refinement starting from the grid best → precise optimum.

Objective: fraction of *responding* trials where the RW+softmax model's
predicted choice (argmax of softmax) matches the participant's actual choice.

Usage
-----
    python fit_rw.py
    python fit_rw.py --session 20250602
    python fit_rw.py --n 50                 # 50×50 grid (default 40)
    python fit_rw.py --save
    python fit_rw.py --show-timeline        # open choice_timeline with best params
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize

from session import Session
from utils import add_session_arg, add_save_arg, maybe_save
from choice_timeline import run_rw_model   # reuse without duplication


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def match_accuracy(trials, alpha: float, beta: float) -> float:
    """Fraction of responding trials where model choice == participant choice."""
    arm         = trials["ChosenArm_G1S0"].to_numpy()
    model_choice, *_ = run_rw_model(trials, alpha=alpha, beta=beta)
    valid = ~np.isnan(model_choice)
    if not valid.any():
        return 0.0
    return float(np.mean(arm[valid] == model_choice[valid]))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    trials,
    n: int = 40,
    alpha_range: tuple = (0.01, 0.60),
    beta_range:  tuple = (0.10, 15.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate match_accuracy on an n×n (alpha, beta) grid.

    Returns
    -------
    grid   : (n, n) float  — accuracy at each grid point
    alphas : (n,)   float  — alpha axis
    betas  : (n,)   float  — beta  axis
    """
    alphas = np.linspace(*alpha_range, n)
    betas  = np.linspace(*beta_range,  n)
    grid   = np.empty((n, n), dtype=float)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            grid[i, j] = match_accuracy(trials, a, b)

    return grid, alphas, betas


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    trials, alpha0: float, beta0: float
) -> tuple[float, float, float]:
    """Nelder-Mead refinement from the grid-search starting point.

    Returns (best_alpha, best_beta, best_accuracy).
    """
    def neg_acc(params):
        a, b = params
        # Soft out-of-bounds penalty so the optimiser stays sensible
        if not (1e-4 < a < 0.9999) or b < 1e-3:
            return 1.0
        return -match_accuracy(trials, a, b)

    result = minimize(
        neg_acc, x0=[alpha0, beta0],
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2_000},
    )
    best_alpha, best_beta = float(result.x[0]), float(result.x[1])
    best_acc              = -float(result.fun)
    return best_alpha, best_beta, best_acc


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(
    grid:        np.ndarray,
    alphas:      np.ndarray,
    betas:       np.ndarray,
    grid_alpha:  float,
    grid_beta:   float,
    fine_alpha:  float,
    fine_beta:   float,
    fine_acc:    float,
    sess_id:     str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        betas, alphas, grid,
        cmap="viridis", shading="auto",
        vmin=grid.min(), vmax=grid.max(),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Match rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Contour lines at every 2 %
    cs = ax.contour(
        betas, alphas, grid,
        levels=10, colors="white", linewidths=0.5, alpha=0.4,
    )
    ax.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=6, inline=True)

    # Mark grid best + fine-tuned best
    ax.plot(grid_beta,  grid_alpha,  "w*",  ms=13, zorder=5,
            label=f"Grid best  α={grid_alpha:.3f}, β={grid_beta:.2f}  "
                  f"({match_accuracy.__doc__ and ''})")
    ax.plot(fine_beta,  fine_alpha,  "r^",  ms=10, zorder=6,
            label=f"Fine-tuned  α={fine_alpha:.4f}, β={fine_beta:.3f}  "
                  f"({fine_acc:.1%})")

    ax.set_xlabel("β  (inverse temperature)", fontsize=11)
    ax.set_ylabel("α  (learning rate)",        fontsize=11)
    ax.set_title(
        f"RW model fit — session {sess_id}\n"
        f"Best: α = {fine_alpha:.4f},  β = {fine_beta:.3f}  "
        f"→  {fine_acc:.1%} match",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid-search + Nelder-Mead fit for RW model parameters.",
    )
    add_session_arg(parser)
    add_save_arg(parser)
    parser.add_argument(
        "--n", type=int, default=40, metavar="N",
        help="Grid resolution N × N  (default: 40)",
    )
    parser.add_argument(
        "--show-timeline", action="store_true",
        help="Open choice_timeline.py with the best-fit parameters after fitting",
    )
    args = parser.parse_args()

    sess   = Session(args.session)
    trials = sess.trials
    n_resp = int(sess.responding_mask.sum())

    ALPHA_RANGE = (0.01, 0.60)
    BETA_RANGE  = (0.01, 15.0)

    print(f"Session  : {sess.id}  ({n_resp} responding / {len(trials)} total trials)")
    print(f"Grid     : {args.n} x {args.n}  ...", end="", flush=True)

    grid, alphas, betas = grid_search(
        trials, n=args.n,
        alpha_range=ALPHA_RANGE,
        beta_range=BETA_RANGE,
    )

    best_ij    = np.unravel_index(np.argmax(grid), grid.shape)
    grid_alpha = float(alphas[best_ij[0]])
    grid_beta  = float(betas [best_ij[1]])
    print(
        f" done.\n"
        f"  Grid best : alpha={grid_alpha:.3f},  beta={grid_beta:.2f}  "
        f"({grid.max():.1%} match)"
    )

    # Warn if the optimum is sitting on a search boundary
    eps = (alphas[1] - alphas[0]) * 0.5
    if (abs(grid_alpha - ALPHA_RANGE[0]) < eps or
            abs(grid_alpha - ALPHA_RANGE[1]) < eps or
            abs(grid_beta  - BETA_RANGE[0])  < (betas[1]-betas[0])*0.5 or
            abs(grid_beta  - BETA_RANGE[1])  < (betas[1]-betas[0])*0.5):
        print(
            "  [!] Best point is at a grid boundary -- "
            "the true optimum may lie outside the search range."
        )

    print("Fine-tune : Nelder-Mead ...  ", end="", flush=True)
    fine_alpha, fine_beta, fine_acc = fine_tune(trials, grid_alpha, grid_beta)
    print(
        f" done.\n"
        f"  Fine best : alpha={fine_alpha:.4f},  beta={fine_beta:.3f}  "
        f"({fine_acc:.1%} match)"
    )

    fig = plot_heatmap(
        grid, alphas, betas,
        grid_alpha, grid_beta,
        fine_alpha, fine_beta, fine_acc,
        sess.id,
    )
    maybe_save(fig, args, prefix="rw_fit")
    plt.show()

    if args.show_timeline:
        print(
            f"\nLaunching choice_timeline.py  "
            f"--alpha {fine_alpha:.4f}  --beta {fine_beta:.3f} ..."
        )
        subprocess.run([
            sys.executable, "choice_timeline.py",
            "--session", args.session,
            "--alpha",   f"{fine_alpha:.4f}",
            "--beta",    f"{fine_beta:.3f}",
        ])


if __name__ == "__main__":
    main()
