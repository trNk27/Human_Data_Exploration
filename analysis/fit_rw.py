"""fit_rw.py — Maximum-likelihood fitting of the RW + perseveration model.

Fits (α, β, φ) to a session's choice data by maximising the log-likelihood
of the participant's choices under the model, then prints a comparison table
(base model without φ vs full model with φ) and optionally plots the
choice_timeline with the best-fit parameters.

Usage
-----
    python analysis/fit_rw.py
    python analysis/fit_rw.py --session 20250602
    python analysis/fit_rw.py --restarts 50
    python analysis/fit_rw.py --no-plot
    python analysis/fit_rw.py --save
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session import Session
from utils import add_session_arg, add_save_arg, maybe_save
from analysis.choice_timeline import run_rw_model, rolling_match, plot_choice_timeline


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _nll(params: np.ndarray, trials, fix_phi: float | None) -> float:
    """Negative log-likelihood of participant choices under the RW model."""
    alpha, beta = params[0], params[1]
    phi = params[2] if fix_phi is None else fix_phi

    _, p_gamble, _, _, _ = run_rw_model(trials, alpha=alpha, beta=beta, phi=phi)

    arm       = trials["ChosenArm_G1S0"].to_numpy().astype(float)
    mask      = (trials["NotResponding"].to_numpy() == 0) & ~np.isnan(p_gamble)
    p         = np.clip(p_gamble[mask], 1e-9, 1.0 - 1e-9)
    choices   = arm[mask]
    return -float(np.sum(choices * np.log(p) + (1.0 - choices) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Fitting routine
# ---------------------------------------------------------------------------

def fit_model(
    trials,
    include_phi: bool,
    n_restarts: int = 25,
    seed: int = 42,
) -> dict:
    """Fit RW model by minimising NLL with multiple random restarts.

    Parameters
    ----------
    trials      : pd.DataFrame  (Trials_Sync columns)
    include_phi : whether to fit φ (full model) or fix φ=0 (base model)
    n_restarts  : total number of optimisation starts (first is deterministic)
    seed        : RNG seed for reproducible random starts

    Returns
    -------
    dict with keys: alpha, beta, phi, nll, aic, bic, acc, n, k
    """
    rng = np.random.default_rng(seed)

    if include_phi:
        bounds    = [(0.01, 0.99), (0.1, 30.0), (-10.0, 10.0)]
        x0_det    = [0.1, 5.0, 0.0]
        k         = 3
        objective = lambda p: _nll(p, trials, fix_phi=None)
    else:
        bounds    = [(0.01, 0.99), (0.1, 30.0)]
        x0_det    = [0.1, 5.0]
        k         = 2
        objective = lambda p: _nll(p, trials, fix_phi=0.0)

    n_resp = int((trials["NotResponding"] == 0).sum())

    starts = [x0_det] + [
        [rng.uniform(lo, hi) for lo, hi in bounds]
        for _ in range(n_restarts - 1)
    ]

    best_nll = np.inf
    best_x   = None
    for x0 in starts:
        res = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if res.fun < best_nll:
            best_nll = res.fun
            best_x   = res.x

    alpha = float(best_x[0])
    beta  = float(best_x[1])
    phi   = float(best_x[2]) if include_phi else 0.0

    arm = trials["ChosenArm_G1S0"].to_numpy()
    mc, _, _, _, _ = run_rw_model(trials, alpha=alpha, beta=beta, phi=phi)
    _, match_arr = rolling_match(arm, mc)
    acc = float(np.nanmean(match_arr))

    return dict(
        alpha=alpha, beta=beta, phi=phi,
        nll=best_nll,
        aic=2.0 * k + 2.0 * best_nll,
        bic=k * np.log(n_resp) + 2.0 * best_nll,
        acc=acc, n=n_resp, k=k,
    )


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def _null_nll(trials) -> float:
    """NLL of a model that always predicts P(Gamble) = 0.5."""
    n_resp = int((trials["NotResponding"] == 0).sum())
    return n_resp * np.log(2.0)


def print_results(sess_id: str, base: dict, full: dict, null_nll: float) -> None:
    """Print a comparison table to stdout."""
    sep = "-" * 52

    pseudo_r2_base = 1.0 - base["nll"] / null_nll
    pseudo_r2_full = 1.0 - full["nll"] / null_nll

    lrt_stat = 2.0 * (base["nll"] - full["nll"])
    lrt_p    = chi2.sf(lrt_stat, df=1)

    print(f"\nSession {sess_id}   (n = {full['n']} responding trials)\n")
    print(f"{'':28s}  {'Base (a,b)':>10s}  {'Full (a,b,phi)':>14s}")
    print(sep)
    print(f"  alpha  learning rate      {base['alpha']:>10.4f}  {full['alpha']:>14.4f}")
    print(f"  beta   inv. temperature   {base['beta']:>10.4f}  {full['beta']:>14.4f}")
    print(f"  phi    perseveration      {'--':>10s}  {full['phi']:>14.4f}")
    print(sep)
    print(f"  NLL                       {base['nll']:>10.2f}  {full['nll']:>12.2f}")
    print(f"  AIC                       {base['aic']:>10.2f}  {full['aic']:>12.2f}")
    print(f"  BIC                       {base['bic']:>10.2f}  {full['bic']:>12.2f}")
    print(f"  McFadden R2               {pseudo_r2_base:>10.4f}  {pseudo_r2_full:>12.4f}")
    print(f"  Match rate                {base['acc']:>9.1%}   {full['acc']:>11.1%}")
    print(sep)

    direction = "sticky  (phi > 0)" if full["phi"] > 0 else "alternating  (phi < 0)"
    sig       = "***" if lrt_p < 0.001 else ("**" if lrt_p < 0.01 else ("*" if lrt_p < 0.05 else "n.s."))
    print(f"  LRT  chi2(1) = {lrt_stat:.2f},  p = {lrt_p:.2e}  {sig}")
    if lrt_p < 0.05:
        print(f"  phi significantly improves fit  ->  {direction}")
    else:
        print("  phi does not significantly improve fit  ->  use base model")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit RW + perseveration model to participant choices.",
    )
    add_session_arg(parser)
    add_save_arg(parser)
    parser.add_argument(
        "--restarts", type=int, default=25, metavar="N",
        help="Number of optimisation restarts  (default: 25)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the choice_timeline plot after fitting",
    )
    parser.add_argument(
        "--trials", type=int, default=None, metavar="N",
        help="Number of trials to display in the plot (default: all; fitting always uses all trials)",
    )
    args = parser.parse_args()

    sess = Session(args.session)

    print("Fitting base model  (alpha, beta)  ...", end="", flush=True)
    base = fit_model(sess.trials, include_phi=False, n_restarts=args.restarts)
    print("  done")

    print("Fitting full model  (alpha, beta, phi)  ...", end="", flush=True)
    full = fit_model(sess.trials, include_phi=True,  n_restarts=args.restarts)
    print("  done\n")

    print_results(sess.id, base, full, _null_nll(sess.trials))

    if not args.no_plot:
        fig = plot_choice_timeline(
            sess,
            alpha=full["alpha"],
            beta=full["beta"],
            phi=full["phi"],
            n_trials=args.trials,
        )
        maybe_save(fig, args, prefix="fit_rw")
        plt.show()


if __name__ == "__main__":
    main()
