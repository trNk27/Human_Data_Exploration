"""End-to-end HGF pipeline.

Usage
-----
    python -m analysis.hgf.run [--sessions 20250714 ...] [--no-hierarchical]
                                [--n-recovery 24] [--n-power 20] [--out DIR]

Stages (executed in order)
---------------------------
1. Load all sessions and build SessionModel objects.
2. Shared-parameter (complete-pooling) MAP fit.
3. Per-session (no-pooling) MAP fits — for drift plot + power benchmark.
4. Extract and save per-session latent trajectory CSVs.
5. Per-session plots: trajectory, learning-rate, volatility.
6. Model comparison vs Rescorla-Wagner (plain + stickiness).
7. Parameter recovery (n_recovery simulate→refit cycles).
8. Power check (session-to-session shift detection).
9. Hierarchical partial-pooling model via NUTS (skippable).
10. Cross-session summary plots.
11. Write fitted_parameters.csv and print a summary table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — works whether called as `python -m analysis.hgf.run`
# or `python analysis/hgf/run.py`
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.hgf.config import (
    SEED, RESULTS_HGF, ensure_results_dir, BASELINE, ParamSpace, make_param_space,
)
from analysis.hgf.data import load_session, list_sessions as available_sessions
from analysis.hgf.model import SessionModel
from analysis.hgf.fit import shared_map_fit, fit_all_separate
from analysis.hgf.trajectories import session_trajectory
from analysis.hgf.comparison import fit_rw, compare_models
from analysis.hgf.recovery import parameter_recovery, recovery_summary
from analysis.hgf.power import power_check, power_summary
from analysis.hgf import plots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HGF pipeline — MAP fit + outputs")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Session IDs to include (default: all available)")
    p.add_argument("--no-hierarchical", action="store_true",
                   help="Skip the PyMC/NUTS hierarchical fit (fast mode)")
    p.add_argument("--n-recovery", type=int, default=24,
                   help="Number of simulate→refit cycles for parameter recovery")
    p.add_argument("--n-power", type=int, default=20,
                   help="Number of simulated session pairs per (param, shift)")
    p.add_argument("--n-restarts", type=int, default=5,
                   help="MAP optimisation restarts per fit")
    p.add_argument("--free", nargs="*", default=None, choices=["omega3", "kappa"],
                   help="Additionally free these fixed HGF params (omega3, kappa). "
                        "Default: none (baseline omega2/beta/bias). When given, output "
                        "defaults to results/hgf_extended/.")
    p.add_argument("--out", default=None,
                   help="Output directory (default: results/hgf/, or results/hgf_extended/ "
                        "when --free is given)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(sessions: list[str] | None = None,
        run_hierarchical: bool = True,
        n_recovery: int = 24,
        n_power: int = 20,
        n_restarts: int = 5,
        out_dir: str | None = None,
        space: ParamSpace = BASELINE) -> dict:
    """Execute the full HGF pipeline and return a results dict.

    ``space`` selects which parameters are free (default: the 3-parameter
    baseline). Freeing extra parameters (e.g. omega3, kappa) routes outputs to a
    separate directory by default so the baseline results are preserved.
    """

    if out_dir is None:
        out_dir = (RESULTS_HGF if space is BASELINE
                   else os.path.join(os.path.dirname(RESULTS_HGF), "hgf_extended"))
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"  Free parameters: {list(space.free)}  (k={space.k})")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Stage 1: Loading session data ===")
    if sessions is None:
        sessions = available_sessions()
    print(f"  Sessions: {sessions}")

    session_data = [load_session(sid) for sid in sessions]
    models = [SessionModel(sd) for sd in session_data]
    schedules = [sd.p_schedule for sd in session_data]
    n_total_trials = sum(sd.n_trials for sd in session_data)
    print(f"  Total responding trials: {n_total_trials}")

    # ------------------------------------------------------------------
    # 2. Shared MAP fit
    # ------------------------------------------------------------------
    print("\n=== Stage 2: Shared MAP fit (complete pooling) ===")
    t0 = time.time()
    shared = shared_map_fit(models, space=space, n_restarts=n_restarts, seed=SEED)
    pstr = "  ".join(f"{n}={shared.natural[n]:.3f}" for n in space.free)
    print(f"  Done in {time.time()-t0:.1f}s  |  {pstr}")
    print(f"  loglik={shared.loglik:.1f}  BIC={shared.as_row()['bic']:.1f}  "
          f"success={shared.success}")

    # ------------------------------------------------------------------
    # 3. Per-session (separate) MAP fits
    # ------------------------------------------------------------------
    print("\n=== Stage 3: Per-session MAP fits ===")
    t0 = time.time()
    sep_fits = fit_all_separate(models, space=space, n_restarts=n_restarts, seed=SEED)
    print(f"  Done in {time.time()-t0:.1f}s")
    for f in sep_fits:
        sid = f.per_session[list(f.per_session.keys())[0]]["session_id"]
        nat = f.natural
        pstr = "  ".join(f"{n}={nat[n]:.3f}" for n in space.free)
        print(f"  {sid}: {pstr}  "
              f"bal_acc={list(f.per_session.values())[0]['balanced_accuracy']:.3f}")

    # ------------------------------------------------------------------
    # 4. Latent trajectory CSVs
    # ------------------------------------------------------------------
    print("\n=== Stage 4: Saving latent trajectory CSVs ===")
    traj_dfs = {}
    for m in models:
        traj = session_trajectory(m, shared.natural)
        traj_dfs[m.session_id] = traj
        traj_path = os.path.join(out_dir, f"trajectory_{m.session_id}.csv")
        traj.to_csv(traj_path, index=False)
        print(f"  Saved {os.path.basename(traj_path)}")

    # ------------------------------------------------------------------
    # 5. Per-session figures
    # ------------------------------------------------------------------
    print("\n=== Stage 5: Per-session plots ===")
    for sid, traj in traj_dfs.items():
        plots.plot_trajectory(traj, save_dir=plots_dir)
        plots.plot_learning_rate(traj, save_dir=plots_dir)
        plots.plot_volatility(traj, save_dir=plots_dir)
        import matplotlib.pyplot as plt
        plt.close("all")
    print(f"  Saved {3 * len(traj_dfs)} figures to {plots_dir}")

    # ------------------------------------------------------------------
    # 6. Model comparison
    # ------------------------------------------------------------------
    print("\n=== Stage 6: Model comparison (HGF vs RW vs RW+stick) ===")
    t0 = time.time()
    rw = fit_rw(models, sticky=False, n_restarts=n_restarts, seed=SEED)
    rw_stick = fit_rw(models, sticky=True, n_restarts=n_restarts, seed=SEED)
    comp_df = compare_models(shared.loglik, hgf_k=space.k, n_trials=n_total_trials,
                              rw=rw, rw_stick=rw_stick)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(comp_df[["model", "k", "loglik", "bic", "delta_bic"]].to_string(index=False))
    comp_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    plots.plot_model_comparison(comp_df, save_dir=plots_dir)
    import matplotlib.pyplot as plt
    plt.close("all")

    # ------------------------------------------------------------------
    # 7. Parameter recovery
    # ------------------------------------------------------------------
    print(f"\n=== Stage 7: Parameter recovery (n={n_recovery}) ===")
    t0 = time.time()
    rec_df = parameter_recovery(schedules, space=space, n_sims=n_recovery,
                                seed=SEED, n_restarts=4)
    rec_sum = recovery_summary(rec_df, space=space)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(rec_sum.to_string(index=False))
    rec_df.to_csv(os.path.join(out_dir, "parameter_recovery.csv"), index=False)
    rec_sum.to_csv(os.path.join(out_dir, "parameter_recovery_summary.csv"), index=False)
    plots.plot_recovery(rec_df, save_dir=plots_dir)
    plt.close("all")

    # ------------------------------------------------------------------
    # 8. Power check
    # ------------------------------------------------------------------
    print(f"\n=== Stage 8: Power check (n={n_power} pairs/condition) ===")
    t0 = time.time()
    base = shared.natural.copy()
    _shift_grid = {"omega2": [0.5, 1.0, 1.5, 2.0], "omega3": [0.5, 1.0, 1.5],
                   "kappa": [0.3, 0.6, 1.0], "beta": [0.3, 0.6, 1.0]}
    shift_sizes = {n: _shift_grid[n] for n in space.free if n in _shift_grid}
    pow_df = power_check(schedules, base_params=base, shift_sizes=shift_sizes,
                         n_sims=n_power, n_restarts=4, seed=SEED, space=space)
    pow_sum = power_summary(pow_df)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(pow_sum.to_string(index=False))
    pow_df.to_csv(os.path.join(out_dir, "power_check.csv"), index=False)
    pow_sum.to_csv(os.path.join(out_dir, "power_check_summary.csv"), index=False)
    plots.plot_power_curve(pow_sum, save_dir=plots_dir)
    plt.close("all")

    # ------------------------------------------------------------------
    # 9. Hierarchical model
    # ------------------------------------------------------------------
    hier_result = None
    if run_hierarchical:
        print("\n=== Stage 9: Hierarchical model (NUTS) ===")
        try:
            from analysis.hgf.hierarchical import run_hierarchical as _run_hier, samples_to_natural_df
            t0 = time.time()
            hier_result = _run_hier(models, n_samples=1000, n_warmup=1000,
                                    n_chains=2, seed=SEED)
            print(f"  Done in {time.time()-t0:.1f}s")
            print(hier_result["summary"].to_string(index=False))
            hier_result["summary"].to_csv(
                os.path.join(out_dir, "hierarchical_summary.csv"), index=False)
            post_df = samples_to_natural_df(hier_result["samples"], sessions)
            post_df.to_csv(os.path.join(out_dir, "hierarchical_samples.csv"), index=False)
            plots.plot_posterior_overview(post_df, sessions, save_dir=plots_dir)
            plt.close("all")
        except Exception as exc:
            print(f"  WARNING: hierarchical fit failed ({exc}); skipping.")
    else:
        print("\n=== Stage 9: Hierarchical model SKIPPED (--no-hierarchical) ===")

    # ------------------------------------------------------------------
    # 10. Cross-session summary plots
    # ------------------------------------------------------------------
    print("\n=== Stage 10: Cross-session summary plots ===")
    per_session_naturals = [f.natural for f in sep_fits]
    plots.plot_parameter_drift(sessions, per_session_naturals,
                               shared_fit=shared.natural, save_dir=plots_dir,
                               params=list(space.free))
    plt.close("all")

    # ------------------------------------------------------------------
    # 11. fitted_parameters.csv
    # ------------------------------------------------------------------
    print("\n=== Stage 11: Writing fitted_parameters.csv ===")
    param_rows = []
    # Shared fit
    r = shared.as_row()
    r["fit_type"] = "shared"
    r["session_id"] = "all"
    param_rows.append(r)
    # Per-session fits
    for f in sep_fits:
        sid = list(f.per_session.keys())[0]
        r = f.as_row()
        r["fit_type"] = "separate"
        r["session_id"] = sid
        r.update(f.per_session[sid])
        param_rows.append(r)
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(os.path.join(out_dir, "fitted_parameters.csv"), index=False)
    print(f"  Saved {os.path.join(out_dir, 'fitted_parameters.csv')}")

    print("\n=== Pipeline complete ===")
    print(f"  All outputs in: {out_dir}")

    return {
        "shared_fit": shared,
        "sep_fits": sep_fits,
        "traj_dfs": traj_dfs,
        "comp_df": comp_df,
        "rec_df": rec_df,
        "rec_summary": rec_sum,
        "pow_df": pow_df,
        "pow_summary": pow_sum,
        "hier_result": hier_result,
        "param_df": param_df,
    }


if __name__ == "__main__":
    args = parse_args()
    run(
        sessions=args.sessions,
        run_hierarchical=not args.no_hierarchical,
        n_recovery=args.n_recovery,
        n_power=args.n_power,
        n_restarts=args.n_restarts,
        out_dir=args.out,
        space=make_param_space(args.free),
    )
