"""Export per-trial Rescorla-Wagner (+ stickiness) prediction-error trajectories.

Fits the **RW + stickiness** model by shared (complete-pooling) MAP across all
sessions — the model that best explains choices in this dataset (lowest BIC; see
``results/hgf/model_comparison.csv``) — then writes one tidy trajectory CSV per
session to ``results/rw/``. The headline column is ``rw_pe`` = δ = outcome − Q,
the reward prediction error used as a per-trial neural regressor by
``viewers/firing_rate_vs_rw_pe.py`` / ``Session(...).neuron(i).fr_vs_rw_pe()``.

The fit reuses ``analysis.hgf.comparison.fit_rw`` (same RW engine and response
rule as the HGF model comparison), so the ``loglik`` written to
``results/rw/fitted_parameters.csv`` matches the ``RW + stickiness`` row of
``model_comparison.csv``. This is a fast, standalone alternative to the full HGF
pipeline (no recovery / power / NUTS stages).

Usage
-----
    python -m analysis.rw_trajectories [--sessions 20250714 ...]
    python analysis/rw_trajectories.py
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Repo root importable when run via ``python -m analysis.rw_trajectories``
# or ``python analysis/rw_trajectories.py``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.hgf.config import SEED
from analysis.hgf.data import load_session, list_sessions as available_sessions
from analysis.hgf.model import SessionModel
from analysis.hgf.comparison import fit_rw, rw_session_trajectory
from utils import RESULTS_DIR


def run(sessions: list[str] | None = None, n_restarts: int = 6,
        out_dir: str | None = None) -> dict:
    """Fit RW+stickiness (shared MAP) and write per-session PE trajectory CSVs."""
    out_dir = out_dir or os.path.join(RESULTS_DIR, "rw")
    os.makedirs(out_dir, exist_ok=True)

    if sessions is None:
        sessions = available_sessions()
    print(f"Sessions: {sessions}")

    models = [SessionModel(load_session(sid)) for sid in sessions]

    print("Fitting RW + stickiness (shared MAP, complete pooling)...")
    fit = fit_rw(models, sticky=True, n_restarts=n_restarts, seed=SEED)
    nat = fit.natural
    print(f"  alpha={nat['alpha']:.3f}  beta={nat['beta']:.3f}  "
          f"bias={nat['bias']:.3f}  phi={nat['phi']:.3f}  loglik={fit.loglik:.1f}")

    for m in models:
        traj = rw_session_trajectory(m, nat)
        path = os.path.join(out_dir, f"trajectory_{m.session_id}.csv")
        traj.to_csv(path, index=False)
        print(f"  Saved {os.path.basename(path)}")

    params_row = {**nat, "loglik": fit.loglik, "k": fit.k,
                  "n_trials": fit.n_trials, "model": fit.label}
    params_path = os.path.join(out_dir, "fitted_parameters.csv")
    pd.DataFrame([params_row]).to_csv(params_path, index=False)
    print(f"  Saved {os.path.basename(params_path)}")
    print(f"\nDone. Outputs in: {out_dir}")

    return {"fit": fit, "sessions": sessions, "out_dir": out_dir}


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export RW + stickiness prediction-error (delta) trajectories")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Session IDs to include (default: all available)")
    p.add_argument("--n-restarts", type=int, default=6,
                   help="MAP optimisation restarts (default: 6)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: results/rw/)")
    args = p.parse_args()
    run(sessions=args.sessions, n_restarts=args.n_restarts, out_dir=args.out)
