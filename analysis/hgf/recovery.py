"""Parameter recovery: simulate with known parameters, refit, compare.

Validates that the fitting procedure can recover the generative parameters on
THIS participant's actual trial schedules. For each draw we simulate one session
of an HGF agent with known (omega2, beta, bias) on a real session's schedule,
refit by single-session MAP, and record recovered vs. true.

Good recovery (high true-vs-recovered correlation, points near the identity
line) is the precondition for trusting the fitted parameters and for freeing
more parameters later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED, THETA_BOUNDS
from .data import SessionData
from .model import SessionModel, simulate_session
from .fit import separate_map_fit


# Plausible generative ranges (centred on the fitted region, within bounds).
_TRUE_RANGES = {
    "omega2": (-3.5, -0.8),
    "beta": (0.5, 3.0),
    "bias": (-2.0, 0.0),
}


def draw_true_params(rng: np.random.Generator) -> dict:
    """Sample a plausible generative parameter set."""
    return {
        "omega2": float(rng.uniform(*_TRUE_RANGES["omega2"])),
        "beta": float(rng.uniform(*_TRUE_RANGES["beta"])),
        "bias": float(rng.uniform(*_TRUE_RANGES["bias"])),
    }


def simulate_and_fit(true: dict, p_schedule: np.ndarray, seed: int,
                     n_restarts: int = 4) -> dict:
    """Simulate one session at ``true`` params and refit; return recovered params."""
    sd = simulate_session(true["omega2"], true["beta"], true["bias"],
                          p_schedule=p_schedule, seed=seed)
    model = SessionModel(sd)
    fit = separate_map_fit(model, n_restarts=n_restarts, seed=seed)
    rec = fit.natural
    return {
        "seed": seed,
        "n_trials": sd.n_trials,
        "gamble_frac": float(np.mean(sd.y)),
        "true_omega2": true["omega2"], "rec_omega2": rec["omega2"],
        "true_beta": true["beta"], "rec_beta": rec["beta"],
        "true_bias": true["bias"], "rec_bias": rec["bias"],
    }


def parameter_recovery(schedules: list[np.ndarray], n_sims: int = 24,
                       seed: int = SEED, n_restarts: int = 4) -> pd.DataFrame:
    """Run ``n_sims`` simulate->refit draws, cycling through real schedules."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        true = draw_true_params(rng)
        sched = schedules[i % len(schedules)]
        rows.append(simulate_and_fit(true, sched, seed=int(rng.integers(1, 2**31)),
                                     n_restarts=n_restarts))
    return pd.DataFrame(rows)


def recovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation + error stats of recovered vs true, per parameter."""
    rows = []
    for p in ("omega2", "beta", "bias"):
        t = df[f"true_{p}"].to_numpy()
        r = df[f"rec_{p}"].to_numpy()
        ok = np.isfinite(t) & np.isfinite(r)
        corr = float(np.corrcoef(t[ok], r[ok])[0, 1]) if ok.sum() > 2 else np.nan
        rows.append({
            "parameter": p,
            "pearson_r": corr,
            "rmse": float(np.sqrt(np.mean((t[ok] - r[ok]) ** 2))),
            "bias_mean_err": float(np.mean(r[ok] - t[ok])),
            "n": int(ok.sum()),
        })
    return pd.DataFrame(rows)
