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

from .config import SEED, BASELINE, NATURAL_DEFAULTS, ParamSpace
from .data import SessionData
from .model import SessionModel, simulate_session
from .fit import separate_map_fit


# Plausible generative ranges (centred on the fitted region, within bounds).
_TRUE_RANGES = {
    "omega2": (-3.5, -0.8),
    "omega3": (-5.5, -2.5),
    "kappa": (0.5, 2.0),
    "beta": (0.5, 3.0),
    "bias": (-2.0, 0.0),
}


def draw_true_params(rng: np.random.Generator, space: ParamSpace = BASELINE) -> dict:
    """Sample a plausible generative parameter set (full natural dict).

    Free parameters are drawn from :data:`_TRUE_RANGES`; held parameters take
    their fixed value, so :func:`~analysis.hgf.model.simulate_session` always
    receives all of (omega2, omega3, kappa, beta, bias).
    """
    nat = dict(NATURAL_DEFAULTS)
    nat.update(space.fixed)
    for name in space.free:
        nat[name] = float(rng.uniform(*_TRUE_RANGES[name]))
    return nat


def simulate_and_fit(true: dict, p_schedule: np.ndarray, seed: int,
                     space: ParamSpace = BASELINE, n_restarts: int = 4) -> dict:
    """Simulate one session at ``true`` params and refit; return recovered params."""
    sd = simulate_session(omega2=true["omega2"], beta=true["beta"], bias=true["bias"],
                          p_schedule=p_schedule, seed=seed,
                          omega3=true["omega3"], kappa=true["kappa"])
    model = SessionModel(sd)
    fit = separate_map_fit(model, space=space, n_restarts=n_restarts, seed=seed)
    rec = fit.natural
    row = {
        "seed": seed,
        "n_trials": sd.n_trials,
        "gamble_frac": float(np.mean(sd.y)),
    }
    for name in space.free:
        row[f"true_{name}"] = true[name]
        row[f"rec_{name}"] = rec[name]
    return row


def parameter_recovery(schedules: list[np.ndarray], space: ParamSpace = BASELINE,
                       n_sims: int = 24, seed: int = SEED,
                       n_restarts: int = 4) -> pd.DataFrame:
    """Run ``n_sims`` simulate->refit draws, cycling through real schedules."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        true = draw_true_params(rng, space)
        sched = schedules[i % len(schedules)]
        rows.append(simulate_and_fit(true, sched, seed=int(rng.integers(1, 2**31)),
                                     space=space, n_restarts=n_restarts))
    return pd.DataFrame(rows)


def recovery_summary(df: pd.DataFrame, space: ParamSpace = BASELINE) -> pd.DataFrame:
    """Correlation + error stats of recovered vs true, per free parameter."""
    rows = []
    for p in space.free:
        t = df[f"true_{p}"].to_numpy()
        r = df[f"rec_{p}"].to_numpy()
        ok = np.isfinite(t) & np.isfinite(r)
        corr = float(np.corrcoef(t[ok], r[ok])[0, 1]) if ok.sum() > 2 else np.nan
        rows.append({
            "parameter": p,
            "pearson_r": corr,
            "rmse": float(np.sqrt(np.mean((t[ok] - r[ok]) ** 2))) if ok.sum() else np.nan,
            "bias_mean_err": float(np.mean(r[ok] - t[ok])) if ok.sum() else np.nan,
            "n": int(ok.sum()),
        })
    return pd.DataFrame(rows)
