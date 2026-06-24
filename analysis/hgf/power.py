"""Power check: can this design detect a session-to-session parameter shift?

Given the actual per-session trial counts, simulate pairs of sessions with a
known shift in omega2 or beta, refit each session independently (no-pooling
MAP), and measure how often the recovered difference has the right sign and
exceeds a meaningful threshold.

This answers the question: "If the participant's ω₂ really changed between two
adjacent sessions, would our fitting procedure notice?"

The check is done for shifts in omega2 and beta separately, across a range of
shift magnitudes, using the real session schedules for realism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED, BASELINE, NATURAL_DEFAULTS, ParamSpace
from .model import simulate_session
from .fit import separate_map_fit
from .data import SessionData
from .model import SessionModel


def _sim_pair(base: dict, shift_param: str, shift_size: float,
              sched_a: np.ndarray, sched_b: np.ndarray,
              seed_a: int, seed_b: int, space: ParamSpace = BASELINE,
              n_restarts: int = 4) -> dict:
    """Simulate two sessions — base params and base+shift — then refit each."""
    params_a = dict(base)
    params_b = dict(base)
    params_b[shift_param] = base[shift_param] + shift_size

    sd_a = simulate_session(omega2=params_a["omega2"], beta=params_a["beta"], bias=params_a["bias"],
                            p_schedule=sched_a, seed=seed_a,
                            omega3=params_a["omega3"], kappa=params_a["kappa"])
    sd_b = simulate_session(omega2=params_b["omega2"], beta=params_b["beta"], bias=params_b["bias"],
                            p_schedule=sched_b, seed=seed_b,
                            omega3=params_b["omega3"], kappa=params_b["kappa"])

    fit_a = separate_map_fit(SessionModel(sd_a), space=space, n_restarts=n_restarts, seed=seed_a)
    fit_b = separate_map_fit(SessionModel(sd_b), space=space, n_restarts=n_restarts, seed=seed_b)

    rec_a = fit_a.natural
    rec_b = fit_b.natural
    return {
        "shift_param": shift_param,
        "shift_size": shift_size,
        "true_delta": shift_size,
        "rec_delta": rec_b[shift_param] - rec_a[shift_param],
        "true_a": params_a[shift_param], "rec_a": rec_a[shift_param],
        "true_b": params_b[shift_param], "rec_b": rec_b[shift_param],
        "n_a": sd_a.n_trials, "n_b": sd_b.n_trials,
        "gamble_frac_a": float(np.mean(sd_a.y)),
        "gamble_frac_b": float(np.mean(sd_b.y)),
    }


def power_check(schedules: list[np.ndarray],
                base_params: dict | None = None,
                shift_sizes: dict[str, list[float]] | None = None,
                n_sims: int = 20,
                n_restarts: int = 4,
                seed: int = SEED,
                space: ParamSpace = BASELINE) -> pd.DataFrame:
    """Simulate n_sims session pairs at each shift size and measure recovery.

    Parameters
    ----------
    schedules:
        Real session schedules (p_schedule arrays). Pairs are drawn from these
        to capture the actual distribution of trial counts and difficulty.
    base_params:
        True generative parameters for the 'before' session; defaults to the
        shared-fit point estimate.
    shift_sizes:
        Dict mapping param name -> list of shift magnitudes to test.
        Defaults to omega2: [0.5, 1.0, 1.5, 2.0] and beta: [0.3, 0.6, 1.0].
    n_sims:
        Number of simulated pairs per (param, shift) combination.
    """
    if base_params is None:
        base_params = {**NATURAL_DEFAULTS, "omega2": -1.7, "beta": 1.3, "bias": -1.2}
    else:
        base_params = {**NATURAL_DEFAULTS, **base_params}   # ensure all 5 keys present
    if shift_sizes is None:
        shift_sizes = {
            "omega2": [0.5, 1.0, 1.5, 2.0],
            "beta":   [0.3, 0.6, 1.0],
        }

    rng = np.random.default_rng(seed)
    rows = []
    n_sched = len(schedules)

    for param, magnitudes in shift_sizes.items():
        for mag in magnitudes:
            for _ in range(n_sims):
                i_a = int(rng.integers(0, n_sched))
                i_b = int(rng.integers(0, n_sched))
                seed_a = int(rng.integers(1, 2**31))
                seed_b = int(rng.integers(1, 2**31))
                row = _sim_pair(base_params, param, mag,
                                schedules[i_a], schedules[i_b],
                                seed_a, seed_b, space=space, n_restarts=n_restarts)
                rows.append(row)

    return pd.DataFrame(rows)


def power_summary(df: pd.DataFrame,
                  sign_threshold: float = 0.0) -> pd.DataFrame:
    """Summarise detection power per (param, shift_size).

    'Detected' = recovered delta has the correct sign AND abs(rec_delta) > sign_threshold.

    Returns a DataFrame with columns:
        shift_param, shift_size, n_sims, power, mean_rec_delta, rmse_delta
    """
    rows = []
    for (param, size), g in df.groupby(["shift_param", "shift_size"]):
        ok = np.isfinite(g["rec_delta"])
        g = g[ok]
        if len(g) == 0:
            continue
        detected = (g["rec_delta"] * np.sign(size) > sign_threshold).mean()
        rows.append({
            "shift_param": param,
            "shift_size": float(size),
            "n_sims": int(ok.sum()),
            "power": float(detected),
            "mean_rec_delta": float(g["rec_delta"].mean()),
            "rmse_delta": float(np.sqrt(((g["rec_delta"] - g["true_delta"]) ** 2).mean())),
        })
    return pd.DataFrame(rows)
