"""Latent trajectory tables — one tidy row per trial.

Given a fitted parameter set, run the HGF filter for a session and assemble the
belief trajectory the project asks for. Column meanings (all per trial):

================== =========================================================
column             meaning
================== =========================================================
trial              0-based trial index within the session (responding trials)
perceived_gamble_prob  s(mu_hat_2): the predicted P(gamble pays big reward).
                   pyhgf binary node 0 expected mean. THE headline latent var.
mu2                posterior mean of HGF level 2 (the tendency, logit space)
mu2_hat            predicted mean of level 2 (before the trial's update)
sa2                predicted variance of level 2, 1/pi_hat_2 (belief uncertainty)
mu3                posterior mean of HGF level 3 (log-volatility)
learning_rate      1/pi_2 (inverse POSTERIOR precision at level 2), per spec
delta1             level-1 prediction error: mu_1 - mu_hat_1 (outcome - p_hat)
delta2             level-2 prediction error: mu_2 - mu_hat_2
p_choose_gamble    response model's P(choose gamble) for this trial
actual_choice      participant's choice (1 = gamble, 0 = safe)
gamble_observed    1 if the gamble outcome was observed (gamble chosen)
gamble_outcome     the observed gamble big-reward outcome (NaN if unobserved)
true_p_schedule    the task's scheduled P(big reward) for the gamble arm
session_id         session label
================== =========================================================

``sa2`` (predicted variance) and ``learning_rate`` (1/posterior-precision) are
distinct quantities by construction; both are reported, as in the spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .model import SessionModel, p_choose_gamble


def session_trajectory(model: SessionModel, natural: dict) -> pd.DataFrame:
    """Return the tidy per-trial latent trajectory for one session."""
    df = model.trajectories(natural["omega2"])
    sd = model.sd

    p_hat = df["x_0_expected_mean"].to_numpy()
    p_g = np.asarray(p_choose_gamble(jnp.asarray(p_hat), natural["beta"], natural["bias"]))

    gamble_outcome = np.where(sd.observed == 1, sd.u, np.nan)

    out = pd.DataFrame({
        "session_id": sd.session_id,
        "trial": np.arange(sd.n_trials),
        "perceived_gamble_prob": p_hat,
        "mu2": df["x_1_mean"].to_numpy(),
        "mu2_hat": df["x_1_expected_mean"].to_numpy(),
        "sa2": 1.0 / df["x_1_expected_precision"].to_numpy(),
        "mu3": df["x_2_mean"].to_numpy(),
        "learning_rate": 1.0 / df["x_1_precision"].to_numpy(),
        "delta1": df["x_0_mean"].to_numpy() - df["x_0_expected_mean"].to_numpy(),
        "delta2": df["x_1_mean"].to_numpy() - df["x_1_expected_mean"].to_numpy(),
        "p_choose_gamble": p_g,
        "actual_choice": sd.y,
        "gamble_observed": sd.observed,
        "gamble_outcome": gamble_outcome,
        "true_p_schedule": sd.p_schedule,
        "original_trial_index": sd.trial_index,
    })
    return out


def all_trajectories(models: list[SessionModel], natural: dict) -> dict[str, pd.DataFrame]:
    """Per-session trajectory tables under one shared parameter set."""
    return {m.session_id: session_trajectory(m, natural) for m in models}
