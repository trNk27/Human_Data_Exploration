"""Model comparison: HGF vs Rescorla-Wagner (with / without stickiness).

All three models share the SAME response model (EV softmax with fixed rewards),
so the comparison isolates the belief-updating mechanism:

  * HGF        — adaptive, volatility-driven learning rate (the fitted model).
  * RW         — constant learning rate alpha; tracks the gamble win probability,
                 updating only on gamble-choice trials (partial feedback).
  * RW + stick — RW plus a perseveration term phi * C_t in the choice logit.

Each model is fit by shared (complete-pooling) MAP across all sessions and
compared by total choice log-likelihood and BIC (lower BIC = better, accounting
for the parameter count). The RW engine mirrors ``analysis/choice_timeline.py``
(Q-learning + softmax + perseveration) but uses the project's EV response rule
for a like-for-like contrast with the HGF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from .config import REWARD_GAMBLE, REWARD_SAFE, SEED
from .model import SessionModel
from .fit import bic, _BARRIER


# ---------------------------------------------------------------------------
# Rescorla-Wagner forward pass (JAX, partial feedback)
# ---------------------------------------------------------------------------

def _rw_session_loglik(alpha, beta, bias, phi, u, observed, y):
    """Choice log-likelihood of one session under the RW(+stick) model."""

    def step(carry, xs):
        q, last_c = carry
        u_t, obs_t, y_t = xs
        ev_gamble = q * REWARD_GAMBLE
        logit = beta * (ev_gamble - REWARD_SAFE) + bias + phi * last_c
        p_g = 1.0 / (1.0 + jnp.exp(-logit))
        p_g = jnp.clip(p_g, 1e-6, 1 - 1e-6)
        ll_t = y_t * jnp.log(p_g) + (1 - y_t) * jnp.log(1 - p_g)
        # update the gamble win-prob estimate only when the gamble was sampled
        q_new = q + obs_t * alpha * (u_t - q)
        last_c_new = jnp.where(y_t == 1, 1.0, -1.0)
        return (q_new, last_c_new), ll_t

    (_, _), ll = jax.lax.scan(step, (0.5, 0.0), (u, observed.astype(float), y))
    return jnp.sum(ll)


def _rw_theta_to_natural(theta, sticky: bool) -> dict:
    alpha = float(1.0 / (1.0 + np.exp(-theta[0])))
    beta = float(np.exp(theta[1]))
    bias = float(theta[2])
    phi = float(theta[3]) if sticky else 0.0
    return {"alpha": alpha, "beta": beta, "bias": bias, "phi": phi}


# Priors (theta space): logit_alpha ~ N(0,1.5), log_beta ~ N(log2,1),
# bias ~ N(0,2), phi ~ N(0,1).
_RW_PRIOR_MEAN = np.array([0.0, np.log(2.0), 0.0, 0.0])
_RW_PRIOR_SD = np.array([1.5, 1.0, 2.0, 1.0])
_RW_BOUNDS = [(-8.0, 8.0), (float(np.log(0.05)), float(np.log(30.0))), (-10.0, 10.0), (-5.0, 5.0)]


def _make_rw_neg_log_post(models: list[SessionModel], sticky: bool):
    k = 4 if sticky else 3

    def neg_log_post(theta):
        alpha = jax.nn.sigmoid(theta[0])
        beta = jnp.exp(theta[1])
        bias = theta[2]
        phi = theta[3] if sticky else 0.0
        ll = 0.0
        for m in models:
            ll = ll + _rw_session_loglik(alpha, beta, bias, phi, m._u, m._observed, m._y)
        # Gaussian priors over the active theta components
        lp = 0.0
        for i in range(k):
            lp = lp - 0.5 * ((theta[i] - _RW_PRIOR_MEAN[i]) / _RW_PRIOR_SD[i]) ** 2
        return -(ll + lp)

    vg = jax.jit(jax.value_and_grad(neg_log_post))
    return vg, k


@dataclass
class RWFitResult:
    label: str
    natural: dict
    loglik: float
    n_trials: int
    k: int
    per_session: dict


def _rw_loglik_total(models, theta, sticky):
    alpha = 1.0 / (1.0 + np.exp(-theta[0]))
    beta = np.exp(theta[1])
    bias = theta[2]
    phi = theta[3] if sticky else 0.0
    per = {}
    total = 0.0
    for m in models:
        ll = float(_rw_session_loglik(alpha, beta, bias, phi, m._u, m._observed, m._y))
        per[m.session_id] = {"session_id": m.session_id, "loglik": ll,
                             "n_trials": m.sd.n_trials, "loglik_per_trial": ll / m.sd.n_trials}
        total += ll
    return total, per


def fit_rw(models: list[SessionModel], sticky: bool = False,
           n_restarts: int = 6, seed: int = SEED) -> RWFitResult:
    """Shared-parameter MAP fit of the RW (or RW+stickiness) model."""
    vg, k = _make_rw_neg_log_post(models, sticky)
    bounds = _RW_BOUNDS[:k]

    def val(theta_k):
        theta = np.zeros(4)
        theta[:k] = theta_k
        v = float(vg(jnp.asarray(theta))[0])
        return v if np.isfinite(v) else _BARRIER

    rng = np.random.default_rng(seed)
    inits = [_RW_PRIOR_MEAN[:k].copy()]
    for _ in range(n_restarts - 1):
        inits.append(_RW_PRIOR_MEAN[:k] + rng.normal(0, _RW_PRIOR_SD[:k]))

    best = None
    for x0 in inits:
        res = minimize(val, np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]),
                       method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res

    theta = np.zeros(4)
    theta[:k] = best.x
    natural = _rw_theta_to_natural(theta, sticky)
    total_ll, per = _rw_loglik_total(models, theta, sticky)
    n_trials = int(sum(m.sd.n_trials for m in models))
    label = "rw_stick" if sticky else "rw"
    return RWFitResult(label=label, natural=natural, loglik=total_ll,
                       n_trials=n_trials, k=k, per_session=per)


# ---------------------------------------------------------------------------
# Per-trial trajectory (the RW analogue of analysis.hgf.trajectories)
# ---------------------------------------------------------------------------

def rw_session_trajectory(model: SessionModel, natural: dict) -> pd.DataFrame:
    """Per-trial RW(+stickiness) latent trajectory for one session.

    Replays the same forward pass as :func:`_rw_session_loglik` (in numpy),
    recording the PRE-update gamble value ``Q`` and the level-1 reward prediction
    error ``δ = outcome − Q`` for every trial. ``δ`` is a genuine reward
    prediction error only when the gamble outcome is observed; on safe trials it
    collapses to ``−Q`` (downstream viewers mask it out), exactly as the HGF
    δ₁ does. The schema parallels :func:`analysis.hgf.trajectories.session_trajectory`
    so the trajectory loader (``utils.load_hgf_trajectory_column``) works unchanged.

    ``natural`` is the fitted RW parameter dict ``{alpha, beta, bias, phi}`` (use
    the shared / complete-pooling fit, mirroring how the HGF trajectory uses the
    shared parameters).
    """
    sd = model.sd
    alpha = float(natural["alpha"])
    beta = float(natural["beta"])
    bias = float(natural["bias"])
    phi = float(natural.get("phi", 0.0))

    u = np.asarray(sd.u, dtype=float)
    observed = np.asarray(sd.observed, dtype=float)
    y = np.asarray(sd.y, dtype=float)
    n = int(sd.n_trials)

    value = np.empty(n)   # pre-update Q (gamble win-prob estimate — RW analogue of p̂)
    pe = np.empty(n)      # δ = outcome − Q
    p_g = np.empty(n)

    q = 0.5
    last_c = 0.0
    for t in range(n):
        value[t] = q
        ev_gamble = q * REWARD_GAMBLE
        logit = beta * (ev_gamble - REWARD_SAFE) + bias + phi * last_c
        p_g[t] = 1.0 / (1.0 + np.exp(-logit))
        pe[t] = u[t] - q
        # update the gamble win-prob estimate only when the gamble was sampled
        q = q + observed[t] * alpha * (u[t] - q)
        last_c = 1.0 if y[t] == 1 else -1.0

    gamble_outcome = np.where(observed == 1, u, np.nan)

    return pd.DataFrame({
        "session_id": sd.session_id,
        "trial": np.arange(n),
        "value_gamble": value,
        "rw_pe": pe,
        "p_choose_gamble": p_g,
        "actual_choice": y,
        "gamble_observed": observed.astype(int),
        "gamble_outcome": gamble_outcome,
        "true_p_schedule": sd.p_schedule,
        "original_trial_index": sd.trial_index,
    })


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_models(hgf_loglik: float, hgf_k: int, n_trials: int,
                   rw: RWFitResult, rw_stick: RWFitResult) -> pd.DataFrame:
    """Assemble a loglik / BIC comparison table (HGF vs RW vs RW+stick)."""
    rows = [
        {"model": "HGF (3-level binary)", "k": hgf_k, "loglik": hgf_loglik,
         "bic": bic(hgf_loglik, hgf_k, n_trials)},
        {"model": "Rescorla-Wagner", "k": rw.k, "loglik": rw.loglik,
         "bic": bic(rw.loglik, rw.k, n_trials)},
        {"model": "RW + stickiness", "k": rw_stick.k, "loglik": rw_stick.loglik,
         "bic": bic(rw_stick.loglik, rw_stick.k, n_trials)},
    ]
    df = pd.DataFrame(rows)
    df["loglik_per_trial"] = df["loglik"] / n_trials
    best_bic = df["bic"].min()
    df["delta_bic"] = df["bic"] - best_bic
    df = df.sort_values("bic").reset_index(drop=True)
    return df
