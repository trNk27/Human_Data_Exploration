"""MAP fitting of the HGF response model.

Two estimators, both maximising the (log-)posterior = sum of per-session choice
log-likelihoods + Gaussian log-priors, over the unconstrained vector
``theta = [omega2, log_beta, bias]`` (``beta = exp(log_beta)``):

  * :func:`shared_map_fit`   — COMPLETE POOLING. One parameter set for the whole
    participant, fit across ALL sessions jointly by running a SEPARATE filter
    pass per session that shares the same parameters and SUMMING the surprise.
    This is the project's default and most reliable fit.
  * :func:`separate_map_fit` — NO POOLING. One independent MAP fit per session;
    used by the power check and for visualising cross-session drift.

Optimisation uses L-BFGS-B with exact JAX gradients and several random restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from .config import (
    PARAM_SPECS, NATURAL_DEFAULTS, BASELINE, ParamSpace, SEED,
)
from .model import SessionModel, _session_logp_jit, p_choose_gamble


# ---------------------------------------------------------------------------
# JAX log-posterior pieces
# ---------------------------------------------------------------------------

def _gauss_logpdf(x, mean, sd):
    return -0.5 * jnp.log(2 * jnp.pi * sd ** 2) - 0.5 * ((x - mean) / sd) ** 2


def _make_neg_log_post(models: list[SessionModel], space: ParamSpace = BASELINE,
                       use_prior: bool = True):
    """Return a jitted value-and-grad of the negative log-posterior over theta.

    ``theta`` has one entry per free parameter of ``space`` (in ``space.free``
    order). Perceptual parameters held fixed by ``space`` enter the likelihood as
    constants, so the same single likelihood serves the baseline and extended
    models.
    """
    free = space.free
    log_space = {n: PARAM_SPECS[n].log_space for n in free}
    fixed = {**NATURAL_DEFAULTS, **space.fixed}     # python floats for held params
    prior_mean = jnp.array([PARAM_SPECS[n].prior.mean for n in free])
    prior_sd = jnp.array([PARAM_SPECS[n].prior.sd for n in free])

    def neg_log_post(theta):
        nat = dict(fixed)
        for i, n in enumerate(free):
            nat[n] = jnp.exp(theta[i]) if log_space[n] else theta[i]
        ll = 0.0
        for m in models:
            ll = ll + _session_logp_jit(
                nat["omega2"], nat["omega3"], nat["kappa"], nat["beta"], nat["bias"],
                m._hgf, m._u, m._observed, m._y,
            )
        if use_prior:
            ll = ll + jnp.sum(_gauss_logpdf(theta, prior_mean, prior_sd))
        return -ll

    return jax.jit(jax.value_and_grad(neg_log_post))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Outcome of a MAP fit (shared or per-session)."""

    label: str
    theta: np.ndarray                       # unconstrained, one entry per free param
    natural: dict                           # full {omega2, omega3, kappa, beta, bias}
    neg_log_post: float                     # at the optimum
    loglik: float                           # choice log-likelihood (no prior)
    n_trials: int
    n_sessions: int
    k: int = 3                              # number of FREE parameters (for BIC)
    per_session: dict = field(default_factory=dict)  # session_id -> metrics
    success: bool = True

    def as_row(self) -> dict:
        row = {
            "fit": self.label,
            "omega2": self.natural["omega2"],
            "omega3": self.natural["omega3"],
            "kappa": self.natural["kappa"],
            "beta": self.natural["beta"],
            "bias": self.natural["bias"],
            "k": self.k,
            "loglik": self.loglik,
            "neg_log_post": self.neg_log_post,
            "n_trials": self.n_trials,
            "n_sessions": self.n_sessions,
            "bic": bic(self.loglik, k=self.k, n=self.n_trials),
            "success": self.success,
        }
        return row


def bic(loglik: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (lower = better)."""
    return float(k * np.log(n) - 2.0 * loglik)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

_BARRIER = 1e8   # finite penalty returned in place of NaN/inf, so line searches back off


def _optimise(neg_log_post_vg, inits: list[np.ndarray], bounds: list):
    """Robust bounded minimisation of the negative log-posterior.

    Primary search is gradient-free, bounded Nelder-Mead from several restarts —
    the response model's gradient w.r.t. log-beta is large near the prior mean and
    a gradient step there overshoots into the degenerate region where the HGF
    returns NaN. A NaN/inf value is mapped to a large finite barrier so the
    simplex simply rejects those points. A short L-BFGS-B polish then refines the
    best simplex optimum, where the gradient is well-behaved.
    """
    def val(theta):
        v = float(neg_log_post_vg(jnp.asarray(theta, dtype=float))[0])
        return v if np.isfinite(v) else _BARRIER

    def val_grad(theta):
        v, g = neg_log_post_vg(jnp.asarray(theta, dtype=float))
        v = float(v)
        g = np.asarray(g, dtype=float)
        if not np.isfinite(v):
            return _BARRIER, np.zeros_like(g)
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
        return v, g

    best = None
    for x0 in inits:
        res = minimize(val, np.asarray(x0, float), method="Nelder-Mead",
                       bounds=bounds,
                       options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res

    converged = bool(best.success)
    polish = minimize(val_grad, np.asarray(best.x, float), jac=True, method="L-BFGS-B",
                      bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    if np.isfinite(polish.fun) and polish.fun <= best.fun:
        polish.success = converged or bool(polish.success)
        return polish
    best.success = converged
    return best


def _restart_inits(space: ParamSpace, n: int = 5, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    init = space.init()
    inits = [init.copy()]
    lo = np.array([b[0] for b in space.bounds()])
    hi = np.array([b[1] for b in space.bounds()])
    jit = space.jitter()
    for _ in range(n - 1):
        inits.append(np.clip(init + rng.normal(0, jit), lo, hi))
    return inits


def _choice_metrics(model: SessionModel, natural: dict) -> dict:
    """Per-session predictive metrics at the fitted parameters."""
    df = model.trajectories(natural["omega2"], natural["omega3"], natural["kappa"])
    p_hat = df["x_0_expected_mean"].to_numpy()
    p_g = np.asarray(p_choose_gamble(jnp.asarray(p_hat), natural["beta"], natural["bias"]))
    y = model.sd.y
    pred = (p_g >= 0.5).astype(float)
    acc = float(np.mean(pred == y))
    # balanced accuracy (guards against the safe-majority baseline)
    tpr = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else np.nan
    tnr = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else np.nan
    bal_acc = float(np.nanmean([tpr, tnr]))
    ll = float(model.logp(natural["omega2"], natural["omega3"], natural["kappa"],
                          natural["beta"], natural["bias"]))
    # likelihood per trial and chance log-likelihood (base rate)
    base = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
    ll_chance = float(np.sum(y * np.log(base) + (1 - y) * np.log(1 - base)))
    pseudo_r2 = 1.0 - ll / ll_chance if ll_chance != 0 else np.nan
    return {
        "session_id": model.session_id,
        "n_trials": model.sd.n_trials,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "loglik": ll,
        "loglik_per_trial": ll / model.sd.n_trials,
        "pseudo_r2": pseudo_r2,
    }


def shared_map_fit(
    models: list[SessionModel], space: ParamSpace = BASELINE,
    n_restarts: int = 5, seed: int = SEED,
) -> FitResult:
    """Complete-pooling MAP fit: one parameter set (per ``space``) across all sessions."""
    vg = _make_neg_log_post(models, space, use_prior=True)
    res = _optimise(vg, _restart_inits(space, n_restarts, seed), space.bounds())
    theta = np.asarray(res.x, float)
    natural = space.to_natural(theta)

    vg_noprior = _make_neg_log_post(models, space, use_prior=False)
    loglik = float(-vg_noprior(jnp.asarray(theta))[0])
    n_trials = int(sum(m.sd.n_trials for m in models))

    per_session = {m.session_id: _choice_metrics(m, natural) for m in models}
    return FitResult(
        label="shared",
        theta=theta,
        natural=natural,
        neg_log_post=float(res.fun),
        loglik=loglik,
        n_trials=n_trials,
        n_sessions=len(models),
        k=space.k,
        per_session=per_session,
        success=bool(res.success),
    )


def separate_map_fit(
    model: SessionModel, space: ParamSpace = BASELINE,
    n_restarts: int = 5, seed: int = SEED,
) -> FitResult:
    """No-pooling MAP fit for a single session."""
    vg = _make_neg_log_post([model], space, use_prior=True)
    res = _optimise(vg, _restart_inits(space, n_restarts, seed), space.bounds())
    theta = np.asarray(res.x, float)
    natural = space.to_natural(theta)

    loglik = float(model.logp(natural["omega2"], natural["omega3"], natural["kappa"],
                             natural["beta"], natural["bias"]))
    return FitResult(
        label=f"separate:{model.session_id}",
        theta=theta,
        natural=natural,
        neg_log_post=float(res.fun),
        loglik=loglik,
        n_trials=model.sd.n_trials,
        n_sessions=1,
        k=space.k,
        per_session={model.session_id: _choice_metrics(model, natural)},
        success=bool(res.success),
    )


def fit_all_separate(
    models: list[SessionModel], space: ParamSpace = BASELINE,
    n_restarts: int = 5, seed: int = SEED,
) -> list[FitResult]:
    return [separate_map_fit(m, space=space, n_restarts=n_restarts, seed=seed) for m in models]
