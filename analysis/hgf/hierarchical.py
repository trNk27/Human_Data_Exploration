"""Hierarchical partial-pooling model: sessions-within-participant.

Each of the K sessions gets its own (omega2_k, log_beta_k, bias_k) drawn
from a shared participant-level Gaussian, estimated jointly via NUTS.

Graphical model (independent priors, centred parameterisation):
    mu_omega2   ~ N(PRIORS["omega2"].mean, PRIORS["omega2"].sd)
    sigma_omega2 ~ HalfNormal(1.0)
    omega2_k    ~ N(mu_omega2, sigma_omega2)   for k=1..K

    mu_log_beta ~ N(PRIORS["log_beta"].mean, PRIORS["log_beta"].sd)
    sigma_log_beta ~ HalfNormal(0.5)
    log_beta_k  ~ N(mu_log_beta, sigma_log_beta)

    mu_bias     ~ N(PRIORS["bias"].mean, PRIORS["bias"].sd)
    sigma_bias  ~ HalfNormal(1.0)
    bias_k      ~ N(mu_bias, sigma_bias)

    y_k ~ Bernoulli(p_choose_gamble(p_hat_k, exp(log_beta_k), bias_k))

We use PyMC's JAX-NUTS back-end (numpyro or blackjax) since there is no C
compiler available for the default pytensor sampler.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from .config import PRIORS, SEED
from .model import SessionModel, _session_logp_jit

if TYPE_CHECKING:
    import pymc as pm
    import arviz as az


# ---------------------------------------------------------------------------
# JAX potential function for NUTS (bypasses pytensor entirely)
# ---------------------------------------------------------------------------

def _make_jax_logp(models: list[SessionModel]):
    """Return a JAX function log p(theta | data) for all K sessions.

    theta layout (flat):
        [mu_o, log_sigma_o, mu_lb, log_sigma_lb, mu_b, log_sigma_b,
         omega2_0..K-1, log_beta_0..K-1, bias_0..K-1]
    """
    K = len(models)
    # Hyper-prior parameters (from config PRIORS)
    hp_om = (PRIORS["omega2"].mean, PRIORS["omega2"].sd)
    hp_lb = (PRIORS["log_beta"].mean, PRIORS["log_beta"].sd)
    hp_bi = (PRIORS["bias"].mean, PRIORS["bias"].sd)

    def logp(theta):
        # unpack hyperparameters
        mu_o      = theta[0]
        sig_o     = jnp.exp(theta[1])       # log_sigma -> sigma (>0)
        mu_lb     = theta[2]
        sig_lb    = jnp.exp(theta[3])
        mu_b      = theta[4]
        sig_b     = jnp.exp(theta[5])
        # log Jacobian for the log-sigma reparameterisation: +1 per sigma
        log_jac = theta[1] + theta[3] + theta[5]

        # per-session parameters
        omega2s  = theta[6:6+K]
        log_betas = theta[6+K:6+2*K]
        biases   = theta[6+2*K:6+3*K]

        def gauss_lp(x, mu, sd):
            return -0.5 * jnp.log(2 * jnp.pi * sd**2) - 0.5 * ((x - mu) / sd)**2

        def half_normal_lp(log_sig, sd=1.0):
            sig = jnp.exp(log_sig)
            return (-jnp.log(sd) - 0.5 * jnp.log(jnp.pi / 2)
                    - 0.5 * (sig / sd)**2 + log_sig)  # +log_sig = Jacobian

        # Hyper-priors
        lp = (gauss_lp(mu_o, hp_om[0], hp_om[1])
              + gauss_lp(mu_lb, hp_lb[0], hp_lb[1])
              + gauss_lp(mu_b, hp_bi[0], hp_bi[1])
              + half_normal_lp(theta[1], 1.0)
              + half_normal_lp(theta[3], 0.5)
              + half_normal_lp(theta[5], 1.0))

        # Session-level priors
        for k in range(K):
            lp = lp + gauss_lp(omega2s[k], mu_o, sig_o)
            lp = lp + gauss_lp(log_betas[k], mu_lb, sig_lb)
            lp = lp + gauss_lp(biases[k], mu_b, sig_b)

        # Likelihoods
        for k, m in enumerate(models):
            beta_k = jnp.exp(log_betas[k])
            lp = lp + _session_logp_jit(
                omega2s[k], beta_k, biases[k],
                m._hgf, m._u, m._observed, m._y
            )

        return lp + log_jac

    return jax.jit(logp), K


def _nuts_sample_jax(models: list[SessionModel],
                     n_samples: int = 1000,
                     n_warmup: int = 1000,
                     n_chains: int = 2,
                     seed: int = SEED):
    """Run NUTS via numpyro's JAX kernel on the hierarchical model.

    Returns a dict of posterior arrays shaped (n_chains, n_samples, ...).
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    K = len(models)

    def model_fn():
        # Hyperparameters
        mu_o   = numpyro.sample("mu_omega2",    dist.Normal(PRIORS["omega2"].mean,   PRIORS["omega2"].sd))
        sig_o  = numpyro.sample("sigma_omega2", dist.HalfNormal(1.0))
        mu_lb  = numpyro.sample("mu_log_beta",  dist.Normal(PRIORS["log_beta"].mean, PRIORS["log_beta"].sd))
        sig_lb = numpyro.sample("sigma_log_beta", dist.HalfNormal(0.5))
        mu_b   = numpyro.sample("mu_bias",      dist.Normal(PRIORS["bias"].mean,     PRIORS["bias"].sd))
        sig_b  = numpyro.sample("sigma_bias",   dist.HalfNormal(1.0))

        # Per-session parameters
        omega2s   = numpyro.sample("omega2",   dist.Normal(mu_o,  sig_o).expand([K]))
        log_betas = numpyro.sample("log_beta", dist.Normal(mu_lb, sig_lb).expand([K]))
        biases    = numpyro.sample("bias",     dist.Normal(mu_b,  sig_b).expand([K]))

        # Likelihoods (one factor per session, each is a scalar)
        for k, m in enumerate(models):
            beta_k = jnp.exp(log_betas[k])
            ll_k = _session_logp_jit(
                omega2s[k], beta_k, biases[k],
                m._hgf, m._u, m._observed, m._y
            )
            numpyro.factor(f"ll_{k}", ll_k)

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples(group_by_chain=True)


def run_hierarchical(models: list[SessionModel],
                     n_samples: int = 1000,
                     n_warmup: int = 1000,
                     n_chains: int = 2,
                     seed: int = SEED) -> dict:
    """Fit the hierarchical model and return posterior samples + summary.

    Returns
    -------
    dict with keys:
        "samples"  : raw chain dict (param -> (n_chains, n_samples, ...))
        "summary"  : pandas DataFrame — mean/sd/HDI per param
        "n_chains" : int
        "n_samples": int
    """
    samples = _nuts_sample_jax(models, n_samples=n_samples, n_warmup=n_warmup,
                                n_chains=n_chains, seed=seed)
    summary = _posterior_summary(samples, models)
    return {"samples": samples, "summary": summary,
            "n_chains": n_chains, "n_samples": n_samples}


def _posterior_summary(samples: dict, models: list[SessionModel]) -> "pd.DataFrame":
    import pandas as pd

    K = len(models)
    rows = []

    def _stats(arr):
        flat = np.asarray(arr).reshape(-1)
        lo, hi = np.percentile(flat, [2.5, 97.5])
        return {"mean": float(np.mean(flat)), "sd": float(np.std(flat)),
                "hdi_2.5": float(lo), "hdi_97.5": float(hi)}

    for name in ["mu_omega2", "sigma_omega2", "mu_log_beta", "sigma_log_beta",
                 "mu_bias", "sigma_bias"]:
        if name in samples:
            row = {"param": name, "session": None}
            row.update(_stats(samples[name]))
            rows.append(row)

    for k, m in enumerate(models):
        for pname in ("omega2", "log_beta", "bias"):
            if pname in samples:
                row = {"param": pname, "session": m.session_id}
                row.update(_stats(samples[pname][..., k]))
                rows.append(row)

    return pd.DataFrame(rows)


def samples_to_natural_df(samples: dict,
                          session_ids: list[str]) -> "pd.DataFrame":
    """Convert raw NUTS samples to a tidy DataFrame in natural-param space.

    Columns: chain, draw, session_id, omega2, beta, bias
    """
    import pandas as pd

    K = len(session_ids)
    rows = []
    omega2s   = np.asarray(samples["omega2"])     # (n_chains, n_samples, K)
    log_betas = np.asarray(samples["log_beta"])
    biases    = np.asarray(samples["bias"])
    n_chains, n_samp = omega2s.shape[:2]

    for c in range(n_chains):
        for s in range(n_samp):
            for k, sid in enumerate(session_ids):
                rows.append({
                    "chain": c, "draw": s, "session_id": sid,
                    "omega2": float(omega2s[c, s, k]),
                    "beta":   float(np.exp(log_betas[c, s, k])),
                    "bias":   float(biases[c, s, k]),
                })
    return pd.DataFrame(rows)
