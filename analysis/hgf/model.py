"""The HGF perceptual model + the custom 'shadowing' response model.

Perceptual model
----------------
A 3-level binary HGF (``pyhgf``) tracking the gamble arm's probability of paying
its big reward. Partial feedback is handled with ``input_data(observed=...)``:
on safe-choice trials the gamble outcome is unobserved and the belief diffuses.

Response model ('shadowing')
----------------------------
Given the HGF's predicted gamble-win probability ``p_hat = s(mu_hat_2)`` (the
binary node's expected mean, available on every trial), with FIXED rewards:

    EV_gamble = p_hat * R_gamble        V_safe = R_safe
    P(choose gamble) = sigmoid( beta * (EV_gamble - V_safe) + bias )

and the model surprise is the total negative log-likelihood of the choices,
``-sum log P(choice)``.

The differentiable per-session log-likelihood (:func:`session_logp`) mirrors
``pyhgf.distribution.logp`` but additionally forwards the ``observed`` mask, so
the same partial-feedback filter is used for the MAP fit, the hierarchical
model, and simulation.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from pyhgf.model import HGF
from pyhgf.math import binary_surprise

from .config import (
    REWARD_GAMBLE,
    REWARD_SAFE,
    N_LEVELS,
    OMEGA3,
    KAPPA,
    TONIC_VOLATILITY_1,
    INIT_MEAN,
    INIT_PRECISION,
)
from .data import SessionData


# ---------------------------------------------------------------------------
# HGF construction
# ---------------------------------------------------------------------------

def build_hgf(omega2: float = -3.0, omega3: float = OMEGA3, kappa: float = KAPPA) -> HGF:
    """Build a fresh 3-level binary HGF with the project's fixed structure.

    ``omega2`` (level-2 tonic volatility) is the only perceptual free parameter;
    ``omega3`` and ``kappa`` (volatility coupling) are fixed.
    """
    return HGF(
        n_levels=N_LEVELS,
        model_type="binary",
        initial_mean=dict(INIT_MEAN),
        initial_precision=dict(INIT_PRECISION),
        tonic_volatility={"1": TONIC_VOLATILITY_1, "2": float(omega2), "3": float(omega3)},
        volatility_coupling={"1": 1.0, "2": float(kappa)},
    )


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

def p_choose_gamble(p_hat, beta, bias):
    """P(choose gamble) from the predicted win prob and response parameters."""
    ev_gamble = p_hat * REWARD_GAMBLE
    logit = beta * (ev_gamble - REWARD_SAFE) + bias
    return 1.0 / (1.0 + jnp.exp(-logit))


def gamble_choice_surprise(hgf, response_function_inputs, response_function_parameters):
    """Custom pyhgf response function: total choice surprise (-sum log P).

    Parameters
    ----------
    hgf :
        The HGF after ``input_data`` has run (trajectories populated).
    response_function_inputs :
        The participant choices ``y`` (1 = gamble, 0 = safe), shape ``(T,)``.
    response_function_parameters :
        Array ``[beta, bias]``.
    """
    y = response_function_inputs
    beta = response_function_parameters[0]
    bias = response_function_parameters[1]

    p_hat = hgf.node_trajectories[0]["expected_mean"]   # s(mu_hat_2) per trial
    p_g = p_choose_gamble(p_hat, beta, bias)
    surprise = binary_surprise(x=y, expected_mean=p_g)
    return jnp.sum(surprise)


# ---------------------------------------------------------------------------
# Differentiable per-session log-likelihood (mirrors pyhgf.distribution.logp,
# but forwards the `observed` mask for partial feedback)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("hgf",))
def _session_logp_jit(omega2, beta, bias, hgf, u, observed, y):
    """JAX log-likelihood of one session's choices given (omega2, beta, bias).

    ``hgf`` is a static, pre-built binary HGF for this session. ``omega2`` is
    injected into the (constant) level-2 tonic volatility, exactly as pyhgf's
    own ``logp`` does, then the filter runs with the partial-feedback mask.
    """
    hgf.attributes[1]["tonic_volatility"] = omega2
    # Pass `observed` pre-wrapped as a per-input-node tuple: a traced JAX array is
    # not an np.ndarray, so pyhgf would not wrap it itself.
    surprise = hgf.input_data(input_data=u, observed=(observed,)).surprise(
        response_function=gamble_choice_surprise,
        response_function_inputs=y,
        response_function_parameters=jnp.array([beta, bias]),
    )
    return -surprise


class SessionModel:
    """A session's static HGF plus cached JAX arrays — the unit of fitting.

    Holds one pre-built :class:`pyhgf.model.HGF` (structure fixed; only ``omega2``
    varies between evaluations) and the session's input/observed/choice arrays as
    device arrays, so repeated likelihood evaluations avoid rebuilding anything.
    """

    def __init__(self, sd: SessionData, omega3: float = OMEGA3, kappa: float = KAPPA):
        self.sd = sd
        self.session_id = sd.session_id
        self._hgf = build_hgf(omega2=-3.0, omega3=omega3, kappa=kappa)
        self._u = jnp.asarray(sd.u, dtype=float)
        self._observed = jnp.asarray(sd.observed, dtype=int)
        self._y = jnp.asarray(sd.y, dtype=float)

    def logp(self, omega2: float, beta: float, bias: float) -> float:
        """Log-likelihood of the choices under (omega2, beta, bias)."""
        return _session_logp_jit(
            jnp.asarray(omega2, float), jnp.asarray(beta, float), jnp.asarray(bias, float),
            self._hgf, self._u, self._observed, self._y,
        )

    def neg_logp(self, omega2: float, beta: float, bias: float) -> float:
        return -self.logp(omega2, beta, bias)

    def trajectories(self, omega2: float):
        """Run the filter at ``omega2`` and return the pyhgf trajectory DataFrame."""
        self._hgf.attributes[1]["tonic_volatility"] = float(omega2)
        self._hgf.input_data(input_data=self.sd.u, observed=self.sd.observed)
        return self._hgf.to_pandas()


def make_session_models(session_data: list[SessionData], **kwargs) -> list[SessionModel]:
    return [SessionModel(sd, **kwargs) for sd in session_data]


# ---------------------------------------------------------------------------
# Closed-loop simulation (uses pyhgf's EXACT step via scan_fn)
# ---------------------------------------------------------------------------

def simulate_session(
    omega2: float,
    beta: float,
    bias: float,
    p_schedule: np.ndarray,
    seed: int,
    omega3: float = OMEGA3,
    kappa: float = KAPPA,
) -> SessionData:
    """Simulate one session of an HGF agent with known parameters.

    Closed-loop and partial-feedback consistent: at each trial the agent forms a
    prediction ``p_hat`` from its current beliefs, samples a choice from
    ``P(choose gamble)``, and — only if it chose gamble — observes a Bernoulli
    outcome drawn from the TRUE schedule and updates its beliefs. Uses pyhgf's
    own single-step function (:pyattr:`HGF.scan_fn`), so the generative model is
    exactly the model that is fit.

    Returns a :class:`SessionData` with the simulated choices, observed mask and
    perceptual input, on the given ``p_schedule``.
    """
    p_schedule = np.asarray(p_schedule, dtype=float)
    T = len(p_schedule)

    hgf = build_hgf(omega2=omega2, omega3=omega3, kappa=kappa)
    if hgf.scan_fn is None:
        hgf = hgf.create_belief_propagation_fn()
    scan_fn = hgf.scan_fn
    init_attributes = hgf.attributes

    def step(carry, xs):
        attributes, key = carry
        p_sched_t, _ = xs
        key, k_choice, k_out = jax.random.split(key, 3)

        # Prediction p_hat[t] depends only on the carry; read it via a throwaway
        # step (its returned attributes are discarded).
        dummy_in = ((jnp.array([0.0]),), (jnp.array(0),), jnp.array(1.0), None)
        _, traj_pred = scan_fn(attributes, dummy_in)
        p_hat = traj_pred[0]["expected_mean"]

        # Sample the choice, then the outcome (only seen if gamble chosen).
        p_g = p_choose_gamble(p_hat, beta, bias)
        choice = (jax.random.uniform(k_choice) < p_g).astype(jnp.int32)   # 1 = gamble
        outcome = (jax.random.uniform(k_out) < p_sched_t).astype(jnp.float32)

        observed_t = choice                                  # gamble -> observed
        value_t = jnp.where(choice == 1, outcome, 0.0)
        real_in = ((jnp.array([value_t]),), (jnp.array(observed_t),), jnp.array(1.0), None)
        new_attributes, _ = scan_fn(attributes, real_in)

        return (new_attributes, key), (p_hat, choice, outcome, observed_t)

    key = jax.random.PRNGKey(int(seed))
    xs = (jnp.asarray(p_schedule), jnp.zeros(T))
    _, (p_hat, choice, outcome, observed) = jax.lax.scan(step, (init_attributes, key), xs)

    choice = np.asarray(choice).astype(float)
    observed = np.asarray(observed).astype(int)
    u = np.where(observed == 1, np.asarray(outcome).astype(float), 0.0)

    return SessionData(
        session_id=f"sim_seed{seed}",
        u=u,
        observed=observed,
        y=choice,
        p_schedule=p_schedule,
        trial_index=np.arange(T),
    )
