"""Smoke test for analysis/hgf core: data, model, logp gradients, simulator.

Run: python scripts/hgf_smoke.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hgf.data import load_session, list_sessions
from analysis.hgf.model import (
    SessionModel, build_hgf, simulate_session, gamble_choice_surprise,
)
import jax.numpy as jnp


def main() -> None:
    sessions = list_sessions()
    sid = sessions[0]
    sd = load_session(sid)
    print("Loaded", sd)
    print("  observed frac (gamble):", round(float(np.mean(sd.observed)), 3),
          "| choice gamble frac:", round(float(np.mean(sd.y)), 3))

    # --- 1. logp evaluates and varies with parameters ---
    sm = SessionModel(sd)
    l1 = float(sm.logp(-3.0, 1.0, 0.0))
    l2 = float(sm.logp(-2.0, 2.0, -0.5))
    print(f"\nlogp(-3,1,0)   = {l1:.3f}")
    print(f"logp(-2,2,-0.5)= {l2:.3f}")
    assert np.isfinite(l1) and np.isfinite(l2), "logp not finite"
    assert l1 != l2, "logp does not vary with parameters"

    # --- 2. gradient flows through the HGF + response model ---
    grad_fn = jax.grad(lambda th: -_neg(sm, th), argnums=0)
    g = grad_fn(jnp.array([-3.0, 1.0, 0.0]))
    print("grad d(logp)/d[omega2,beta,bias] =", np.asarray(g))
    assert np.all(np.isfinite(np.asarray(g))), "non-finite gradient"

    # --- 3. cross-check logp against pyhgf's direct surprise path ---
    hgf = build_hgf(omega2=-3.0)
    hgf.input_data(input_data=sd.u, observed=sd.observed)
    s_direct = float(hgf.surprise(
        response_function=gamble_choice_surprise,
        response_function_inputs=jnp.asarray(sd.y),
        response_function_parameters=jnp.array([1.0, 0.0]),
    ))
    print(f"\ndirect pyhgf surprise = {s_direct:.3f}  vs  -logp = {-l1:.3f}")
    assert abs(s_direct - (-l1)) < 1e-2, "logp disagrees with direct pyhgf surprise"

    # --- 4. simulator: closed-loop, and beliefs match a full-sequence refilter ---
    sim = simulate_session(omega2=-2.5, beta=2.0, bias=-0.5, p_schedule=sd.p_schedule, seed=1)
    print("\nSimulated", sim, "| gamble frac:", round(float(np.mean(sim.y)), 3))
    # Refit-filter the simulated (u, observed) with pyhgf and compare p_hat to a
    # second simulator run's internal predictions is implicit; here we just check
    # that the simulated sequence filters to a finite surprise.
    sm_sim = SessionModel(sim)
    lsim = float(sm_sim.logp(-2.5, 2.0, -0.5))
    print("logp of simulated data at true params:", round(lsim, 3))
    assert np.isfinite(lsim)

    # --- 5. p_hat from pyhgf full-sequence filter is in (0,1) and reacts ---
    df = sm.trajectories(-2.5)
    ph = df["x_0_expected_mean"].to_numpy()
    print("\np_hat range:", round(float(ph.min()), 3), "to", round(float(ph.max()), 3),
          "| mean", round(float(ph.mean()), 3))
    assert ph.min() >= 0 and ph.max() <= 1

    print("\nALL SMOKE CHECKS PASSED")


def _neg(sm, theta):
    # helper so jax.grad sees a pure function of theta=[omega2, beta, bias]
    from analysis.hgf.model import _session_logp_jit
    return -_session_logp_jit(theta[0], theta[1], theta[2],
                              sm._hgf, sm._u, sm._observed, sm._y)


if __name__ == "__main__":
    main()
