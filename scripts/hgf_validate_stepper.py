"""Validate the closed-loop simulator's stepping against pyhgf's filter.

Two checks:
  A. The prediction (expected_mean) read from scan_fn is independent of the
     current input — so reading p_hat via a throwaway observed=0 step is valid.
  B. Stepping a fixed (u, observed) sequence one trial at a time with scan_fn
     reproduces pyhgf's full-sequence input_data trajectory exactly.

Run: python scripts/hgf_validate_stepper.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hgf.model import build_hgf


def main() -> None:
    rng = np.random.default_rng(0)
    T = 60
    u = (rng.random(T) < 0.5).astype(float)
    observed = (rng.random(T) < 0.5).astype(int)

    # Full-sequence pyhgf filter
    hgf = build_hgf(omega2=-2.5)
    hgf.input_data(input_data=u, observed=observed)
    ph_full = np.asarray(hgf.node_trajectories[0]["expected_mean"])

    # Manual stepping with scan_fn
    hgf2 = build_hgf(omega2=-2.5)
    if hgf2.scan_fn is None:
        hgf2 = hgf2.create_belief_propagation_fn()
    scan_fn = hgf2.scan_fn
    attributes = hgf2.attributes

    ph_step = []
    pred_indep_ok = True
    for t in range(T):
        # prediction via throwaway observed=0 step
        dummy = ((jnp.array([0.0]),), (jnp.array(0),), jnp.array(1.0), None)
        _, traj_dummy = scan_fn(attributes, dummy)
        p_dummy = float(traj_dummy[0]["expected_mean"])

        # real step
        real = ((jnp.array([u[t]]),), (jnp.array(int(observed[t])),), jnp.array(1.0), None)
        new_attr, traj_real = scan_fn(attributes, real)
        p_real = float(traj_real[0]["expected_mean"])

        # Check A: prediction independent of input
        if not np.isclose(p_dummy, p_real, atol=1e-6):
            pred_indep_ok = False
        ph_step.append(p_real)
        attributes = new_attr

    ph_step = np.array(ph_step)
    max_diff = float(np.max(np.abs(ph_full - ph_step)))
    print("Check A (prediction independent of current input):", pred_indep_ok)
    print(f"Check B (stepwise vs full-sequence p_hat) max|diff| = {max_diff:.2e}")
    assert pred_indep_ok, "prediction depends on input — dummy-read trick invalid"
    assert max_diff < 1e-5, "stepwise filter disagrees with pyhgf full-sequence"
    print("\nSTEPPER VALIDATED — simulator uses pyhgf's exact update.")


if __name__ == "__main__":
    main()
