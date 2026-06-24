"""Confirm where pyhgf stores omega3 and kappa, and that gradients flow through
them, so the extended (5-param) HGF can inject them like omega2."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from analysis.hgf.model import build_hgf, gamble_choice_surprise
from analysis.hgf.data import load_session

hgf = build_hgf(omega2=-3.0, omega3=-4.0, kappa=1.0)

print("=== attribute keys per node ===")
for i in sorted(k for k in hgf.attributes.keys() if isinstance(k, int)):
    print(f"node {i}: {sorted(hgf.attributes[i].keys())}")

print("\n=== current values ===")
print("attributes[1]['tonic_volatility'] (omega2):", hgf.attributes[1].get("tonic_volatility"))
print("attributes[2]['tonic_volatility'] (omega3):", hgf.attributes[2].get("tonic_volatility"))
print("attributes[1]['volatility_coupling_parents']:", hgf.attributes[1].get("volatility_coupling_parents"))
print("attributes[2]['volatility_coupling_children']:", hgf.attributes[2].get("volatility_coupling_children"))

# --- gradient test: does logp depend (differentiably) on omega3 and kappa? ---
sd = load_session("20250605")
u = jnp.asarray(sd.u, float)
observed = jnp.asarray(sd.observed, int)
y = jnp.asarray(sd.y, float)


def logp(omega2, omega3, kappa, beta, bias):
    h = build_hgf(omega2=-3.0, omega3=-4.0, kappa=1.0)
    h.attributes[1]["tonic_volatility"] = omega2
    h.attributes[2]["tonic_volatility"] = omega3
    h.attributes[1]["volatility_coupling_parents"] = (kappa,)
    h.attributes[2]["volatility_coupling_children"] = (kappa,)
    surprise = h.input_data(input_data=u, observed=(observed,)).surprise(
        response_function=gamble_choice_surprise,
        response_function_inputs=y,
        response_function_parameters=jnp.array([beta, bias]),
    )
    return -surprise


args = (jnp.array(-3.0), jnp.array(-4.0), jnp.array(1.0), jnp.array(2.0), jnp.array(0.0))
val = logp(*args)
grads = jax.grad(logp, argnums=(0, 1, 2, 3, 4))(*args)
print("\n=== logp and gradients ===")
print("logp:", float(val))
for name, g in zip(["omega2", "omega3", "kappa", "beta", "bias"], grads):
    print(f"  d logp / d {name:7s} = {float(g): .5f}")
print("\nOK" if all(np.isfinite(float(g)) for g in grads) else "\nNON-FINITE GRAD")
