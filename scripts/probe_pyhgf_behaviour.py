"""De-risk pyhgf behaviours that determine the HGF pipeline architecture.

1. Does a NaN input auto-trigger the missing/unobserved path (so the PyMC
   HGFDistribution route, which never passes an `observed` mask, can still do
   partial feedback by NaN-encoding safe trials)?
2. Is NaN-encoding equivalent to the explicit `observed=` mask?
3. Does node_trajectories[0]['expected_mean'] == to_pandas x_0_expected_mean?
4. Timing: cost of a full ~900-trial filter + surprise, and whether re-running
   input_data after editing tonic_volatility recompiles.

Run: python scripts/probe_pyhgf_behaviour.py
"""

from __future__ import annotations

import time

import numpy as np
import jax.numpy as jnp

from pyhgf.model import HGF
from pyhgf.math import binary_surprise


def build(omega2: float = -3.0, omega3: float = -4.0) -> HGF:
    return HGF(
        n_levels=3,
        model_type="binary",
        tonic_volatility={"1": 0.0, "2": omega2, "3": omega3},
        volatility_coupling={"1": 1.0, "2": 1.0},
    )


def main() -> None:
    rng = np.random.default_rng(0)
    T = 40
    u = (rng.random(T) < 0.6).astype(float)
    observed = (rng.random(T) < 0.5).astype(int)   # 1 = gamble chosen / observed

    # ---- 1 & 2: NaN vs observed-mask equivalence ----
    u_nan = u.copy()
    u_nan[observed == 0] = np.nan

    h_mask = build()
    h_mask.input_data(input_data=u, observed=observed)
    df_mask = h_mask.to_pandas()

    nan_ok = True
    try:
        h_nan = build()
        h_nan.input_data(input_data=u_nan)
        df_nan = h_nan.to_pandas()
        print("NaN input accepted (no observed mask).")
        print("  x_0_observed (nan-encoded):", df_nan["x_0_observed"].to_numpy().astype(int)[:20])
        print("  observed mask            :", observed[:20])
        same_obs = np.array_equal(df_nan["x_0_observed"].to_numpy().astype(int), observed)
        print("  observed columns match mask:", same_obs)
        ph_mask = df_mask["x_0_expected_mean"].to_numpy()
        ph_nan = df_nan["x_0_expected_mean"].to_numpy()
        print("  max |p_hat(nan) - p_hat(mask)|:", float(np.nanmax(np.abs(ph_nan - ph_mask))))
    except Exception as e:
        nan_ok = False
        print("NaN input WITHOUT observed mask FAILED:", repr(e))

    # ---- 3: node_trajectories vs to_pandas ----
    nt_pe = np.asarray(h_mask.node_trajectories[0]["expected_mean"])
    tp_pe = df_mask["x_0_expected_mean"].to_numpy()
    print("\nnode_trajectories[0]['expected_mean'] == to_pandas x_0_expected_mean:",
          np.allclose(nt_pe, tp_pe))
    print("  available keys node0:", list(h_mask.node_trajectories[0].keys()))
    print("  available keys node1:", list(h_mask.node_trajectories[1].keys()))

    # ---- 4: timing on a realistic ~900-trial sequence ----
    Tbig = 900
    ub = (rng.random(Tbig) < 0.5).astype(float)
    ob = (rng.random(Tbig) < 0.4).astype(int)
    yb = (rng.random(Tbig) < 0.4).astype(float)

    def run_surprise(omega2, beta, bias):
        h = build(omega2=omega2)
        h.input_data(input_data=ub, observed=ob)
        p_hat = h.node_trajectories[0]["expected_mean"]
        ev_g = p_hat * 4.0
        logit = beta * (ev_g - 1.0) + bias
        p_g = 1.0 / (1.0 + jnp.exp(-logit))
        return float(jnp.sum(binary_surprise(x=jnp.asarray(yb), expected_mean=p_g)))

    t0 = time.perf_counter()
    s1 = run_surprise(-3.0, 1.0, 0.0)
    t1 = time.perf_counter()
    s2 = run_surprise(-2.0, 1.5, -0.3)   # different omega2 -> recompile?
    t2 = time.perf_counter()
    s3 = run_surprise(-2.5, 1.2, 0.1)
    t3 = time.perf_counter()
    print(f"\nsurprise eval 1 (cold): {t1-t0:.3f}s  surprise={s1:.2f}")
    print(f"surprise eval 2 (new omega2): {t2-t1:.3f}s  surprise={s2:.2f}")
    print(f"surprise eval 3: {t3-t2:.3f}s  surprise={s3:.2f}")
    print("=> per-eval cost after warmup ~", f"{t3-t2:.3f}s")


if __name__ == "__main__":
    main()
