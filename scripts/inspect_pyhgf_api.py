"""Introspect the installed pyhgf API (authoritative — not the docs).

Prints versions and the exact signatures/docstrings we depend on:
HGF constructor, input_data (missing-input / observed mask), the response
module, HGFDistribution (PyMC), and trajectory extraction. Run once to pin
the API before writing model code.

Run: python scripts/inspect_pyhgf_api.py
"""

from __future__ import annotations

import inspect


def show(obj, name: str) -> None:
    print("=" * 78)
    print(name)
    print("-" * 78)
    try:
        print("signature:", inspect.signature(obj))
    except (TypeError, ValueError) as e:
        print("signature: <unavailable>", e)
    doc = inspect.getdoc(obj) or ""
    print(doc[:1500])


def main() -> None:
    import jax
    import pyhgf
    import pymc
    print("pyhgf", pyhgf.__version__, "| jax", jax.__version__,
          "| pymc", pymc.__version__)

    from pyhgf.model import HGF
    show(HGF.__init__, "HGF.__init__")
    show(HGF.input_data, "HGF.input_data")
    show(HGF.surprise, "HGF.surprise")
    if hasattr(HGF, "to_pandas"):
        show(HGF.to_pandas, "HGF.to_pandas")

    import pyhgf.response as resp
    print("=" * 78)
    print("pyhgf.response members:")
    for nm in dir(resp):
        if not nm.startswith("_"):
            print("   ", nm)
    show(resp.binary_softmax_inverse_temperature,
         "response.binary_softmax_inverse_temperature")
    if hasattr(resp, "binary_softmax"):
        show(resp.binary_softmax, "response.binary_softmax")

    from pyhgf.distribution import HGFDistribution, hgf_logp
    show(HGFDistribution.__init__, "HGFDistribution.__init__")
    show(hgf_logp, "hgf_logp")

    # utils export
    try:
        from pyhgf.utils import to_pandas
        show(to_pandas, "pyhgf.utils.to_pandas")
    except Exception as e:
        print("to_pandas import failed:", e)

    # Build a tiny binary 3-level HGF to confirm construction + attributes.
    print("=" * 78)
    print("BUILD CHECK: 3-level binary HGF")
    import numpy as np
    hgf = HGF(n_levels=3, model_type="binary",
              tonic_volatility={"2": -3.0, "3": -2.0})
    print("type:", type(hgf))
    print("attributes of interest:",
          [a for a in dir(hgf) if not a.startswith("_")][:40])

    # Run a short sequence with a missing observation to confirm masking.
    u = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=float)
    observed = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=int)  # 0 = unobserved
    try:
        hgf.input_data(input_data=u, observed=observed)
        print("input_data(observed=...) OK")
    except TypeError as e:
        print("input_data observed kw failed:", e)
        hgf.input_data(input_data=u)
        print("input_data without observed OK")

    # Trajectory extraction
    try:
        df = hgf.to_pandas()
        print("to_pandas columns:", list(df.columns))
        print(df.head(3).to_string())
    except Exception as e:
        print("to_pandas failed:", e)
        nt = getattr(hgf, "node_trajectories", None)
        print("node_trajectories type:", type(nt))
        if nt is not None:
            print("len:", len(nt))
            print("keys of node[0]:",
                  list(nt[0].keys()) if hasattr(nt[0], "keys") else "n/a")


if __name__ == "__main__":
    main()
