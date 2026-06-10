"""Quick test of the shared + separate MAP fits on the real data.

Run: python scripts/hgf_test_fit.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hgf.data import load_all_sessions, list_sessions
from analysis.hgf.model import make_session_models
from analysis.hgf.fit import shared_map_fit, fit_all_separate


def main() -> None:
    t0 = time.perf_counter()
    sessions = list_sessions()
    print("Sessions:", sessions)
    sds = load_all_sessions(sessions)
    models = make_session_models(sds)
    print(f"Loaded + built {len(models)} session models in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    shared = shared_map_fit(models, n_restarts=4)
    print(f"\nShared MAP fit in {time.perf_counter()-t1:.1f}s  success={shared.success}")
    print("  omega2 = %.3f | beta = %.3f | bias = %.3f"
          % (shared.natural["omega2"], shared.natural["beta"], shared.natural["bias"]))
    print("  total loglik = %.1f  (n_trials=%d)  BIC=%.1f"
          % (shared.loglik, shared.n_trials, shared.as_row()["bic"]))

    print("\nPer-session predictive accuracy (shared params):")
    rows = pd.DataFrame(list(shared.per_session.values()))
    print(rows.to_string(index=False))
    print("  mean balanced accuracy: %.3f | mean pseudo-R2: %.3f"
          % (rows["balanced_accuracy"].mean(), rows["pseudo_r2"].mean()))

    t2 = time.perf_counter()
    sep = fit_all_separate(models, n_restarts=3)
    print(f"\nSeparate per-session MAP fits in {time.perf_counter()-t2:.1f}s")
    sep_df = pd.DataFrame([{
        "session": r.label.split(":")[1],
        "omega2": r.natural["omega2"],
        "beta": r.natural["beta"],
        "bias": r.natural["bias"],
        "loglik_per_trial": r.loglik / r.n_trials,
    } for r in sep])
    print(sep_df.to_string(index=False))
    print("  omega2 range: %.3f..%.3f | beta range: %.3f..%.3f | bias range: %.3f..%.3f"
          % (sep_df.omega2.min(), sep_df.omega2.max(),
             sep_df.beta.min(), sep_df.beta.max(),
             sep_df.bias.min(), sep_df.bias.max()))

    print(f"\nTotal time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
