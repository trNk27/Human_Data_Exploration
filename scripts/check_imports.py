"""Import-smoke every module after the refactor.

Run from the repo root:  python scripts/check_imports.py

Compiles all .py files (syntax) and imports the modules that don't require
optional heavy deps. Modules needing `zetapy` are syntax-checked only.
"""

import compileall
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- syntax check everything (skips results/, __pycache__ handled by compileall) ---
ok = compileall.compile_dir(ROOT, quiet=1, force=True,
                            rx=__import__("re").compile(r"(__pycache__|\.git|results|test_session)"))
print(f"compileall: {'OK' if ok else 'FAILED'}")

# --- import the modules that are safe to import (no zetapy) ---
SAFE = [
    "utils", "session", "compute", "explore",
    "viewers.psth", "viewers.raster_plot", "viewers.autocorrelogram",
    "viewers.firing_rate_vs_perc_p", "viewers.browser",
    "analysis.population_heatmap", "analysis.export_acg",
    "analysis.choice_timeline", "analysis.behavioural_simulation",
    "analysis.outcome_direction", "analysis.responsive_region",
    "analysis.batch_export_acg",
]
# zetapy-dependent: syntax-checked above, imported only if zetapy is present.
ZETA = ["analysis.zeta_analysis", "analysis.zeta_outcome", "analysis.batch_zeta"]

failures = []
for mod in SAFE:
    try:
        importlib.import_module(mod)
        print(f"  import OK   {mod}")
    except Exception as exc:                       # noqa: BLE001
        failures.append((mod, repr(exc)))
        print(f"  import FAIL {mod}: {exc!r}")

have_zeta = importlib.util.find_spec("zetapy") is not None
for mod in ZETA:
    if not have_zeta:
        print(f"  skip (no zetapy) {mod}")
        continue
    try:
        importlib.import_module(mod)
        print(f"  import OK   {mod}")
    except Exception as exc:                       # noqa: BLE001
        failures.append((mod, repr(exc)))
        print(f"  import FAIL {mod}: {exc!r}")

if failures or not ok:
    print(f"\nFAILURES: {len(failures)}")
    raise SystemExit(1)
print("\nAll import/compile checks passed.")
