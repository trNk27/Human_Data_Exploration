"""Bare-mode load of ui/app.py to catch import errors / NameErrors / typos.

Run from the repo root:  python scripts/check_ui.py

Streamlit needs a browser session to fully run; without one, widget calls return
defaults and `st.session_state` access raises a Streamlit context error. We treat
*Streamlit* runtime errors as "fine — needs a browser", but surface any real
ImportError / NameError / AttributeError from our edits.
"""

import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

APP = os.path.join(ROOT, "ui", "app.py")
spec = importlib.util.spec_from_file_location("ui_app", APP)
mod = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(mod)
    print("ui/app.py executed fully (unexpected without a browser, but fine).")
except (ImportError, NameError, AttributeError, SyntaxError) as exc:
    print(f"REAL ERROR in ui/app.py: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
except Exception as exc:                       # noqa: BLE001
    name = type(exc).__module__ + "." + type(exc).__name__
    msg = str(exc).lower()
    if "streamlit" in name.lower() or "session_state" in msg or "scriptrun" in msg or "context" in msg:
        print(f"ui/app.py imports + body OK (stopped at Streamlit runtime: {type(exc).__name__}).")
    else:
        print(f"REAL ERROR in ui/app.py: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
