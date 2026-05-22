"""Run export_acg.py for every session not yet processed.

Invokes each session via `--session <id>`; no utils.py rewriting.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from utils import REPO_ROOT

DONE = {"20250521", "20250602"}

HERE = Path(REPO_ROOT)


def main():
    argparse.ArgumentParser(
        description="Run export_acg.py for every YYYYMMDD/ session not in the DONE set."
    ).parse_args()

    sessions = sorted(
        d.name for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}', d.name) and d.name not in DONE
    )

    if not sessions:
        print("Nothing to do — all sessions already exported.")
        return

    print(f"Sessions to export: {sessions}\n")

    for session in sessions:
        print(f"=== {session} ===")
        result = subprocess.run(
            [sys.executable, str(HERE / "export_acg.py"), "--session", session],
            cwd=str(HERE),
        )
        if result.returncode != 0:
            print(f"  WARNING: export_acg.py exited with code {result.returncode} for {session}")
        print()


if __name__ == "__main__":
    main()
