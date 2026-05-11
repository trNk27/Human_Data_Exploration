"""Run export_acg.py for every session not yet processed."""

import re
import subprocess
import sys
from pathlib import Path

DONE = {"20250521", "20250602"}

HERE = Path(__file__).parent
UTILS = HERE / "utils.py"
SESSION_RE = re.compile(r'^(SESSION\s*=\s*")[^"]*(")', re.MULTILINE)


def set_session(text: str, session: str) -> str:
    return SESSION_RE.sub(rf'\g<1>{session}\g<2>', text)


def main():
    sessions = sorted(
        d.name for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}', d.name) and d.name not in DONE
    )

    if not sessions:
        print("Nothing to do — all sessions already exported.")
        return

    print(f"Sessions to export: {sessions}\n")

    original = UTILS.read_text(encoding="utf-8")
    try:
        for session in sessions:
            print(f"=== {session} ===")
            UTILS.write_text(set_session(original, session), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(HERE / "export_acg.py")],
                cwd=str(HERE),
            )
            if result.returncode != 0:
                print(f"  WARNING: export_acg.py exited with code {result.returncode} for {session}")
            print()
    finally:
        UTILS.write_text(original, encoding="utf-8")
        print("utils.py restored.")


if __name__ == "__main__":
    main()
