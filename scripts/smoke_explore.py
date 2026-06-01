"""Smoke test for the fluent single-neuron `explore` layer.

Run from the repo root:  python scripts/smoke_explore.py

Loads a real session, generates one of each single-neuron plot via the fluent
API, saves them with auto-naming, and asserts the PNGs land on disk.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session import Session   # noqa: E402

SESSION = "20250714"


def main():
    sess = Session(SESSION)
    n0 = sess.neuron(0)
    print(repr(n0))

    panels = [
        n0.psth(condition="G+R", align="reward"),
        n0.psth(condition="all", align="cue", sigma_ms=20),
        n0.psth(align="cue"),                       # all responding trials
        n0.raster(condition="G+N", align="cue"),
        n0.acg(),
        n0.fr_vs_p(window="cue_to_reward"),
        sess.neuron(1).psth(condition="S+R", align="reward"),
    ]

    for p in panels:
        path = p.save()
        assert os.path.exists(path), f"missing: {path}"
        p.close()

    # grid path (many neurons, same draws)
    from explore import grid
    g = grid(sess, "psth", neurons=[0, 1, 2], align="reward", condition="all")
    gpath = g.save(os.path.join("results", "figures", SESSION, "grid_psth_smoke.png"))
    assert os.path.exists(gpath)
    g.close()

    print(f"\nOK — {len(panels)} panels + 1 grid saved under results/figures/{SESSION}/")


if __name__ == "__main__":
    main()
