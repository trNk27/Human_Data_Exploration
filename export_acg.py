"""Batch-export autocorrelograms for every neuron in a session.

For each neuron: one PNG with two side-by-side panels (lag +/-75 ms and +/-300 ms),
both at 1 ms bin resolution. Output goes to acg_export/<session>/.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from utils import (
    REPO_ROOT, get_spike_trains,
    add_session_arg, session_data_dir,
)
from autocorrelogram import compute_acg

BIN_MS = 1
LAGS   = (75, 300)


def export_all(session, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    trains, labels = get_spike_trains(data_dir=data_dir)
    n_neurons = len(trains)
    pad = int(math.log10(n_neurons)) + 1   # zero-pad width for filenames

    for idx, (train, label) in enumerate(zip(trains, labels)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

        for ax, lag_ms in zip(axes, LAGS):
            centres, counts = compute_acg(train, lag_ms=lag_ms, bin_ms=BIN_MS)
            ax.bar(centres, counts, width=BIN_MS, color="steelblue", edgecolor="none")
            ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
            ax.set_title(f"Lag +/-{lag_ms} ms", fontsize=9)
            ax.set_xlabel("Lag (ms)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"[{idx}] {label}  -  session {session}, bin {BIN_MS} ms",
                     fontsize=8, y=1.01)
        fig.tight_layout()

        safe_label = "".join(c if c.isalnum() or c in "._- " else "_" for c in label).strip("_")
        fname = f"{idx:0{pad}d}_{safe_label}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  [{idx+1}/{n_neurons}] {fname}")

    print(f"\nDone. {n_neurons} PNGs saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch ACG export for one session.")
    add_session_arg(parser)
    args = parser.parse_args()

    data_dir = session_data_dir(args.session)
    out_dir  = os.path.join(REPO_ROOT, "acg_export", args.session)

    print(f"Exporting ACGs for session {args.session} -> {out_dir}")
    export_all(args.session, data_dir, out_dir)


if __name__ == "__main__":
    main()
