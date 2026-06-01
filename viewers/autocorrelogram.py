"""Autocorrelogram grid CLI — one ACG subplot per neuron.

Thin wrapper over `explore.grid(sess, "acg", ...)`. For a single neuron:
    Session(...).neuron(12).acg().save()

The ACG kernel itself lives in `compute.compute_acg`.

    python -m viewers.autocorrelogram --session 20250714 --area ACC --lag 200 --save
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib.pyplot as plt

from session import Session
from explore import grid
from utils import (
    add_session_arg, add_selection_args, add_save_arg, maybe_save,
    handle_list, session_data_dir,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocorrelogram grid")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--lag", type=float, default=200, help="Max lag in ms (default 200)")
    parser.add_argument("--bin", type=float, default=1,   help="Bin size in ms (default 1)")
    add_save_arg(parser)
    args = parser.parse_args()

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess  = Session(args.session)
    panel = grid(sess, "acg", neurons=args.neurons, area=args.area,
                 lag_ms=args.lag, bin_ms=args.bin)
    maybe_save(panel.fig, args, prefix="acg")
    plt.show()
