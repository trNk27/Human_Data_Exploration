"""PSTH grid CLI — one PSTH subplot per neuron, aligned to a behavioural event.

Thin wrapper over `explore.grid(sess, "psth", ...)`. For a single neuron use the
fluent API instead:  `Session(...).neuron(7).psth(condition="G+R").save()`.

    python -m viewers.psth --session 20250714 --area ACC --event reward --save
    python -m viewers.psth --neurons 0 1 5 --by-condition
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib.pyplot as plt

from session import Session
from explore import grid
from utils import (
    EVENTS, add_session_arg, add_selection_args,
    add_save_arg, maybe_save, handle_list, session_data_dir,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peristimulus time histogram (PSTH) grid")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--event", type=str, default="cue", choices=list(EVENTS),
                        help="Behavioural event to align to (default: cue)")
    parser.add_argument("--pre",   type=float, default=500,  help="ms before event (default: 500)")
    parser.add_argument("--post",  type=float, default=1000, help="ms after event (default: 1000)")
    parser.add_argument("--bin",   type=float, default=50,   help="Bin width in ms (default: 50)")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian smoothing SD in ms (default: off)")
    parser.add_argument("--by-condition", action="store_true",
                        help="Overlay one curve per (arm, reward) condition")
    add_save_arg(parser)
    args = parser.parse_args()

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess  = Session(args.session)
    panel = grid(
        sess, "psth",
        neurons=args.neurons, area=args.area,
        align=args.event, pre_ms=args.pre, post_ms=args.post,
        bin_ms=args.bin, sigma_ms=args.sigma,
        condition="all" if args.by_condition else None,
    )
    maybe_save(panel.fig, args, prefix="psth")
    plt.show()
