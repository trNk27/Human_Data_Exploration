"""Interactive neuron browser.

Shows a PSTH (top) and autocorrelogram (bottom) for one neuron at a time.
Navigate with Prev / Next buttons, arrow keys, or type a neuron index directly.
The two panels reuse `explore.draw_psth` / `explore.draw_acg`.

All time parameters are in milliseconds.

    python -m viewers.browser --session 20250714 --event reward
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from session import Session
from explore import draw_psth, draw_acg
from utils import (
    EVENTS, select_neurons,
    add_session_arg, add_selection_args,
)

# Defaults — override via CLI
_PRE_MS     = 500
_POST_MS    = 1000
_BIN_MS     = 50
_LAG_MS     = 200
_BIN_ACG_MS = 1


def build_browser(sess, neuron_indices=None, area=None, event="cue",
                  pre_ms=_PRE_MS, post_ms=_POST_MS, bin_ms=_BIN_MS,
                  lag_ms=_LAG_MS, bin_acg_ms=_BIN_ACG_MS):
    trains, labels = sess.spike_trains
    trains, labels = select_neurons(trains, labels,
                                    indices=neuron_indices, area=area,
                                    enforce_cap=False)

    state = {"idx": 0}

    # ---- layout ----
    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.18, hspace=0.5)
    ax_psth = fig.add_subplot(2, 1, 1)
    ax_acg  = fig.add_subplot(2, 1, 2)

    ax_prev = fig.add_axes([0.12, 0.04, 0.15, 0.07])
    ax_box  = fig.add_axes([0.38, 0.04, 0.24, 0.07])
    ax_next = fig.add_axes([0.73, 0.04, 0.15, 0.07])

    btn_prev = Button(ax_prev, "< Prev")
    btn_next = Button(ax_next, "Next >")
    txt_box  = TextBox(ax_box, "Neuron #", initial="0")

    def draw(idx):
        ax_psth.cla()
        ax_acg.cla()

        train, label = trains[idx], labels[idx]

        draw_psth(ax_psth, sess, train, label, align=event,
                  pre_ms=pre_ms, post_ms=post_ms, bin_ms=bin_ms)
        ax_psth.set_title(f"PSTH — aligned to: {event}", fontsize=8)
        ax_psth.legend(fontsize=6, loc="upper right")

        draw_acg(ax_acg, sess, train, label, lag_ms=lag_ms, bin_ms=bin_acg_ms)
        ax_acg.set_title(f"Autocorrelogram (+/-{lag_ms} ms, bin {bin_acg_ms} ms)", fontsize=8)

        fig.suptitle(
            f"[{idx} / {len(trains) - 1}]  {labels[idx]}\nSession {sess.id}",
            fontsize=9, y=0.97,
        )
        txt_box.set_val(str(idx))
        fig.canvas.draw_idle()

    def go(idx):
        state["idx"] = idx % len(trains)
        draw(state["idx"])

    def on_prev(_):    go(state["idx"] - 1)
    def on_next(_):    go(state["idx"] + 1)
    def on_submit(v):
        try:
            go(int(v))
        except ValueError:
            pass

    def on_key(event):
        if event.key == "left":
            go(state["idx"] - 1)
        elif event.key == "right":
            go(state["idx"] + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    txt_box.on_submit(on_submit)
    fig.canvas.mpl_connect("key_press_event", on_key)

    fig._widgets = [btn_prev, btn_next, txt_box]
    draw(0)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive neuron browser")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--event",   type=str, default="cue", choices=list(EVENTS),
                        help="PSTH alignment event (default: cue)")
    parser.add_argument("--pre",     type=float, default=_PRE_MS,
                        help=f"ms before event (default: {_PRE_MS})")
    parser.add_argument("--post",    type=float, default=_POST_MS,
                        help=f"ms after event (default: {_POST_MS})")
    parser.add_argument("--bin",     type=float, default=_BIN_MS,
                        help=f"PSTH bin width in ms (default: {_BIN_MS})")
    parser.add_argument("--bin-acg", type=float, default=_BIN_ACG_MS,
                        help=f"ACG bin width in ms (default: {_BIN_ACG_MS})")
    parser.add_argument("--lag",     type=float, default=_LAG_MS,
                        help=f"ACG max lag in ms (default: {_LAG_MS})")
    args = parser.parse_args()

    sess = Session(args.session)
    build_browser(
        sess,
        neuron_indices=args.neurons, area=args.area,
        event=args.event,
        pre_ms=args.pre, post_ms=args.post, bin_ms=args.bin,
        lag_ms=args.lag, bin_acg_ms=args.bin_acg,
    )
    plt.show()
