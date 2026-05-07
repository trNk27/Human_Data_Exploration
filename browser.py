"""Interactive neuron browser.

Shows a PSTH (top) and autocorrelogram (bottom) for one neuron at a time.
Navigate with Prev / Next buttons, arrow keys, or type a neuron index directly.

All time parameters are in milliseconds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from utils import SESSION, get_spike_trains, load_trials_sync, load_sr, sp_to_s
from psth import EVENTS, EVENT_STYLE, compute_psth
from autocorrelogram import compute_acg

# Defaults — override via CLI
_PRE_MS     = 500
_POST_MS    = 1000
_BIN_MS     = 50
_LAG_MS     = 200
_BIN_ACG_MS = 1


def build_browser(neuron_indices=None, area=None, event="cue",
                  pre_ms=_PRE_MS, post_ms=_POST_MS, bin_ms=_BIN_MS,
                  lag_ms=_LAG_MS, bin_acg_ms=_BIN_ACG_MS):
    trains, labels = get_spike_trains()

    if neuron_indices is not None:
        trains = [trains[i] for i in neuron_indices]
        labels = [labels[i] for i in neuron_indices]
    if area is not None:
        mask   = [area.lower() in lbl.lower() for lbl in labels]
        trains = [t for t, m in zip(trains, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    if not trains:
        raise ValueError("No neurons match the given selection.")

    trials      = load_trials_sync()
    sr          = load_sr()["SamplingRate_Hz"].iloc[0]
    align_times = sp_to_s(trials, sr, EVENTS[event])

    # Mean relative timing of other events in ms for PSTH marker lines.
    markers = {}
    for name, col in EVENTS.items():
        if name == event:
            continue
        rel = sp_to_s(trials, sr, col) - align_times
        if np.any(np.isfinite(rel)):
            mean_rel_ms = float(np.nanmean(rel)) * 1000
            if -pre_ms <= mean_rel_ms <= post_ms:
                markers[name] = mean_rel_ms

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

        train = trains[idx]

        # PSTH — centres and marker positions are in ms
        centres, rate = compute_psth(train, align_times, pre_ms, post_ms, bin_ms)
        ax_psth.bar(centres, rate, width=bin_ms, color="steelblue", edgecolor="none", alpha=0.6)
        ax_psth.axvline(0, color="red", linewidth=1.0, linestyle="--",
                        label=f"{EVENT_STYLE[event]['label']} (align)")
        for name, t_rel_ms in markers.items():
            ax_psth.axvline(t_rel_ms, linewidth=0.8, **EVENT_STYLE[name])
        ax_psth.set_xlabel("Time rel. to event (ms)", fontsize=8)
        ax_psth.set_ylabel("Firing rate (Hz)", fontsize=8)
        ax_psth.set_title(f"PSTH — aligned to: {event}", fontsize=8)
        ax_psth.legend(fontsize=6, loc="upper right")
        ax_psth.tick_params(labelsize=7)

        # ACG
        c_acg, cnt = compute_acg(train, lag_ms=lag_ms, bin_ms=bin_acg_ms)
        ax_acg.bar(c_acg, cnt, width=bin_acg_ms, color="steelblue", edgecolor="none")
        ax_acg.axvline(0, color="red", linewidth=0.8, linestyle="--")
        ax_acg.set_xlabel("Lag (ms)", fontsize=8)
        ax_acg.set_ylabel("Count", fontsize=8)
        ax_acg.set_title(f"Autocorrelogram (±{lag_ms} ms, bin {bin_acg_ms} ms)", fontsize=8)
        ax_acg.tick_params(labelsize=7)

        fig.suptitle(
            f"[{idx} / {len(trains) - 1}]  {labels[idx]}\nSession {SESSION}",
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

    # Keep widget references alive — without this, GC drops buttons and kills callbacks.
    fig._widgets = [btn_prev, btn_next, txt_box]

    draw(0)
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive neuron browser")
    parser.add_argument("--event",   type=str,   default="cue", choices=list(EVENTS),
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
    parser.add_argument("--neurons", nargs="+",  type=int, default=None,
                        help="Restrict to these neuron indices")
    parser.add_argument("--area",    type=str,   default=None,
                        help="Filter by area label substring")
    args = parser.parse_args()

    build_browser(
        neuron_indices=args.neurons,
        area=args.area,
        event=args.event,
        pre_ms=args.pre,
        post_ms=args.post,
        bin_ms=args.bin,
        lag_ms=args.lag,
        bin_acg_ms=args.bin_acg,
    )
    plt.show()
