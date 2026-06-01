"""explore — the single-neuron, fluent plotting layer.

The atomic unit here is *one neuron on one axes*. Each `draw_*` renders a
single neuron onto a Matplotlib `ax`; everything else is built on top:

  - `NeuronView`  : returned by `Session.neuron(i)`; `.psth()/.raster()/.acg()/
                    .fr_vs_p()` each return a `Panel` for a single-neuron figure.
  - `Panel`       : thin (fig, ax) wrapper whose `.save()` auto-names into
                    `results/figures/<session>/`.
  - `grid()`      : tiles the same `draw_*` across many neurons — the one and
                    only copy of the subplot-grid scaffolding.

Typical use::

    from session import Session
    sess = Session("20250714")

    n = sess.neuron(7)
    n.psth(condition="G+R", align="reward").save()
    n.raster(condition="G+N", align="cue").save()
    sess.neuron(12).acg().save()

    # many neurons at once (same draw functions):
    from explore import grid
    grid(sess, "psth", area="ACC", align="reward").save("acc_psth.png")

`condition` (psth / raster): a CONDITIONS key ("G+R"/"G+N"/"S+R") filters
trials to that outcome; None = all responding trials; "all" = overlay the three.
All time parameters are in milliseconds; spike/event times stay in seconds.
"""

from __future__ import annotations

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from utils import EVENTS, EVENT_STYLE, CONDITIONS, select_neurons, RESULTS_DIR
from compute import (
    compute_psth, compute_aligned_raster, compute_acg,
    perceived_probability, trial_firing_rates, binned_stats,
    WINDOWS, WINDOW_LABELS,
)


# ---------------------------------------------------------------------------
# draw_* — one neuron on one axes. The single home for the plotting commands.
# Each takes the uniform signature (ax, sess, train, label, **kw) so `grid`
# can dispatch them through one table. Artists are labelled; the *caller*
# decides whether to draw a legend (once per grid vs. once per single panel).
# ---------------------------------------------------------------------------

def draw_psth(ax, sess, train, label, condition=None, align="cue",
              pre_ms=500, post_ms=1000, bin_ms=50, sigma_ms=None):
    """PSTH for one neuron, aligned to `align`, optionally split/filtered by condition."""
    if align not in EVENTS:
        raise ValueError(f"align must be one of {list(EVENTS)}")

    markers = sess.marker_times_ms(align, pre_ms, post_ms)

    if condition == "all":
        cond_masks = sess.condition_masks()
        align_all  = sess.event_times(align)
        for name, cfg in CONDITIONS.items():
            cond_align = align_all[cond_masks[name]]
            if cond_align.size == 0:
                continue
            centres, rate = compute_psth(train, cond_align, pre_ms, post_ms, bin_ms)
            n_tr = int(cond_masks[name].sum())
            if sigma_ms is not None:
                rate = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
                ax.plot(centres, rate, color=cfg["color"], linewidth=1.2,
                        label=f"{cfg['label']} (n={n_tr})")
            else:
                ax.step(centres, rate, where="mid", color=cfg["color"],
                        linewidth=1.0, label=f"{cfg['label']} (n={n_tr})")
    else:
        align_times = sess.aligned_event_times(align, condition)
        centres, rate = compute_psth(train, align_times, pre_ms, post_ms, bin_ms)
        color = CONDITIONS[condition]["color"] if condition else "steelblue"
        ax.bar(centres, rate, width=bin_ms, color=color, edgecolor="none",
               alpha=0.6, label="_nolegend_")
        if sigma_ms is not None:
            smoothed = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
            ax.plot(centres, smoothed, color="navy", linewidth=1.2, label="_nolegend_")

    ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
               label=f"{EVENTS[align]['label']} (align)")
    for name, t_rel_ms in markers.items():
        ax.axvline(t_rel_ms, linewidth=0.8, label=EVENTS[name]["label"], **EVENT_STYLE[name])

    ax.set_title(label, fontsize=7)
    ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
    ax.set_ylabel("Firing rate (Hz)", fontsize=7)
    ax.tick_params(labelsize=6)


def draw_raster(ax, sess, train, label, condition=None, align="cue",
                pre_ms=500, post_ms=1000):
    """Aligned raster for one neuron (spikes from all trials overlaid)."""
    if align not in EVENTS:
        raise ValueError(f"align must be one of {list(EVENTS)}")

    markers = sess.marker_times_ms(align, pre_ms, post_ms)

    if condition == "all":
        cond_masks = sess.condition_masks()
        align_all  = sess.event_times(align)
        for name, cfg in CONDITIONS.items():
            cond_align  = align_all[cond_masks[name]]
            cond_spikes = compute_aligned_raster(train, cond_align, pre_ms, post_ms)
            flat = np.concatenate(cond_spikes) if cond_spikes else np.array([])
            if len(flat):
                ax.eventplot([flat], lineoffsets=[0], colors=[cfg["color"]],
                             linelengths=0.8, linewidths=0.5, alpha=0.1,
                             label=cfg["label"])
    else:
        align_times = sess.aligned_event_times(align, condition)
        spikes = compute_aligned_raster(train, align_times, pre_ms, post_ms)
        flat   = np.concatenate(spikes) if spikes else np.array([])
        color  = CONDITIONS[condition]["color"] if condition else "black"
        if len(flat):
            ax.eventplot([flat], lineoffsets=[0], colors=[color],
                         linelengths=0.8, linewidths=0.5, alpha=0.1)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
               label=f"{EVENTS[align]['label']} (align)")
    for name, t_rel_ms in markers.items():
        ax.axvline(t_rel_ms, linewidth=0.8, label=EVENTS[name]["label"], **EVENT_STYLE[name])

    ax.set_xlim(-pre_ms, post_ms)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Time rel. to event (ms)", fontsize=7)
    ax.set_title(label, fontsize=7)
    ax.tick_params(labelsize=6)


def draw_acg(ax, sess, train, label, lag_ms=200, bin_ms=1):
    """Autocorrelogram for one neuron. (`sess` unused; kept for uniform dispatch.)"""
    centres, counts = compute_acg(train, lag_ms=lag_ms, bin_ms=bin_ms)
    ax.bar(centres, counts, width=bin_ms, color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=7)
    ax.set_xlabel("Lag (ms)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6)


def draw_fr_vs_p(ax, sess, train, label, window="trial", history=10, n_bins=8):
    """Firing rate vs perceived P(reward) for one neuron (gamble trials only)."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")

    gamble_mask = sess.responding_mask & (sess.trials["ChosenArm_G1S0"].to_numpy() == 1)
    perc_prob   = perceived_probability(sess.trials, sess.responding_mask, history=history)
    rates       = trial_firing_rates([train], sess.trials, sess.sampling_rate,
                                     window=window, trial_mask=gamble_mask)

    x  = perc_prob
    y  = rates[:, 0]
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() >= 4:
        xv, yv = x[ok], y[ok]
        ax.scatter(xv, yv, s=9, alpha=0.25, color="steelblue",
                   linewidths=0, rasterized=True, zorder=2)
        cx, mn, se = binned_stats(xv, yv, n_bins)
        if cx.size > 0:
            ax.errorbar(cx, mn, yerr=se, fmt="o-", color="navy", markersize=4,
                        linewidth=1.4, capsize=3, label="Bin mean ± SEM", zorder=5)
        slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
        x_line = np.array([0.0, 1.0])
        p_str  = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        ax.plot(x_line, intercept + slope * x_line, color="firebrick",
                linewidth=1.2, linestyle="--", zorder=4,
                label=f"r = {r_val:.2f},  p = {p_str}")
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.8)
    else:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="gray")

    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Perceived P(reward)", fontsize=7)
    ax.set_ylabel("Firing rate (Hz)",    fontsize=7)
    ax.set_title(label, fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


# Dispatch tables — the single source of truth for "what kinds exist".
_DRAW = {
    "psth":    draw_psth,
    "raster":  draw_raster,
    "acg":     draw_acg,
    "fr_vs_p": draw_fr_vs_p,
}
_GRID_FIGSIZE = {
    "psth":    (4.0, 3.0),
    "raster":  (5.0, 2.0),
    "acg":     (4.0, 3.0),
    "fr_vs_p": (4.5, 3.5),
}
_PANEL_FIGSIZE = {
    "psth":    (7.0, 3.0),
    "raster":  (7.0, 3.0),
    "acg":     (7.0, 3.0),
    "fr_vs_p": (6.5, 4.0),
}
# Plots whose legend is identical on every axis -> draw it once in a grid.
_LEGEND_SHARED = {"psth", "raster"}
_KIND_TITLE = {
    "psth":    "PSTH",
    "raster":  "Aligned raster",
    "acg":     "Autocorrelograms",
    "fr_vs_p": "FR vs perceived P(reward)",
}


# ---------------------------------------------------------------------------
# Panel — a (fig, ax) wrapper that knows how to save itself.
# ---------------------------------------------------------------------------

class Panel:
    """A finished figure. `.save()` auto-names into results/figures/<session>/."""

    def __init__(self, fig, ax_or_axes, autoname=None):
        self.fig = fig
        self._ax = ax_or_axes
        self._autoname = autoname   # (session_id, [name_parts]) or None

    @property
    def ax(self):
        return self._ax

    def save(self, path: str | None = None, dpi: int = 150) -> str:
        """Save the figure. With no path, auto-names into results/figures/<session>/."""
        if path is None:
            if self._autoname is None:
                raise ValueError("No auto-name available — pass an explicit path to save().")
            session, parts = self._autoname
            name = "_".join(str(p) for p in parts if p) + ".png"
            path = os.path.join(RESULTS_DIR, "figures", session, name)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved -> {path}")
        return path

    def show(self) -> "Panel":
        plt.show()
        return self

    def close(self) -> None:
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# NeuronView — fluent single-neuron entry point (Session.neuron(i)).
# ---------------------------------------------------------------------------

def _cond_tag(condition) -> str:
    if condition is None:
        return "alltrials"
    if condition == "all":
        return "bycond"
    return condition


class NeuronView:
    """One neuron of one session. Methods return a `Panel` for a single figure."""

    def __init__(self, sess, idx: int):
        trains, labels = sess.spike_trains
        if not 0 <= idx < len(trains):
            raise IndexError(f"neuron {idx} out of range (0..{len(trains) - 1})")
        self.sess  = sess
        self.idx   = idx
        self.train = trains[idx]
        self.label = labels[idx]

    def _panel(self, kind: str, draw_kw: dict, name_parts: list) -> Panel:
        fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE[kind])
        _DRAW[kind](ax, self.sess, self.train, self.label, **draw_kw)
        if kind in _LEGEND_SHARED:
            ax.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"neuron {self.idx}  ·  session {self.sess.id}", fontsize=9)
        fig.tight_layout()
        return Panel(fig, ax, autoname=(self.sess.id, name_parts))

    def psth(self, condition=None, align="cue", pre_ms=500, post_ms=1000,
             bin_ms=50, sigma_ms=None) -> Panel:
        return self._panel(
            "psth",
            dict(condition=condition, align=align, pre_ms=pre_ms,
                 post_ms=post_ms, bin_ms=bin_ms, sigma_ms=sigma_ms),
            ["psth", f"n{self.idx}", _cond_tag(condition), align],
        )

    def raster(self, condition=None, align="cue", pre_ms=500, post_ms=1000) -> Panel:
        return self._panel(
            "raster",
            dict(condition=condition, align=align, pre_ms=pre_ms, post_ms=post_ms),
            ["raster", f"n{self.idx}", _cond_tag(condition), align],
        )

    def acg(self, lag_ms=200, bin_ms=1) -> Panel:
        return self._panel(
            "acg",
            dict(lag_ms=lag_ms, bin_ms=bin_ms),
            ["acg", f"n{self.idx}"],
        )

    def fr_vs_p(self, window="trial", history=10, n_bins=8) -> Panel:
        return self._panel(
            "fr_vs_p",
            dict(window=window, history=history, n_bins=n_bins),
            ["fr_vs_p", f"n{self.idx}", window],
        )

    def __repr__(self) -> str:
        return f"NeuronView(session={self.sess.id!r}, idx={self.idx}, label={self.label!r})"


# ---------------------------------------------------------------------------
# grid — tile a draw_* across many neurons. The ONLY grid-scaffolding copy.
# ---------------------------------------------------------------------------

def grid(sess, kind: str, neurons=None, area=None, ncols: int = 4, **kw) -> Panel:
    """Tile single-neuron `kind` panels across a subplot grid.

    `kind` is one of {"psth", "raster", "acg", "fr_vs_p"}. `neurons`/`area`
    select the units (see utils.select_neurons); remaining `**kw` are forwarded
    to the matching `draw_*`.
    """
    if kind not in _DRAW:
        raise ValueError(f"kind must be one of {list(_DRAW)}")

    trains, labels = sess.spike_trains
    trains, labels = select_neurons(trains, labels, indices=neurons, area=area)
    draw = _DRAW[kind]

    n     = len(trains)
    ncols = min(n, ncols)
    nrows = math.ceil(n / ncols)
    w, h  = _GRID_FIGSIZE[kind]
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False)

    for idx in range(n):
        draw(axes[idx // ncols][idx % ncols], sess, trains[idx], labels[idx], **kw)

    if kind in _LEGEND_SHARED:
        axes[0][0].legend(fontsize=5, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(_grid_title(sess, kind, n, kw), fontsize=9)
    fig.tight_layout()
    return Panel(fig, axes)


def _grid_title(sess, kind: str, n: int, kw: dict) -> str:
    parts = [f"{_KIND_TITLE[kind]} — session {sess.id}", f"{n} neuron(s)"]
    if kw.get("align"):
        parts.append(f"aligned to: {kw['align']}")
    if kw.get("condition"):
        parts.append(f"condition: {kw['condition']}")
    if kw.get("window"):
        parts.append(WINDOW_LABELS.get(kw["window"], kw["window"]))
    return "   |   ".join(parts)
