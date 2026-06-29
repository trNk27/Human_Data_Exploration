"""Streamlit UI for the Human Data analysis scripts.

Run from the project root:
    streamlit run ui/app.py
"""

import sys
import os
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from session import Session
from explore import grid
from compute import WINDOWS, WINDOW_LABELS, compute_aligned_raster
from utils import EVENTS, CONDITIONS, EVENT_STYLE

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLOT_TYPES = ["PSTH", "Aligned Raster", "Trial Raster", "ACG",
              "FR vs Perceived P", "FR vs HGF Perceived P",
              "FR vs Prediction Error"]

COND_CHOICES = [None, "G+R", "G+N", "S+R", "all"]


def cond_label(c):
    if c is None:
        return "All responding trials"
    if c == "all":
        return "Overlay 3 conditions"
    return CONDITIONS[c]["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def available_sessions():
    pat = re.compile(r"^\d{8}$")
    return sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and pat.match(d)
    )


@st.cache_resource
def load_session(session_id: str) -> Session:
    return Session(session_id, data_root=DATA_ROOT)


@st.cache_data
def cached_labels(session: str) -> list[str]:
    _, labels = utils.get_spike_trains(
        data_dir=os.path.join(DATA_ROOT, session)
    )
    return labels


def extract_areas(labels: list[str]) -> list[str]:
    areas = set()
    for lbl in labels:
        if "|" in lbl:
            part = lbl.split("|", 1)[1].strip()
            word = part.split()[0] if part else ""
            if word:
                areas.add(word)
    return sorted(areas)


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Trial raster with parameter strips
# ─────────────────────────────────────────────────────────────────────────────

def render_trial_raster_fig(sess, neuron_idx, align="cue", pre_ms=500, post_ms=1000,
                             by_condition=True, sort_keys=None, filter_conditions=None):
    """Aligned raster — one row per trial (responding only) — with decision / reward strips.

    sort_keys : ordered list of sort levels, each one of
                "Decision", "Reward", "Condition"  (primary first).
    """
    trains, labels = sess.spike_trains
    train = trains[neuron_idx]
    label = labels[neuron_idx]

    trials = sess.trials
    align_times_all = sess.event_times(align)
    n_total = len(trials)

    responding = (trials["NotResponding"] == 0).to_numpy()
    df = pd.DataFrame({
        "orig_idx":  np.arange(n_total),
        "decision":  trials["ChosenArm_G1S0"].fillna(0).astype(int).to_numpy(),
        "reward":    trials["Rewarded"].fillna(0).astype(int).to_numpy(),
        "align_time": align_times_all,
    })
    df = df[responding].copy()

    if filter_conditions:
        cond_masks = sess.condition_masks()
        keep = np.zeros(n_total, dtype=bool)
        for c in filter_conditions:
            keep |= cond_masks[c]
        df = df[keep[df["orig_idx"].to_numpy()]].copy()

    if sort_keys:
        # Map each key to a (column, ascending) pair; "Condition" gets a helper column.
        df["_cond_order"] = df.apply(
            lambda row: (
                0 if (row["decision"] == 1 and row["reward"] == 1) else   # G+R
                1 if (row["decision"] == 1 and row["reward"] == 0) else   # G+N
                2 if (row["decision"] == 0 and row["reward"] == 1) else   # S+R
                3
            ), axis=1,
        )
        col_map  = {"Decision": ("decision", False),
                    "Reward":   ("reward",   False),
                    "Condition": ("_cond_order", True)}
        cols = [col_map[k][0] for k in sort_keys if k in col_map]
        asc  = [col_map[k][1] for k in sort_keys if k in col_map]
        df = df.sort_values(cols, ascending=asc, kind="stable").drop(columns="_cond_order")

    df = df.reset_index(drop=True)
    spikes  = compute_aligned_raster(train, df["align_time"].to_numpy(), pre_ms, post_ms)
    markers = sess.marker_times_ms(align, pre_ms, post_ms)
    n       = len(df)

    row_h  = max(0.08, min(0.14, 12 / max(n, 1)))
    fig_h  = max(4, min(n * row_h + 1.5, 16))
    fig = plt.figure(figsize=(12, fig_h))
    gs  = fig.add_gridspec(1, 3, width_ratios=[8, 1, 1], wspace=0.03,
                           left=0.07, right=0.99, top=0.93, bottom=0.07)
    ax_raster = fig.add_subplot(gs[0])
    ax_dec    = fig.add_subplot(gs[1], sharey=ax_raster)
    ax_rew    = fig.add_subplot(gs[2], sharey=ax_raster)

    # ── Raster ──
    if by_condition:
        cond_masks_all = sess.condition_masks()
        orig_to_color: dict[int, str] = {}
        for cname, mask in cond_masks_all.items():
            for oi in np.where(mask)[0]:
                orig_to_color[oi] = CONDITIONS[cname]["color"]

    for row, sp in enumerate(spikes):
        if not len(sp):
            continue
        orig  = int(df.at[row, "orig_idx"])
        color = orig_to_color.get(orig, "gray") if by_condition else "black"
        ax_raster.eventplot([sp], lineoffsets=[row], colors=[color],
                            linelengths=0.8, linewidths=0.5)

    ax_raster.axvline(0, color="red", linewidth=1.0, linestyle="--",
                      label=f"{EVENTS[align]['label']} (align)")
    for name, t_ms in markers.items():
        ax_raster.axvline(t_ms, linewidth=0.8, label=EVENTS[name]["label"],
                          **EVENT_STYLE[name])
    ax_raster.legend(fontsize=6, loc="upper right")
    ax_raster.set_xlim(-pre_ms, post_ms)
    ax_raster.set_ylim(n - 0.5, -0.5)   # row 0 at top
    ax_raster.set_xlabel("Time rel. to event (ms)", fontsize=8)
    y_label = f"Trial (n={n})"
    if sort_keys:
        y_label += f",  sorted by {' → '.join(sort_keys)}"
    ax_raster.set_ylabel(y_label, fontsize=8)
    ax_raster.set_title(
        f"{label}  |  {sess.id}  |  align: {EVENTS[align]['label']}",
        fontsize=9,
    )

    # ── Parameter strips ──
    y_edges = np.arange(n + 1) - 0.5
    x_edges = np.array([0.0, 1.0])

    strip_specs = [
        (ax_dec, "decision", ["steelblue", "darkorange"], "Decision", "Safe | Gamble"),
        (ax_rew, "reward",   ["lightcoral", "seagreen"],  "Reward",   "No | Yes"),
    ]
    for ax, col, colors, title, xlabel in strip_specs:
        vals = df[col].to_numpy().reshape(n, 1).astype(float)
        ax.pcolormesh(x_edges, y_edges, vals,
                      cmap=ListedColormap(colors), vmin=-0.5, vmax=1.5)
        ax.set_xticks([])
        ax.tick_params(left=False, labelleft=False)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=5)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Human Data Explorer", layout="wide")
st.title("Human Data Explorer")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session")
    sessions = available_sessions()
    if not sessions:
        st.error("No session directories (YYYYMMDD) found.")
        st.stop()
    session = st.selectbox("Session", sessions, index=len(sessions) - 1)

    st.header("Neurons")
    labels = cached_labels(session)
    areas  = extract_areas(labels)

    sel_mode = st.radio("Select by", ["Area", "Neuron indices"])
    area_filter    = None
    neuron_indices = None

    if sel_mode == "Area":
        chosen_area = st.selectbox("Area", ["(all)"] + areas)
        if chosen_area != "(all)":
            area_filter = chosen_area
            n_match = sum(area_filter.lower() in lbl.lower() for lbl in labels)
            st.caption(f"{n_match} neuron(s) match")
    else:
        picked = st.multiselect(
            "Neurons",
            options=list(range(len(labels))),
            format_func=lambda i: f"{i}: {labels[i]}",
        )
        st.caption("Leave empty to show all (up to 90 neurons)")
        neuron_indices = picked if picked else None

    st.header("Plot type")
    plot_type = st.radio("Graph", PLOT_TYPES)

    st.header("Parameters")

    if plot_type == "PSTH":
        event    = st.selectbox("Align to", list(EVENTS.keys()))
        pre_ms   = st.slider("Pre-event (ms)",  100, 2000, 500,  50)
        post_ms  = st.slider("Post-event (ms)", 100, 2000, 1000, 50)
        bin_ms   = st.slider("Bin size (ms)",     5,  200,  50,   5)
        sig_raw  = st.slider("Smoothing sigma ms (0 = off)", 0, 200, 0, 5)
        sigma_ms = sig_raw if sig_raw > 0 else None
        condition = st.selectbox("Condition", COND_CHOICES, format_func=cond_label)

    elif plot_type == "Aligned Raster":
        event   = st.selectbox("Align to", list(EVENTS.keys()))
        pre_ms  = st.slider("Pre-event (ms)",  100, 2000, 500,  50)
        post_ms = st.slider("Post-event (ms)", 100, 2000, 1000, 50)
        condition = st.selectbox("Condition", COND_CHOICES, format_func=cond_label)

    elif plot_type == "Trial Raster":
        tr_neuron = st.selectbox(
            "Neuron",
            options=list(range(len(labels))),
            format_func=lambda i: f"{i}: {labels[i]}",
            key="tr_neuron",
        )
        tr_event    = st.selectbox("Align to", list(EVENTS.keys()), key="tr_event")
        tr_pre_ms   = st.slider("Pre-event (ms)",  100, 2000,  500, 50, key="tr_pre")
        tr_post_ms  = st.slider("Post-event (ms)", 100, 2000, 1000, 50, key="tr_post")
        tr_by_cond = st.checkbox("Colour by condition", value=True, key="tr_by_cond")
        st.caption("Sort levels (primary → secondary → …)")
        _sort_opts = ["None", "Decision", "Reward", "Condition"]
        tr_sort_1  = st.selectbox("Primary sort",   _sort_opts, key="tr_sort_1")
        tr_sort_2  = st.selectbox("Secondary sort", _sort_opts, key="tr_sort_2")
        tr_sort_3  = st.selectbox("Tertiary sort",  _sort_opts, key="tr_sort_3")
        tr_sort_keys = [s for s in [tr_sort_1, tr_sort_2, tr_sort_3] if s != "None"] or None
        tr_filter = st.multiselect(
            "Show conditions",
            list(CONDITIONS.keys()),
            default=list(CONDITIONS.keys()),
            format_func=lambda c: CONDITIONS[c]["label"],
            key="tr_filter",
        )
        tr_filter = tr_filter if tr_filter else None

    elif plot_type == "ACG":
        lag_ms = st.slider("Max lag (ms)",  50, 1000, 200, 10)
        bin_ms = st.slider("Bin size (ms)",  1,   20,   1,  1)

    elif plot_type == "FR vs Perceived P":
        fr_window = st.selectbox(
            "Firing-rate window", options=list(WINDOWS),
            format_func=lambda k: WINDOW_LABELS[k],
        )
        fr_history = st.slider("Reward history (trials)", 3, 30, 10,
                                help="Past responding trials used to estimate P(reward)")
        fr_bins    = st.slider("Probability bins", 4, 15, 8)
        fr_by_cond = st.checkbox("Colour by outcome (per-condition fit)", value=False,
                                  key="fr_by_cond")

    elif plot_type == "FR vs HGF Perceived P":
        hgf_window = st.selectbox(
            "Firing-rate window", options=list(WINDOWS),
            format_func=lambda k: WINDOW_LABELS[k], key="hgf_window",
        )
        hgf_bins    = st.slider("Probability bins", 4, 15, 8, key="hgf_bins")
        hgf_by_cond = st.checkbox("Colour by outcome (per-condition fit)", value=False,
                                   key="hgf_by_cond")
        st.caption(
            "x-axis: HGF latent belief P(gamble reward). "
            "Requires results/hgf/trajectory_<session>.csv."
        )

    elif plot_type == "FR vs Prediction Error":
        pe_window = st.selectbox(
            "Firing-rate window", options=list(WINDOWS),
            format_func=lambda k: WINDOW_LABELS[k], key="pe_window",
        )
        pe_bins    = st.slider("Bins (pooled overlay)", 4, 15, 8, key="pe_bins")
        pe_by_cond = st.checkbox("Colour by outcome (per-condition fit)", value=True,
                                  key="pe_by_cond")
        st.caption(
            "x-axis: HGF δ₁ = outcome − p̂. Gamble trials only. "
            "Requires results/hgf/trajectory_<session>.csv."
        )

    st.divider()
    run = st.button("Plot", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────

if "fig_bytes" not in st.session_state:
    st.session_state.fig_bytes = None
    st.session_state.fig_name  = "plot.png"

if run:
    plt.close("all")
    try:
        with st.spinner("Computing..."):
            sess = load_session(session)

            if plot_type == "PSTH":
                fig = grid(
                    sess, "psth",
                    neurons=neuron_indices, area=area_filter,
                    align=event, pre_ms=pre_ms, post_ms=post_ms,
                    bin_ms=bin_ms, sigma_ms=sigma_ms, condition=condition,
                ).fig

            elif plot_type == "Aligned Raster":
                fig = grid(
                    sess, "raster",
                    neurons=neuron_indices, area=area_filter,
                    align=event, pre_ms=pre_ms, post_ms=post_ms,
                    condition=condition,
                ).fig

            elif plot_type == "Trial Raster":
                fig = render_trial_raster_fig(
                    sess, tr_neuron,
                    align=tr_event,
                    pre_ms=tr_pre_ms,
                    post_ms=tr_post_ms,
                    by_condition=tr_by_cond,
                    sort_keys=tr_sort_keys,
                    filter_conditions=tr_filter,
                )

            elif plot_type == "ACG":
                fig = grid(
                    sess, "acg",
                    neurons=neuron_indices, area=area_filter,
                    lag_ms=lag_ms, bin_ms=bin_ms,
                ).fig

            elif plot_type == "FR vs Perceived P":
                fig = grid(
                    sess, "fr_vs_p",
                    neurons=neuron_indices, area=area_filter,
                    window=fr_window, history=fr_history, n_bins=fr_bins,
                    by_condition=fr_by_cond,
                ).fig

            elif plot_type == "FR vs HGF Perceived P":
                fig = grid(
                    sess, "fr_vs_hgf_p",
                    neurons=neuron_indices, area=area_filter,
                    window=hgf_window, n_bins=hgf_bins,
                    by_condition=hgf_by_cond,
                ).fig

            elif plot_type == "FR vs Prediction Error":
                fig = grid(
                    sess, "fr_vs_delta1",
                    neurons=neuron_indices, area=area_filter,
                    window=pe_window, n_bins=pe_bins,
                    by_condition=pe_by_cond,
                ).fig

        st.session_state.fig_bytes = fig_to_png(fig)
        st.session_state.fig_name  = (
            f"{plot_type.lower().replace(' ', '_')}_{session}.png"
        )
        plt.close(fig)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise

if st.session_state.fig_bytes:
    st.image(st.session_state.fig_bytes, use_column_width=True)
    st.download_button(
        "Download PNG",
        data=st.session_state.fig_bytes,
        file_name=st.session_state.fig_name,
        mime="image/png",
    )
