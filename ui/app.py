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
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from psth import plot_psth, EVENTS
from raster_plot import plot_raster, plot_aligned_raster
from autocorrelogram import plot_acg

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def available_sessions():
    pat = re.compile(r"^\d{8}$")
    return sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and pat.match(d)
    )


@st.cache_data
def cached_labels(session):
    """Load neuron labels for a session (cached per session)."""
    _, labels = utils.get_spike_trains(data_dir=os.path.join(DATA_ROOT, session))
    return labels


def extract_areas(labels):
    areas = set()
    for lbl in labels:
        if "|" in lbl:
            part = lbl.split("|", 1)[1].strip()
            word = part.split()[0] if part else ""
            if word:
                areas.add(word)
    return sorted(areas)


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Human Data Explorer", layout="wide")
st.title("Human Data Explorer")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Session
    st.header("Session")
    sessions = available_sessions()
    if not sessions:
        st.error("No session directories (YYYYMMDD) found.")
        st.stop()
    session = st.selectbox("Session", sessions, index=len(sessions) - 1)

    # Point utils globals at the selected session.
    # DATA_DIR is referenced inside loader functions via the utils module namespace,
    # so this update is seen on the next call even for names imported with 'from'.
    utils.SESSION  = session
    utils.DATA_DIR = os.path.join(DATA_ROOT, session)

    # Neuron selection
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

    # Plot type
    st.header("Plot type")
    plot_type = st.radio("Graph", ["PSTH", "Raster", "Aligned Raster", "ACG"])

    # Parameters
    st.header("Parameters")

    if plot_type == "PSTH":
        event    = st.selectbox("Align to", list(EVENTS.keys()))
        pre_ms   = st.slider("Pre-event (ms)",  100, 2000, 500,  50)
        post_ms  = st.slider("Post-event (ms)", 100, 2000, 1000, 50)
        bin_ms   = st.slider("Bin size (ms)",     5,  200,  50,   5)
        sig_raw  = st.slider("Smoothing σ ms (0 = off)", 0, 200, 0, 5)
        sigma_ms = sig_raw if sig_raw > 0 else None
        by_cond  = st.checkbox("Split by condition (arm × reward)")

    elif plot_type == "Raster":
        use_win = st.checkbox("Limit time window")
        if use_win:
            t_start = st.number_input("Start (s)", value=0.0,   step=1.0, format="%.1f")
            t_end   = st.number_input("End (s)",   value=100.0, step=1.0, format="%.1f")
        else:
            t_start = t_end = None

    elif plot_type == "Aligned Raster":
        event    = st.selectbox("Align to", list(EVENTS.keys()))
        pre_ms   = st.slider("Pre-event (ms)",  100, 2000, 500,  50)
        post_ms  = st.slider("Post-event (ms)", 100, 2000, 1000, 50)
        by_cond  = st.checkbox("Colour by condition (arm × reward)")

    elif plot_type == "ACG":
        lag_ms = st.slider("Max lag (ms)",  50, 1000, 200, 10)
        bin_ms = st.slider("Bin size (ms)",  1,   20,   1,  1)

    st.divider()
    run = st.button("Plot", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────

if "fig_bytes" not in st.session_state:
    st.session_state.fig_bytes = None
    st.session_state.fig_name  = "plot.png"

if run:
    plt.close("all")
    try:
        with st.spinner("Computing…"):
            if plot_type == "PSTH":
                fig, _ = plot_psth(
                    neuron_indices=neuron_indices,
                    area=area_filter,
                    event=event,
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    bin_ms=bin_ms,
                    sigma_ms=sigma_ms,
                    by_condition=by_cond,
                )
            elif plot_type == "Raster":
                fig, _ = plot_raster(
                    t_start=t_start,
                    t_end=t_end,
                    neuron_indices=neuron_indices,
                    area=area_filter,
                )
            elif plot_type == "Aligned Raster":
                fig, _ = plot_aligned_raster(
                    neuron_indices=neuron_indices,
                    area=area_filter,
                    event=event,
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    by_condition=by_cond,
                )
            elif plot_type == "ACG":
                fig, _ = plot_acg(
                    neuron_indices=neuron_indices,
                    area=area_filter,
                    lag_ms=lag_ms,
                    bin_ms=bin_ms,
                )
        st.session_state.fig_bytes = fig_to_png(fig)
        st.session_state.fig_name  = f"{plot_type.lower().replace(' ', '_')}_{session}.png"
        plt.close(fig)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise

if st.session_state.fig_bytes:
    st.image(st.session_state.fig_bytes, use_column_width=True)
    st.download_button(
        "⬇ Download PNG",
        data=st.session_state.fig_bytes,
        file_name=st.session_state.fig_name,
        mime="image/png",
    )
