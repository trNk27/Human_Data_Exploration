"""Shared data-access utilities for the Human Data analysis scripts.

Single source of truth for:
  - the active session (overridable via --session everywhere)
  - .mat loaders (load_sr, load_stmtx, load_trials_sync)
  - spike-train + event-time helpers (get_spike_trains, sp_to_s,
    event_times, condition_masks, condition_event_times)
  - behavioural events (EVENTS, EVENT_STYLE) and outcome conditions (CONDITIONS)
  - the standard CLI flags (--session, --neurons, --area, --list, --save)
    and the neuron-selection logic that goes with them
  - output directory conventions (RESULTS_DIR, RESULTS_SUBDIRS)
"""

import argparse
import os

import scipy.io
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Kept for CLI defaults. New code: Session(session_id) in analysis scripts.
SESSION  = "20250714"
DATA_DIR = os.path.join(REPO_ROOT, SESSION)

MAX_NEURONS = 90        # per-figure cap shared by all plotting scripts

# Where the analysis pipelines write their CSVs and PNGs.
RESULTS_DIR     = os.path.join(REPO_ROOT, "results")
RESULTS_SUBDIRS = {
    "responsiveness": os.path.join(RESULTS_DIR, "zeta_responsiveness"),
    "outcome":        os.path.join(RESULTS_DIR, "zeta_outcome"),
    "direction":      os.path.join(RESULTS_DIR, "direction"),
}


# ---------------------------------------------------------------------------
# Behavioural events  —  one canonical dict, keyed identically everywhere
# ---------------------------------------------------------------------------

# col: Trials_Sync column name; unit: "sp" (sampling points -> divide by SR)
#      or "s" (already seconds); label: display name.
EVENTS = {
    "cue":         {"col": "CuePresent_sp",      "unit": "sp", "label": "Cue"},
    "response":    {"col": "RespWindowStart_sp", "unit": "sp", "label": "Resp. window"},
    "reward":      {"col": "RewardOnset_sp",     "unit": "sp", "label": "Reward"},
    "trial_start": {"col": "TrialStart_sp",      "unit": "sp", "label": "Trial start"},
}

# Plotting styles for the same events (parallel to EVENTS, keyed the same).
EVENT_STYLE = {
    "cue":         dict(color="royalblue",  linestyle="--"),
    "response":    dict(color="green",      linestyle=":"),
    "reward":      dict(color="darkorange", linestyle="-."),
    "trial_start": dict(color="gray",       linestyle="--"),
}


# ---------------------------------------------------------------------------
# Outcome conditions  —  the three trial types of interest
# (safe + no-reward exists but is ignored everywhere)
# ---------------------------------------------------------------------------

CONDITIONS = {
    "G+R": dict(arm=1, rewarded=1, color="darkorange", label="Gamble + Rew"),
    "G+N": dict(arm=1, rewarded=0, color="firebrick",  label="Gamble + No"),
    "S+R": dict(arm=0, rewarded=1, color="seagreen",   label="Safe + Rew"),
}


# ---------------------------------------------------------------------------
# .mat loaders  (each takes optional data_dir; default = the active session)
# ---------------------------------------------------------------------------

def session_data_dir(session=None):
    """Absolute path to a session directory; falls back to the default SESSION."""
    return os.path.join(REPO_ROOT, session or SESSION)


def load_sr(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data = scipy.io.loadmat(os.path.join(data_dir, "SR.mat"))
    sr = int(data["SR"].flat[0])
    return pd.DataFrame({"SamplingRate_Hz": [sr]})


def load_stmtx(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data   = scipy.io.loadmat(os.path.join(data_dir, "STMtx.mat"))
    matrix = data["STMtx"]    # (max_spikes, nNeurons), spike times in seconds, NaN-padded
    info   = data["infoCell"] # (nNeurons, 4): area, electrode, unit, type

    def clean(cell):
        return str(cell).strip().strip("[]'\"")

    cols = [
        f"{clean(info[i,2])} | {clean(info[i,0])} {clean(info[i,1])} ({clean(info[i,3])})"
        for i in range(info.shape[0])
    ]
    return pd.DataFrame(matrix, columns=cols)


def load_trials_sync(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data   = scipy.io.loadmat(os.path.join(data_dir, "Trials_Sync.mat"))
    matrix = data["Trials_Sync"]  # (nTrials, 19)
    col_names = [
        "TrialStart_s", "TrialEnd_s", "TrialDuration_s", "Block",
        "GambleSide_R1L0", "P_BigReward_Gamble", "P_SmallReward_Safe",
        "Amount_BigReward_Gamble", "Amount_SmallReward_Safe",
        "PriorWheelNotStopping", "NotResponding",
        "ChosenSide_unreliable", "ChosenArm_G1S0", "Rewarded",
        "TrialStart_sp", "CuePresent_sp", "RespWindowStart_sp",
        "RewardOnset_sp", "TrialEnd_sp",
    ]
    return pd.DataFrame(matrix, columns=col_names)


# ---------------------------------------------------------------------------
# Spike-train and event helpers
# ---------------------------------------------------------------------------

def get_spike_trains(data_dir=None):
    """Return (trains, labels): list of 1-D spike-time arrays (seconds) and column labels."""
    df = load_stmtx(data_dir=data_dir)
    trains = [df[col].dropna().to_numpy() for col in df.columns]
    return trains, list(df.columns)


def sp_to_s(trials, sr, col):
    """Convert a sampling-point column in a Trials_Sync DataFrame to seconds."""
    return trials[col].to_numpy() / sr


def event_times(trials, sr, event):
    """Return event times (seconds) for the named EVENT key (one per trial)."""
    cfg = EVENTS[event]
    if cfg["unit"] == "sp":
        return sp_to_s(trials, sr, cfg["col"])
    return trials[cfg["col"]].to_numpy()


def condition_masks(trials):
    """Return {condition_name: boolean mask} for the three outcome conditions.

    Non-responding trials are excluded from every mask.
    """
    responding = (trials["NotResponding"] == 0).to_numpy()
    arm        = trials["ChosenArm_G1S0"].to_numpy()
    rew        = trials["Rewarded"].to_numpy()
    return {
        name: responding & (arm == cfg["arm"]) & (rew == cfg["rewarded"])
        for name, cfg in CONDITIONS.items()
    }


def condition_event_times(trials, sr, event="reward"):
    """Per-condition event times (seconds), responding trials only.

    Default event is 'reward' (reward-onset alignment) — matches the
    two-sample ZETA analysis. Pass `event` to align to another event.
    """
    times = event_times(trials, sr, event)
    return {name: times[mask] for name, mask in condition_masks(trials).items()}


# ---------------------------------------------------------------------------
# CLI helpers — every script uses these so the flags stay consistent
# ---------------------------------------------------------------------------

def add_session_arg(parser):
    """Add the standard --session flag (default: utils.SESSION)."""
    parser.add_argument(
        "--session", default=SESSION, metavar="YYYYMMDD",
        help=f"Session to load (default: {SESSION}).",
    )


def add_selection_args(parser):
    """Add the standard --neurons / --area / --list flags."""
    parser.add_argument(
        "--neurons", nargs="+", type=int, default=None,
        help="Neuron indices to show, e.g. --neurons 0 1 5",
    )
    parser.add_argument(
        "--area", type=str, default=None,
        help="Show only neurons whose label contains this string (case-insensitive)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print neuron indices and labels, then exit",
    )


def add_save_arg(parser):
    """Add --save [FILE] to an argparse parser."""
    parser.add_argument(
        "--save", metavar="FILE", nargs="?", const="",
        help="Save figure to FILE (auto-named <prefix>_<session>.png if no path given)",
    )


def select_neurons(trains, labels, indices=None, area=None, enforce_cap=True):
    """Filter (trains, labels) by index list and/or area substring.

    Enforces MAX_NEURONS unless `enforce_cap=False` (e.g. full-recording rasters
    where the whole population fits without per-neuron subplots).
    """
    if indices is not None:
        trains = [trains[i] for i in indices]
        labels = [labels[i] for i in indices]
    if area is not None:
        mask   = [area.lower() in lbl.lower() for lbl in labels]
        trains = [t for t, m in zip(trains, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    if not trains:
        raise ValueError("No neurons match the given selection.")
    if enforce_cap and len(trains) > MAX_NEURONS:
        raise ValueError(
            f"{len(trains)} neurons selected — limit is {MAX_NEURONS}. "
            "Use --neurons or --area to narrow the selection."
        )
    return trains, labels


def handle_list(args, data_dir=None):
    """If --list was passed, print every neuron index + label and return True.

    Callers should exit when this returns True.
    """
    if not getattr(args, "list", False):
        return False
    _, labels = get_spike_trains(data_dir=data_dir)
    for i, lbl in enumerate(labels):
        print(f"{i:4d}  {lbl}")
    return True


def maybe_save(fig, args, prefix="plot", subdir=None):
    """Save the figure when --save was passed.

    `args.save is None`   : do nothing (flag not passed).
    `args.save == ""`     : auto-name as `<prefix>_<session>.png`, drop into
                            `subdir` if given (created on demand) else cwd.
    `args.save == "path"` : save to exactly that path.
    """
    if args.save is None:
        return
    if args.save:
        path = args.save
    else:
        session = getattr(args, "session", None) or SESSION
        name    = f"{prefix}_{session}.png"
        if subdir:
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, name)
        else:
            path = name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
