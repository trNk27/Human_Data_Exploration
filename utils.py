"""Shared data-access utilities for the Human Data analysis scripts.

Data loaders, spike-train extraction, and common constants used across
raster_plot.py, psth.py, and autocorrelogram.py.
"""

import os
import scipy.io
import numpy as np
import pandas as pd

SESSION  = "20250707"
DATA_DIR = os.path.join(os.path.dirname(__file__), SESSION)

MAX_NEURONS = 90  # hard cap shared by psth and autocorrelogram


# ---------------------------------------------------------------------------
# .mat loaders
# ---------------------------------------------------------------------------

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
    # Columns 1-14: behavioural (seconds). Columns 15-19: timing in sampling points.
    # ChosenSide (col 12) is unreliable per README — use ChosenArm (col 13) instead.
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
# Higher-level helpers
# ---------------------------------------------------------------------------

def get_spike_trains(data_dir=None):
    """Return (trains, labels): list of 1-D spike-time arrays (seconds) and column labels."""
    df = load_stmtx(data_dir=data_dir)
    trains = [df[col].dropna().to_numpy() for col in df.columns]
    return trains, list(df.columns)


def sp_to_s(trials, sr, col):
    """Convert a sampling-point column in a Trials_Sync DataFrame to seconds."""
    return trials[col].to_numpy() / sr
