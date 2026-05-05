"""Generate synthetic session data for testing raster_plot.py and file_explorer.py.

Creates test_session/ with SR.mat, STMtx.mat, and a minimal Trials_Sync.mat.
To use: set SESSION = "test_session" in file_explorer.py, then run either script.

Neurons generated:
  0  regular_5Hz   — 5 Hz regular spikes,  0–30 s
  1  random_10Hz   — Poisson ~10 Hz spikes, 30–60 s
  2  mixed         — 5 Hz in 0–30 s + Poisson in 30–60 s
"""

import os
import numpy as np
import scipy.io

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_session")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

SR = 30_000  # Hz

# --- Spike trains (seconds) ---
regular_5hz   = np.arange(0, 30, 1 / 5)                           # 150 spikes, 0–30 s
random_30_60  = np.sort(rng.uniform(30, 60, size=300))             # ~300 random spikes, 30–60 s
mixed         = np.sort(np.concatenate([regular_5hz,
                                        rng.uniform(30, 60, size=150)]))

trains = [regular_5hz, random_30_60, mixed]

# Pad columns with NaN to uniform length
max_spikes = max(len(t) for t in trains)
STMtx = np.full((max_spikes, len(trains)), np.nan)
for i, t in enumerate(trains):
    STMtx[: len(t), i] = t

# infoCell: (n_neurons, 4) object array — area, electrode, unit, type
infoCell = np.empty((len(trains), 4), dtype=object)
infoCell[0] = ["TestArea", "ele1", "regular_5Hz", "su"]
infoCell[1] = ["TestArea", "ele2", "random_10Hz", "su"]
infoCell[2] = ["TestArea", "ele3", "mixed",       "mu"]

# --- Save SR.mat ---
scipy.io.savemat(os.path.join(OUT_DIR, "SR.mat"), {"SR": np.array([[SR]], dtype=float)})

# --- Save STMtx.mat ---
scipy.io.savemat(os.path.join(OUT_DIR, "STMtx.mat"), {"STMtx": STMtx, "infoCell": infoCell})

# --- Save minimal Trials_Sync.mat (1 dummy trial so the loader doesn't crash) ---
dummy_trials = np.zeros((1, 19))
dummy_trials[0, 0] = 0.0   # TrialStart_s
dummy_trials[0, 1] = 60.0  # TrialEnd_s
scipy.io.savemat(os.path.join(OUT_DIR, "Trials_Sync.mat"), {"Trials_Sync": dummy_trials})

print(f"Saved to {OUT_DIR}/")
print(f"  SR.mat          SR = {SR} Hz")
print(f"  STMtx.mat       {STMtx.shape[1]} neurons, {STMtx.shape[0]} rows (NaN-padded)")
for i, (name, t) in enumerate(zip(infoCell[:, 2], trains)):
    print(f"    neuron {i}  {name:15s}  {len(t):4d} spikes")
print(f"  Trials_Sync.mat 1 dummy trial")
print()
print('To inspect: set SESSION = "test_session" in file_explorer.py')
