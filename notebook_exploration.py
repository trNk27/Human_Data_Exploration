# %% [markdown]
# # Exploration notebook
# **Morning routine:** run cells 1–3 to load the session and get oriented.
# Everything else is pick-and-run exploration / development cells.

# %% ------------------------------------------------------------------
# 1. Imports — run once
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from session import Session
from utils import EVENTS, EVENT_STYLE, select_neurons, REPO_ROOT
from psth import compute_psth, plot_psth
from raster_plot import plot_raster, plot_aligned_raster
from browser import build_browser
from autocorrelogram import compute_acg

# %% ------------------------------------------------------------------
# 2. Load session — change SESSION_ID to switch
# ---------------------------------------------------------------------
SESSION_ID = "20250602"
sess = Session(SESSION_ID)

trains, labels = sess.spike_trains   # triggers loading
print(f"Session  : {sess.id}")
print(f"Neurons  : {len(trains)}")
print(f"Trials   : {len(sess.trials)}")
print(f"SR       : {sess.sampling_rate} Hz")

# %% ------------------------------------------------------------------
# 3. Morning summary — responding trials and condition breakdown
# ---------------------------------------------------------------------
mask = sess.responding_mask
print(f"Responding : {mask.sum()} / {len(mask)} trials  "
      f"({100 * mask.mean():.1f} %)\n")
for name, times in sess.condition_event_times().items():
    print(f"  {name:5s}  {len(times):4d} trials")

# %% [markdown]
# ---
# ## Whole-population views

# %% PSTH — all neurons, aligned to cue
fig, _ = plot_psth(sess, event="cue", pre_ms=500, post_ms=1500, bin_ms=50,
                   sigma_ms=80)
plt.show()

# %% PSTH — split by condition
fig, _ = plot_psth(sess, event="reward", pre_ms=500, post_ms=1500,
                   bin_ms=50, sigma_ms=80, by_condition=True)
plt.show()

# %% Aligned raster — colour-coded by (arm × outcome)
fig, _ = plot_aligned_raster(sess, event="cue", pre_ms=500, post_ms=1500,
                              by_condition=True)
plt.show()

# %% Full-recording raster (zoomed to first 300 s)
fig, _ = plot_raster(sess, t_start=0, t_end=300)
plt.show()

# %% [markdown]
# ---
# ## Single-neuron deep dive

# %% Pick a neuron and inspect it
NEURON = 0   # ← change this

train = trains[NEURON]
print(f"Neuron   : {NEURON}  —  {labels[NEURON]}")
print(f"Spikes   : {len(train)}")
print(f"Mean FR  : {len(train) / (train[-1] - train[0]):.2f} Hz")

# %% Single-neuron PSTH (manual, no subplot grid)
cue_times = sess.event_times("cue")
centres, rate = compute_psth(train, cue_times, pre_ms=500, post_ms=1500, bin_ms=50)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(centres, rate, width=50, color="steelblue", edgecolor="none", alpha=0.7)
ax.axvline(0, color="red", lw=1.0, ls="--", label="Cue")
for name, t_ms in sess.marker_times_ms("cue", 500, 1500).items():
    ax.axvline(t_ms, label=EVENTS[name]["label"], **EVENT_STYLE[name], lw=0.8)
ax.set_xlabel("Time rel. to cue (ms)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"Neuron {NEURON}  —  {labels[NEURON]}")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% Autocorrelogram for the same neuron
c_acg, cnt = compute_acg(train, lag_ms=200, bin_ms=1)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(c_acg, cnt, width=1, color="steelblue", edgecolor="none")
ax.axvline(0, color="red", lw=0.8, ls="--")
ax.set_xlabel("Lag (ms)")
ax.set_ylabel("Spike count")
ax.set_title(f"ACG — neuron {NEURON}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Interactive browser

# %% Launch browser — navigate with arrow keys or Prev / Next
fig = build_browser(sess, event="cue")
plt.show()

# %% [markdown]
# ---
# ## Data exploration

# %% Browse the trials DataFrame
sess.trials.head(10)

# %% Event time arrays at a glance
for name in EVENTS:
    t = sess.event_times(name)
    valid = np.isfinite(t).sum()
    print(f"{name:12s}  {valid:4d} valid  "
          f"mean={np.nanmean(t):.2f} s  std={np.nanstd(t):.2f} s")

# %% Condition masks — trial counts and overlap check
cond = sess.condition_masks()
for name, m in cond.items():
    print(f"{name}: {m.sum()} trials")
total = sum(m.sum() for m in cond.values())
print(f"Total across conditions: {total}  (responding: {mask.sum()})")

# %% List neurons by area  (uncomment and filter)
for i, lbl in enumerate(labels):
    print(f"{i:4d}  {lbl}")

# %% Select neurons by brain area
area_trains, area_labels = select_neurons(trains, labels,
                                          area="Amy",   # ← change area string
                                          enforce_cap=False)
print(f"{len(area_trains)} neurons matched")

# %% [markdown]
# ---
# ## Multi-session loop

# %% Discover all available sessions
session_dirs = sorted(
    d.name for d in Path(REPO_ROOT).iterdir()
    if d.is_dir() and d.name.isdigit() and len(d.name) == 8
)
print("Available sessions:", session_dirs)

# %% Load all sessions (lazy — nothing hits disk until you access a property)
all_sessions = [Session(sid) for sid in session_dirs]

# Summary table across sessions
rows = []
for s in all_sessions:
    t, _ = s.spike_trains
    m    = s.responding_mask
    rows.append({"session": s.id, "neurons": len(t),
                 "trials": len(s.trials), "responding": int(m.sum())})
pd.DataFrame(rows)

# %% [markdown]
# ---
# ## Scratch

# %% Free-form cell — ad-hoc analysis goes here

