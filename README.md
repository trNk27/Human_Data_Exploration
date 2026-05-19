# FOR_Jacob_Data

This folder contains scripts and data files prepared for the Human Data project in Simon Jacob's lab. The core session files used across the repository are described below.

---

## Session files

### `STMtx.mat`

Matrix of spike times for all neurons recorded in a session.

- **Size:** `max_spikes × nNeurons` (double)
- **Columns:** one per neuron (STMtx column index = neuron ID used throughout the pipeline)
- **Rows:** spike times in seconds. Each column is padded with `NaN` at the bottom to fill the matrix to uniform size.

---

### `SR.mat`

Sampling rate of the session.

- **Variable:** `SR` — scalar double (e.g. 30000 Hz)
- Used to convert timing columns in `Trials_Sync` (columns 15–19, stored in sampling points) to seconds.

---

### `Trials_Sync.mat`

Behavioural trial table. Each row is one trial, each column is a behavioural variable.

- **Size:** `nTrials × 19` (double)

| Column | Description |
|--------|-------------|
| 1  | Trial Start (seconds from behavioural start) |
| 2  | Trial End (seconds from behavioural start) |
| 3  | Trial duration (seconds, from synchronisation) |
| 4  | Block number |
| 5  | Gamble Arm side (Right = 1, Left = 0) |
| 6  | Probability Big Reward (Gamble arm) |
| 7  | Probability Small Reward (Safe arm) |
| 8  | Amount Big Reward (Gamble arm) |
| 9  | Amount Small Reward (Safe arm) |
| 10 | Number of previous wheel-not-stopping events |
| 11 | Not-responding trial (1 = no response) |
| 12 | Chosen side (Right = 1, Left = 0) — **see warning below** |
| 13 | Chosen arm (Gamble = 1, Safe = 0) |
| 14 | Rewarded trial (1 = rewarded, 0 = not) |
| 15 | Start of the trial (sampling points) |
| 16 | Cue presentation (sampling points) |
| 17 | Start of response window (sampling points) |
| 18 | Reward Period onset (sampling points) |
| 19 | End of the trial (sampling points) |

> **Warning — column 12 (Chosen side):** This column is not reliably extracted and should not be trusted. Use column 13 (Chosen arm: Gamble/Safe) for trial-by-trial choice information.

---

## Derived files

### `Human_Data_Table.mat`

MATLAB table derived from `Trials_Sync` and `SR` by `Scripts_Matlab/TrialsSync_to_HumanTable.m`. Contains one row per trial and only includes columns that can be fully computed from `Trials_Sync` — no pipeline-specific variables (strategy detection, RL outputs, neural data) are included.

All timing values are in **seconds** (sampling-point columns divided by SR).

| Column | Description |
|--------|-------------|
| `F` | Forward arm choice (always 0 — not present in this task) |
| `L` | Left arm chosen (1/0) |
| `R` | Right arm chosen (1/0) |
| `REWARD` | Rewarded trial (1/0) |
| `gamble` | Gamble arm chosen (1/0) |
| `safe` | Safe arm chosen (1/0) |
| `R_sum` | Cumulative right choices |
| `L_sum` | Cumulative left choices |
| `F_sum` | Cumulative forward choices (always 0) |
| `G_sum` | Cumulative gamble choices |
| `S_sum` | Cumulative safe choices |
| `duration` | Trial duration: Reward Period onset − Cue presentation (seconds) |
| `total_sum` | Cumulative non-missed trials |
| `probabilityL` | Reward probability of the Left arm |
| `probabilityR` | Reward probability of the Right arm |
| `probabilityG` | Reward probability of the Gamble arm |
| `probabilityS` | Reward probability of the Safe arm |
| `dropsOfReward` | Reward amount received on this trial (drops) |
| `max_reward` | Maximum possible reward on this trial (Gamble + Safe amounts) |
| `objValueGamble` | Objective value of Gamble arm (= probabilityG) |
| `objValueSafe` | Objective value of Safe arm (= probabilityS) |
| `objectiveValueR` | Objective value of Right arm |
| `objectiveValueL` | Objective value of Left arm |
| `valueDiffObj` | probabilityG − probabilityS |
| `correctDecisions` | 1 = chose the higher-probability arm, 0 = did not, NaN = missed |
| `block_change` | 1 on the first trial of a new block, 0 otherwise |
| `reward` | Reward Period onset (seconds) |
| `Cue_present` | Cue presentation time (seconds) |
| `trial_start` | Trial start time (seconds) |
| `trial_end` | Trial end time (seconds) |
| `wheel_stop` | Start of response window (seconds) |
| `inter_trial_int_start` | Inter-trial interval start (= reward onset) |
| `inter_trial_int_end` | Inter-trial interval end (= trial end) |
| `reward_iti_start` | Reward ITI start (= reward onset) |
| `reward_iti_end` | Reward ITI end (= trial end) |
| `iti_duration` | ITI duration: trial end − reward onset (seconds) |
| `diff_than_previous` | 1 if arm choice differs from previous trial, 0 if same, NaN if either missed |
| `previous_rewarded` | Reward outcome of the previous trial (1/0/NaN) |
| `diff_than_next` | 1 if arm choice differs from next trial, 0 if same, NaN if either missed |
| `consecutive_rewards` | Number of consecutive rewarded trials up to and including this one |
| `consecutive_not_rewarded` | Number of consecutive non-rewarded trials up to and including this one |
| `no_rewards_until_reward` | For reward trials: how many consecutive non-rewards preceded this reward. NaN on non-reward trials |
| `rewards_until_no_reward` | For non-reward trials: how many consecutive rewards preceded this non-reward. NaN on reward trials |
| `choice_change_after_no_reward` | 1 if arm changed on the next trial after a non-reward, 0 if not, NaN if missed |
| `choice_change_after_reward` | 1 if arm changed on the next trial after a reward, 0 if not, NaN if missed |
| `not_REWARD` | 1 − REWARD (non-rewarded trial flag) |
| `not_REWARD_gamble` | Non-rewarded trial on the Gamble arm (1/0) |
| `not_REWARD_safe` | Non-rewarded trial on the Safe arm (1/0) |

Generated by: `Scripts_Matlab/Run_TrialsSync_to_HumanTable.m` → calls `Scripts_Matlab/TrialsSync_to_HumanTable.m` for each session in `Human_Folders.mat`.

---

# Analysis codebase

This repository loads the per-session `.mat` exports into pandas and runs
visualisation and statistical analyses on the spike data. The goal is to
analyse reward encoding in the human brain. All scripts are plain Python files
run from the repo root.

## Environment

- Python with `numpy`, `scipy`, `pandas`, `matplotlib`. The ZETA scripts also
  need `zetapy` (`pip install zetapy`).
- Conda is the environment manager (see `.vscode/settings.json`); the project
  environment is `humandata`.
- Run any script directly: `python <script>.py [arguments]`.

## Selecting the active session

Every loader reads the session named by `SESSION` in `utils.py` (line 12). To
switch sessions, edit that one constant — e.g. `SESSION = "20250602"`. A
session is one `YYYYMMDD/` directory holding `SR.mat`, `STMtx.mat`,
`Trials_Sync.mat` (and optionally `Human_Data_Table.mat`). Adding a session =
dropping a new `YYYYMMDD/` directory next to the existing ones. The batch
scripts rewrite this constant automatically to sweep every session.

## Conventions shared across scripts

- **Units:** analysis parameters (bins, windows, lags) are in **milliseconds**
  (`--bin`, `--pre`, `--post`, `--lag`); absolute spike/event timestamps stay
  in **seconds**, as stored in the data.
- **Neuron IDs:** the column index in `STMtx` is the neuron ID. Labels read
  `unit | area electrode (su/mu)` (single- / multi-unit).
- `--list` — most plotting scripts: print every neuron index + label, then exit.
- `--neurons 0 1 5` — restrict to specific neuron indices.
- `--area MFG` — restrict to neurons whose label contains a substring
  (case-insensitive).
- `--save [FILE]` — save the figure; with no path it auto-names
  `<prefix>_<session>.png`. Omitting `--save` only shows the plot on screen.
- Per-figure plotting scripts cap at `MAX_NEURONS` (90) — narrow the selection
  with `--neurons`/`--area` if you exceed it.

## `utils.py` — shared utilities

Single source of truth, imported by every other script; not run directly.
Provides the `SESSION` / `DATA_DIR` / `MAX_NEURONS` constants, the `.mat`
loaders (`load_sr`, `load_stmtx`, `load_trials_sync` — each accepts an optional
`data_dir`), `get_spike_trains()` → `(trains, labels)`, `sp_to_s()` for
sampling-point → seconds conversion, and the shared `--save` plumbing
(`add_save_arg` / `maybe_save`).

---

## Data inspection

### `file_explorer.py`
Prints a textual overview of the active session: sampling rate, the first rows
of the spike matrix and trial table, and shape/summary info. No arguments.

```
python file_explorer.py
```

### `mat_to_csv.py`
Converts every session's `.mat` files to CSV under `csv/<session>/`. Walks all
session directories and handles `SR`, `STMtx`, `Trials_Sync` and (best-effort)
`Human_Data_Table`. No arguments.

```
python mat_to_csv.py
```

---

## Single-session visualisation

### `raster_plot.py`
Spike raster plots. Default mode is a full-recording raster (one row per
neuron); `--aligned` switches to a trial-by-trial raster aligned to a
behavioural event (one subplot per neuron, one row per trial).

| Argument | Default | Description |
|---|---|---|
| `t_start t_end` | none | (full mode) positional start / end time in seconds |
| `--aligned` | off | trial-by-trial aligned raster instead of full recording |
| `--event {cue,response,reward,start}` | `cue` | (aligned) event to align to |
| `--pre` | 500 | (aligned) ms before event |
| `--post` | 1000 | (aligned) ms after event |
| `--by-condition` | off | (aligned) colour trials by (arm × reward) condition |
| `--neurons N [N ...]` | all | restrict to neuron indices |
| `--area STR` | all | filter neurons by label substring |
| `--list` | — | print neuron list and exit |
| `--save [FILE]` | — | save the figure |

```
python raster_plot.py 0 60
python raster_plot.py --aligned --event reward --pre 500 --post 1500 --area MFG
```

### `psth.py`
Peristimulus time histogram — firing rate (Hz) aligned to an event, binned
across trials. Marker lines show the mean timing of the other events.

| Argument | Default | Description |
|---|---|---|
| `--event {cue,response,reward,start}` | `cue` | alignment event |
| `--pre` | 500 | ms before event |
| `--post` | 1000 | ms after event |
| `--bin` | 50 | histogram bin width (ms) |
| `--sigma` | off | Gaussian smoothing SD (ms) |
| `--by-condition` | off | overlay one curve per (arm, reward) condition |
| `--neurons` / `--area` / `--list` / `--save` | — | as in the shared conventions |

```
python psth.py --event reward --bin 25 --sigma 50 --by-condition
```

### `autocorrelogram.py`
Per-neuron autocorrelogram via FFT-based circular autocorrelation.

| Argument | Default | Description |
|---|---|---|
| `--lag` | 200 | max lag (ms) |
| `--bin` | 1 | bin width (ms) |
| `--neurons` / `--area` / `--list` / `--save` | — | as in the shared conventions |

```
python autocorrelogram.py --area IFG --lag 100
```

### `browser.py`
Interactive single-neuron browser: PSTH (top) + autocorrelogram (bottom), one
neuron at a time. Navigate with the Prev/Next buttons, arrow keys, or by typing
a neuron index.

| Argument | Default | Description |
|---|---|---|
| `--event {cue,response,reward,start}` | `cue` | PSTH alignment event |
| `--pre` / `--post` | 500 / 1000 | PSTH window (ms) |
| `--bin` | 50 | PSTH bin width (ms) |
| `--lag` | 200 | ACG max lag (ms) |
| `--bin-acg` | 1 | ACG bin width (ms) |
| `--neurons` / `--area` | — | restrict the neuron selection |

```
python browser.py --event reward --area MFG
```

---

## ZETA statistics

ZETA (Zenith of Event-based Time-locked Anomalies) is a parameter-free test for
whether spike timing is modulated relative to events. Requires `zetapy`. Both
scripts test every neuron in parallel across CPU cores and write results to
`results/`. Pass `--csv` to write the table and `--save` to write the plot;
without them the scripts only print and show.

### `zeta_analysis.py` — one-sample ZETA (responsiveness)
Tests whether each neuron *responds* to a behavioural event, against a uniform
null. Outputs a p-value table per event (sorted by p-value) and an IFR grid of
the top-N most significant neurons.

| Argument | Default | Description |
|---|---|---|
| `--event {cue,response,reward,trial_start,all}` | `all` | event(s) to test |
| `--dur` | 2.0 | analysis window (s) after the event |
| `--resamp` | 100 | jitter iterations |
| `--alpha` | 0.05 | significance threshold |
| `--top` | 8 | top-N significant neurons to plot |
| `--csv` | off | write `results/zeta_<event>_<session>.csv` |
| `--jobs` | all cores | parallel worker processes (`1` = serial) |
| `--save [FILE]` | — | save the IFR plot (auto-named `zeta_<event>_<session>.png`) |

```
python zeta_analysis.py --event reward --csv --save
```

### `zeta_outcome.py` — two-sample ZETA (outcome differences)
Tests whether a neuron's reward-aligned response *differs between trial
outcomes*. All trials are aligned to reward onset, responding trials only.
Outcomes — `G+R` (gamble + rewarded), `G+N` (gamble + not rewarded), `S+R`
(safe + rewarded) — feed two contrasts, each isolating one factor:

- `reward` — G+R vs G+N (effect of reward; choice held = gamble)
- `choice` — G+R vs S+R (effect of choice; reward held constant)

| Argument | Default | Description |
|---|---|---|
| `--contrast {reward,choice,all}` | `all` | contrast(s) to run |
| `--dur` | 2.0 | analysis window (s) |
| `--resamp` | 250 | jitter iterations |
| `--alpha` | 0.05 | significance threshold |
| `--top` | 8 | top-N significant neurons to plot |
| `--csv` | off | write `results/zeta2_<contrast>_<session>.csv` |
| `--jobs` | all cores | parallel worker processes (`1` = serial) |
| `--save [FILE]` | — | save the plot (auto-named `zeta2_<contrast>_<session>.png`) |

```
python zeta_outcome.py --contrast reward --csv --save
```

> **One-sample vs two-sample.** `zeta_analysis.py` tells you *whether* a neuron
> is responsive; `zeta_outcome.py` tells you whether the response *differs*
> between conditions. Comparing two separate one-sample p-values is not a valid
> difference test — use `zeta_outcome.py` for that.

---

## Batch & export tools

### `batch_zeta.py`
Runs the ZETA scripts for **every session**. For each session it temporarily
rewrites `SESSION` in `utils.py` (restored afterwards), runs the selected
script(s) with `--csv --save --top 8`, and collects all CSVs + PNGs in
`results/`.

| Argument | Default | Description |
|---|---|---|
| `--analysis {responsiveness,outcome,both}` | `both` | `responsiveness` runs `zeta_analysis.py`, `outcome` runs `zeta_outcome.py` |
| `--sessions YYYYMMDD ...` | all | sessions to process |
| `--resamp` | each script's own | jitter-iteration passthrough |
| `--dur` | — | analysis-window passthrough |
| `--jobs` | all cores | parallel workers per session |

```
python batch_zeta.py --analysis outcome
python batch_zeta.py --sessions 20250521 20250602 --jobs 6
```

### `export_acg.py`
Batch-exports an autocorrelogram PNG for every neuron in the active session
(two panels: ±75 ms and ±300 ms) to `acg_export/<session>/`. No arguments.

```
python export_acg.py
```

### `batch_export_acg.py`
Runs `export_acg.py` for every session not already in its `DONE` set,
rewriting `SESSION` per session. No arguments.

```
python batch_export_acg.py
```

### `generate_test_data.py`
Creates a synthetic `test_session/` (SR, STMtx with three known neurons, a
minimal Trials_Sync) for testing the loaders and plots. No arguments.

```
python generate_test_data.py
```

---

## Typical workflows

- **Inspect a new session** — drop the `YYYYMMDD/` folder into the repo, set
  `SESSION` in `utils.py`, run `python file_explorer.py`.
- **Look at single neurons** — `python browser.py`, or
  `python psth.py --by-condition` for a condition-split overview.
- **Full ZETA sweep across all sessions** — `python batch_zeta.py` writes
  every session's responsiveness and outcome CSVs + plots to `results/`.

> The `csv/` and `.mat` files are gitignored — large, externally produced
> inputs/outputs that are not committed.
