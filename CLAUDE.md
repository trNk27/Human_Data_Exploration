# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Analysis scripts for the Human Data project in **Simon Jacob's lab**. Subjects perform a two-armed risky-choice task (Gamble arm vs. Safe arm) while intracranial neural activity is recorded. This repo loads the per-session MATLAB exports into pandas for inspection and downstream analysis.
The goal is to analyze the data and gain insights into reward encoding in the human brain.

## Environment

- Python with `scipy`, `numpy`, `pandas`. Env manager is conda (see `.vscode/settings.json`).
- Run scripts directly: `python file_explorer.py`.

## Data layout

Each top-level directory named `YYYYMMDD` is one recording session and contains the same four `.mat` files. The README is the authoritative spec for these — read it before changing any loader.

### `SR.mat`
Scalar sampling rate in Hz (variable `SR`, e.g. 30000). Used to convert `Trials_Sync` columns 15–19 (stored in sampling points) into seconds.

### `STMtx.mat`
**Spike times**, not raw signal. Shape `max_spikes × nNeurons`, each column is one neuron, values are spike times in **seconds**, columns are bottom-padded with `NaN` to a uniform length. The column index *is* the neuron ID used throughout the upstream pipeline.


### `Trials_Sync.mat`
Behavioural matrix `(nTrials × 19)` under key `Trials_Sync`. Column meanings (1-indexed in the README, 0-indexed in pandas):

1. Trial Start (s, behavioural clock) · 2. Trial End (s) · 3. Trial duration (s) · 4. Block number · 5. Gamble arm side (R=1, L=0) · 6. P(big reward) · 7. P(small reward) · 8. Big reward amount · 9. Small reward amount · 10. # prior wheel-not-stopping events · 11. Not-responding flag · 12. Chosen side · 13. Chosen arm (Gamble=1, Safe=0) · 14. Rewarded · 15–19. Trial start / cue / response window / reward onset / trial end (**all in sampling points** — divide by `SR`).

> **Warning — column 12 (Chosen side) is unreliable** per the README. Use column 13 (Chosen arm: Gamble/Safe) for choice information.


### `Human_Data_Table.mat`
Derived per-trial table (one row per trial, all timing in **seconds**) with ~50 columns including `gamble`/`safe`, `REWARD`, cumulative counts (`G_sum`, `S_sum`, …), reward probabilities, value/objective-value columns, choice-change and consecutive-reward streak features. Generated upstream by `Scripts_Matlab/Run_TrialsSync_to_HumanTable.m` → `TrialsSync_to_HumanTable.m` (those scripts live outside this repo). Not currently loaded by `file_explorer.py`. Full column list is in `README.md`.

## Code structure

| File | Role |
|---|---|
| `utils.py` | Single source of truth for all shared utilities — loaders, `get_spike_trains()`, `sp_to_s()`, `SESSION`, `DATA_DIR`, `MAX_NEURONS`. **All analysis scripts import from here.** |
| `file_explorer.py` | Interactive data explorer (`python file_explorer.py`). Thin wrapper; imports from `utils.py`. |
| `raster_plot.py` | Spike raster visualisation. |
| `psth.py` | Peristimulus time histogram aligned to behavioural events (cue / response / reward / trial start). |
| `autocorrelogram.py` | Per-neuron autocorrelogram via FFT. |

The active session is set by `SESSION` in `utils.py:12` — change it there to switch sessions. All loaders (`load_sr`, `load_stmtx`, `load_trials_sync`, `get_spike_trains`) accept an optional `data_dir` to target a different session without changing the constant.

## Repo conventions

- `.mat`; `.csv` files are gitignored — they are large, static, externally produced inputs. Do not commit them. Adding a session = dropping a new `YYYYMMDD/` directory next to the existing ones.

## Python conventions
- Always save Python code to a `.py` file before running it.
- Run scripts with `python <file>.py`, not via inline `python -c` or heredocs.
- Use `scripts/` for one-off scripts, `src/` for application code.