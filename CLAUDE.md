# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Analysis scripts for the Human Data project in **Simon Jacob's lab**. Subjects perform a two-armed risky-choice task (Gamble arm vs. Safe arm) while intracranial neural activity is recorded. This repo loads the per-session MATLAB exports into pandas for inspection and downstream analysis.

## Environment

- Python with `scipy`, `numpy`, `pandas`. Env manager is conda (see `.vscode/settings.json`).
- Run scripts directly: `python file_explorer.py`.

## Data layout

Each top-level directory named `YYYYMMDD` is one recording session and contains the same four `.mat` files. The README is the authoritative spec for these — read it before changing any loader.

### `SR.mat`
Scalar sampling rate in Hz (variable `SR`, e.g. 30000). Used to convert `Trials_Sync` columns 15–19 (stored in sampling points) into seconds.

### `STMtx.mat`
**Spike times**, not raw signal. Shape `max_spikes × nNeurons`, each column is one neuron, values are spike times in **seconds**, columns are bottom-padded with `NaN` to a uniform length. The column index *is* the neuron ID used throughout the upstream pipeline.

> The README does not document an `infoCell` variable, but `file_explorer.py:21-32` reads `data["infoCell"]` (area, electrode, unit, type) and uses it for column labels. Either `infoCell` is an undocumented extra in the export, or these sessions carry it as a non-standard addition. Verify against the actual `.mat` before relying on it.

### `Trials_Sync.mat`
Behavioural matrix `(nTrials × 19)` under key `Trials_Sync`. Column meanings (1-indexed in the README, 0-indexed in pandas):

1. Trial Start (s, behavioural clock) · 2. Trial End (s) · 3. Trial duration (s) · 4. Block number · 5. Gamble arm side (R=1, L=0) · 6. P(big reward) · 7. P(small reward) · 8. Big reward amount · 9. Small reward amount · 10. # prior wheel-not-stopping events · 11. Not-responding flag · 12. Chosen side · 13. Chosen arm (Gamble=1, Safe=0) · 14. Rewarded · 15–19. Trial start / cue / response window / reward onset / trial end (**all in sampling points** — divide by `SR`).

> **Warning — column 12 (Chosen side) is unreliable** per the README. Use column 13 (Chosen arm: Gamble/Safe) for choice information.

> **Bug in `file_explorer.py:39`:** the hard-coded column names (`TrialDuration`, `ITI`, `RT`, `Coherence`, `Direction`, `DotsOnset`, …) do not match the README and look copy-pasted from a different (motion / dots) task. Treat that list as wrong and rename against the README before using `load_trials_sync()` for analysis.

### `Human_Data_Table.mat`
Derived per-trial table (one row per trial, all timing in **seconds**) with ~50 columns including `gamble`/`safe`, `REWARD`, cumulative counts (`G_sum`, `S_sum`, …), reward probabilities, value/objective-value columns, choice-change and consecutive-reward streak features. Generated upstream by `Scripts_Matlab/Run_TrialsSync_to_HumanTable.m` → `TrialsSync_to_HumanTable.m` (those scripts live outside this repo). Not currently loaded by `file_explorer.py`. Full column list is in `README.md`.

## Working with `file_explorer.py`

The session being inspected is selected by the `SESSION` constant at `file_explorer.py:6` — change it to switch sessions rather than passing an argument. All paths are resolved relative to the script's own directory, so it can be run from anywhere.

## Repo conventions

- `.mat` files are gitignored — they are large, static, externally produced inputs. Do not commit them. Adding a session = dropping a new `YYYYMMDD/` directory next to the existing ones.
