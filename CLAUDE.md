# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Analysis scripts for the Human Data project in **Simon Jacob's lab**. Subjects perform a two-armed risky-choice task (Gamble arm vs. Safe arm) while intracranial neural activity is recorded. This repo loads the per-session MATLAB exports into pandas for inspection and downstream analysis.
The goal is to analyze the data and gain insights into reward encoding in the human brain.

## Environment

- Python with `scipy`, `numpy`, `pandas`. Env manager is conda (see `.vscode/settings.json`).
- environment is called `humandata`
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

Three layers: a stable **foundation**, a single-neuron **plotting layer**, and the
**CLIs / analyses** built on top.

**Foundation (top level)**

| File | Role |
|---|---|
| `utils.py` | Single source of truth for shared utilities — `.mat` loaders, `get_spike_trains()`, `sp_to_s()`, `EVENTS`, `CONDITIONS`, `select_neurons()`, the CLI-flag helpers, `SESSION`/`DATA_DIR`/`MAX_NEURONS`, and figure-save helpers. **Everything imports from here.** |
| `session.py` | `Session(session_id)` — lazy, cached per-session data (trials, sampling rate, spike trains) + alignment helpers. `Session.neuron(i)` returns the fluent `NeuronView`. |
| `compute.py` | Pure numeric kernels, no matplotlib: `compute_psth`, `compute_aligned_raster`, `compute_acg`, `perceived_probability`, `trial_firing_rates`, `binned_stats`. |
| `explore.py` | The single-neuron plotting layer — `draw_*` (one neuron / one axes), `NeuronView`, `Panel` (auto-saving), and `grid()` (tiles the same draws across many neurons). **Import this for exploratory plotting.** |
| `file_explorer.py` | Interactive data explorer (`python file_explorer.py`). |

**`viewers/`** — multi-neuron grid CLIs, thin wrappers over `explore.grid`: `psth.py`, `raster_plot.py`, `autocorrelogram.py`, `firing_rate_vs_perc_p.py`, `browser.py`.

**`analysis/`** — aggregate / population / behavioural analyses (one figure or CSV over many neurons/sessions): `zeta_analysis.py`, `zeta_outcome.py`, `batch_zeta.py`, `population_heatmap.py`, `responsive_region.py`, `outcome_direction.py`, `choice_timeline.py`, `behavioural_simulation.py`, `export_acg.py`, `batch_export_acg.py`.

**`ui/app.py`** — Streamlit explorer (`streamlit run ui/app.py`). **`scripts/`** — one-offs, test-data generator, smoke checks.

### Exploratory plotting — the fluent API

One neuron, one condition, saved — then switch:

```python
from session import Session
sess = Session("20250714")

n = sess.neuron(7)                                   # sticky: reuse for several plots
n.psth(condition="G+R", align="reward").save()       # -> results/figures/20250714/psth_n7_G+R_reward.png
n.raster(condition="G+N", align="cue").save()
sess.neuron(12).acg().save()
sess.neuron(3).fr_vs_p(window="cue_to_reward").save()

from explore import grid                              # many neurons, same draws
grid(sess, "psth", area="ACC", align="reward", condition="all").save("acc.png")
```

`condition` (psth / raster): a CONDITIONS key (`"G+R"`/`"G+N"`/`"S+R"`) filters trials to
that outcome; `None` = all responding trials; `"all"` = overlay the three. `Panel.save()`
with no path auto-names into `results/figures/<session>/`; pass a path to save elsewhere.

The active session is set by `SESSION` in `utils.py` — change it there, or pass `--session`
to any CLI. All loaders (`load_sr`, `load_stmtx`, `load_trials_sync`, `get_spike_trains`)
accept an optional `data_dir` to target a different session without changing the constant.

## Units convention

- **Analysis time parameters** (bin sizes, PSTH windows) are in **milliseconds** — `bin_ms`, `pre_ms`, `post_ms`, `lag_ms`.
- **Absolute timestamps and event times** (spike times, trial event columns after `sp_to_s`) remain in **seconds**, as stored in the data.
- Internal conversion at function entry: `bin_s = bin_ms / 1000`, etc. `compute_psth` returns bin centres in ms.

## Repo conventions

- `.mat`; `.csv` files are gitignored — they are large, static, externally produced inputs. Do not commit them. Adding a session = dropping a new `YYYYMMDD/` directory next to the existing ones.

## Python conventions
- Always save Python code to a `.py` file before running it.
- Run scripts with `python <file>.py`, not via inline `python -c` or heredocs.
- Subfolder CLIs run either way (each has a `sys.path` bootstrap): `python -m viewers.psth …` / `python viewers/psth.py …`; likewise `python -m analysis.population_heatmap …`.
- Use `scripts/` for one-off scripts. Application code lives at the top level (`utils`, `session`, `compute`, `explore`) plus `viewers/` and `analysis/`.