# COMMANDS.md

Every command for exploring, analysing, and plotting the Human Data project.
Each entry gives the **default** invocation, an **example** with options, and the
**settable parameters**.

All commands run from the repo root in the `humandata` conda environment.

---

## Conventions

### How to run

- Top-level scripts:            `python file_explorer.py`
- `viewers/` and `analysis/` CLIs run either way (each has a `sys.path` bootstrap):
  - `python -m viewers.psth …`   (module form)
  - `python viewers/psth.py …`   (path form)

### Shared CLI flags

Most CLIs share these flags (defined in [utils.py](utils.py)):

| Flag | Default | Meaning |
|---|---|---|
| `--session YYYYMMDD` | `20250714` (`utils.SESSION`) | Which recording session to load. |
| `--neurons i j k` | all | Neuron indices to show (plotting viewers only). |
| `--area STR` | all | Keep only neurons whose label contains `STR` (case-insensitive). |
| `--list` | off | Print neuron indices + labels, then exit. |
| `--save [FILE]` | off | Save the figure. No path → auto-named `<prefix>_<session>.png`; with a path → save there. |

> The per-figure neuron cap is **90** (`utils.MAX_NEURONS`). Narrow with `--neurons`/`--area` if you exceed it.
> Available time windows for firing-rate plots (`compute.WINDOWS`): `trial`, `cue_to_reward`, `reward_to_end`.
> Alignment events (`utils.EVENTS`): `cue`, `response`, `reward`, `trial_start`.
> Outcome conditions (`utils.CONDITIONS`): `G+R` (gamble+rewarded), `G+N` (gamble+no reward), `S+R` (safe+rewarded).

---

## 1. Inspect raw data

### `file_explorer.py` — textual overview of a session

Prints sampling rate, a slice of the spike matrix, the first 20 trials, and a size summary.

- **Default:** `python file_explorer.py`
- **Example:** `python file_explorer.py --session 20250605`
- **Parameters:** `--session`

### `scripts/mat_to_csv.py` — dump every `.mat` to CSV

Writes `csv/<session>/<file>.csv` for all sessions. No parameters.

- **Default / only form:** `python scripts/mat_to_csv.py`

---

## 2. Exploratory plotting — fluent Python API

The primary single-neuron interface ([explore.py](explore.py), [session.py](session.py)). Use from a
Python/IPython session or a `scripts/` file. `Panel.save()` with no path auto-names into
`results/figures/<session>/`.

```python
from session import Session
sess = Session("20250714")               # change session here

n = sess.neuron(7)                         # sticky: reuse for several plots
n.psth(condition="G+R", align="reward").save()
n.raster(condition="G+N", align="cue").save()
sess.neuron(12).acg().save()
sess.neuron(3).fr_vs_p(window="cue_to_reward").save()
sess.neuron(3).fr_vs_hgf_p(window="cue_to_reward").save()   # needs HGF CSVs (see §6)
sess.neuron(3).fr_vs_delta1(window="cue_to_reward").save()  # needs HGF CSVs (see §6)
```

| Method | Defaults | Settable parameters |
|---|---|---|
| `.psth(...)` | `condition=None, align="cue", pre_ms=500, post_ms=1000, bin_ms=50, sigma_ms=None` | condition (`G+R`/`G+N`/`S+R`/`None`/`"all"`), align, pre/post/bin/sigma (ms) |
| `.raster(...)` | `condition=None, align="cue", pre_ms=500, post_ms=1000` | condition, align, pre/post (ms) |
| `.acg(...)` | `lag_ms=200, bin_ms=1` | lag, bin (ms) |
| `.fr_vs_p(...)` | `window="trial", history=10, n_bins=8, by_condition=False` | window, history, n_bins, by_condition |
| `.fr_vs_hgf_p(...)` | `window="trial", n_bins=8, by_condition=False` | window, n_bins, by_condition |
| `.fr_vs_delta1(...)` | `window="trial", n_bins=8, by_condition=True` | window, n_bins, by_condition |

**Many neurons at once** — `grid()` tiles the same draw across neurons:

```python
from explore import grid
grid(sess, "psth", area="ACC", align="reward", condition="all").save("acc.png")
```

- `grid(sess, kind, neurons=None, area=None, ncols=4, **kw)` where `kind` is one of
  `psth`, `raster`, `acg`, `fr_vs_p`, `fr_vs_hgf_p`, `fr_vs_delta1`; `**kw` are the
  per-method parameters above.

---

## 3. Plotting CLIs (`viewers/`)

These are thin wrappers over `explore.grid`. All accept the shared flags (§Conventions).

### `viewers/psth.py` — PSTH grid (one subplot per neuron)

- **Default:** `python -m viewers.psth`
- **Example:** `python -m viewers.psth --area ACC --event reward --bin 25 --sigma 40 --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--event` | `cue` | Alignment event (`cue`/`response`/`reward`/`trial_start`). |
| `--pre` | `500` | ms before event. |
| `--post` | `1000` | ms after event. |
| `--bin` | `50` | Bin width (ms). |
| `--sigma` | off | Gaussian smoothing SD (ms). |
| `--by-condition` | off | Overlay one curve per (arm, reward) condition. |
| + shared | | `--session --neurons --area --list --save` |

### `viewers/raster_plot.py` — spike raster

Two modes: full-recording raster (default) and trial-aligned raster (`--aligned`).

- **Default (full recording):** `python -m viewers.raster_plot`
- **Example (window 0–100 s):** `python -m viewers.raster_plot 0 100 --area ACC`
- **Example (aligned):** `python -m viewers.raster_plot --aligned --event reward --by-condition --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `t_start t_end` (positional) | none | Full-raster time window in seconds. |
| `--aligned` | off | Trial-by-trial aligned raster instead of full recording. |
| `--event` | `cue` | (aligned) Alignment event. |
| `--pre` / `--post` | `500` / `1000` | (aligned) ms before / after event. |
| `--by-condition` | off | (aligned) Colour trials by condition. |
| + shared | | `--session --neurons --area --list --save` |

### `viewers/autocorrelogram.py` — ACG grid

- **Default:** `python -m viewers.autocorrelogram`
- **Example:** `python -m viewers.autocorrelogram --area ACC --lag 100 --bin 0.5 --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--lag` | `200` | Max lag (ms). |
| `--bin` | `1` | Bin size (ms). |
| + shared | | `--session --neurons --area --list --save` |

### `viewers/browser.py` — interactive single-neuron browser

PSTH (top) + ACG (bottom), navigate with Prev/Next, arrow keys, or a neuron-index box.

- **Default:** `python -m viewers.browser`
- **Example:** `python -m viewers.browser --session 20250605 --event reward --area ACC`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--event` | `cue` | PSTH alignment event. |
| `--pre` / `--post` | `500` / `1000` | PSTH ms before / after. |
| `--bin` | `50` | PSTH bin width (ms). |
| `--bin-acg` | `1` | ACG bin width (ms). |
| `--lag` | `200` | ACG max lag (ms). |
| + shared | | `--session --neurons --area` (no `--save`; it is interactive) |

### `viewers/firing_rate_vs_perc_p.py` — FR vs behavioural perceived P(reward)

x-axis = rolling fraction of rewards over the last `--history` gamble trials. Gamble trials only.

- **Default:** `python -m viewers.firing_rate_vs_perc_p`
- **Example:** `python -m viewers.firing_rate_vs_perc_p --window cue_to_reward --history 10 --bins 8 --by-condition --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--window` | `trial` | FR window (`trial`/`cue_to_reward`/`reward_to_end`). |
| `--history` | `10` | Past-trial window for perceived probability. |
| `--bins` | `8` | Number of probability bins for the mean overlay. |
| `--by-condition` | off | Colour by outcome; one regression per condition. |
| + shared | | `--session --neurons --area --list --save` |

### `viewers/firing_rate_vs_hgf_p.py` — FR vs HGF latent belief p̂

x-axis = HGF model's latent P(gamble pays) per trial. Reads `results/hgf/trajectory_<session>.csv`
(run §6 first). All responding trials.

- **Default:** `python -m viewers.firing_rate_vs_hgf_p`
- **Example:** `python -m viewers.firing_rate_vs_hgf_p --window cue_to_reward --bins 8 --area ACC --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--window` | `trial` | FR window. |
| `--bins` | `8` | Number of belief bins for the mean overlay. |
| `--by-condition` | off | Colour by outcome; one regression per condition. |
| + shared | | `--session --neurons --area --list --save` |

### `viewers/firing_rate_vs_delta1.py` — FR vs HGF prediction error δ₁

x-axis = HGF level-1 prediction error δ₁ = outcome − p̂. Gamble trials only.
Reads `results/hgf/trajectory_<session>.csv` (run §6 first).

- **Default:** `python -m viewers.firing_rate_vs_delta1`
- **Example:** `python -m viewers.firing_rate_vs_delta1 --window cue_to_reward --bins 8 --pooled --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--window` | `trial` | FR window. |
| `--bins` | `8` | Number of bins for the pooled mean overlay. |
| `--pooled` | off (= per-condition) | Single global fit instead of one regression per outcome. |
| + shared | | `--session --neurons --area --list --save` |

---

## 4. Population & responsiveness analyses (`analysis/`)

### `analysis/population_heatmap.py` — z-scored population activity heatmap

Trial-start-aligned, z-scored per neuron; neurons × time heatmap with median cue/reward markers.

- **Default:** `python -m analysis.population_heatmap`
- **Example:** `python -m analysis.population_heatmap --session 20250521 --bin-ms 25 --smooth-ms 100 --sort-by region --vmax 3 --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Session. |
| `--bin-ms` | `50` | Time bin width (ms). |
| `--smooth-ms` | `0` (off) | Gaussian smoothing SD (ms) before z-scoring. |
| `--post-ms` | 90th pct of trial duration | X-axis upper bound (ms from trial start). |
| `--sort-by` | `peak_time` | Neuron ordering (`peak_time`/`region`/`none`). |
| `--vmax` | `2.5` | Z-score colour clamp. |
| `--min-spikes` | `20` | Skip neurons with fewer spikes. |
| `--save` | off | Save figure. |

### `analysis/zeta_analysis.py` — one-sample ZETA responsiveness

Tests each neuron's responsiveness to events; ranks by p-value and plots IFR for top-N. Parallel across CPU cores. *(Requires `pip install zetapy`.)*

- **Default:** `python analysis/zeta_analysis.py`  (runs all events)
- **Example:** `python analysis/zeta_analysis.py --session 20250605 --event reward --top 10 --csv --save`
- **Example (variable window):** `python analysis/zeta_analysis.py --event cue --window-end reward --csv`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Session. |
| `--event` | `all` | Event (`cue`/`response`/`reward`/`trial_start`/`all`). |
| `--dur` | `2.0` | Fixed analysis window (s). Ignored with `--window-end`. |
| `--window-end` | none | Variable-duration mode `[event, window_end]` per trial; needs a specific `--event`. |
| `--resamp` | `100` | Jitter iterations. |
| `--alpha` | `0.05` | Significance threshold. |
| `--top` | `8` | Top-N significant neurons to plot. |
| `--csv` | off | Write full results table to `results/zeta_responsiveness/`. |
| `--jobs` | all cores | Worker processes (`1` = serial). |
| `--save` | off | Save IFR plots. |

### `analysis/zeta_outcome.py` — two-sample ZETA (outcome differences)

Reward-aligned; tests whether responses differ between outcomes. Contrasts: `reward` (G+R vs G+N), `choice` (G+R vs S+R). *(Requires `zetapy`.)*

- **Default:** `python analysis/zeta_outcome.py`  (both contrasts)
- **Example:** `python analysis/zeta_outcome.py --session 20250605 --contrast reward --top 10 --csv --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Session. |
| `--contrast` | `all` | `reward` / `choice` / `all`. |
| `--dur` | `2.0` | Analysis window (s). |
| `--resamp` | `250` | Jitter iterations. |
| `--alpha` | `0.05` | Significance threshold. |
| `--top` | `8` | Top-N neurons to plot. |
| `--csv` | off | Write table to `results/zeta_outcome/`. |
| `--jobs` | all cores | Worker processes (`1` = serial). |
| `--save` | off | Save diff plots. |

### `analysis/batch_zeta.py` — run ZETA scripts over all sessions

Discovers sessions and invokes the two ZETA scripts with `--csv --save --top 8`.

- **Default:** `python analysis/batch_zeta.py`  (both analyses, all sessions)
- **Example:** `python analysis/batch_zeta.py --analysis outcome --sessions 20250521 20250602 --jobs 6 --resamp 100`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--analysis` | `both` | `responsiveness` / `outcome` / `both`. |
| `--sessions` | all `YYYYMMDD` dirs | Sessions to process. |
| `--resamp` | each script's own | Override jitter iterations. |
| `--dur` | each script's own | Override analysis window (s). |
| `--jobs` | all cores | Worker processes per session. |
| `--event` | `all` | Restrict responsiveness to one event. |
| `--window-end` | none | Variable-duration mode (responsiveness only; requires `--event`). |

### `analysis/responsive_region.py` — region × responsiveness chi-squared

Pools the per-session ZETA responsiveness CSVs and tests whether % responsive depends on brain region. Reads `results/zeta_responsiveness/` (run ZETA first).

- **Default:** `python -m analysis.responsive_region`
- **Example:** `python -m analysis.responsive_region --event reward --alpha 0.01 --min-n 10 --csv --plot`
- **Example (variable window):** `python -m analysis.responsive_region --event cue --window-end reward --plot`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--event` | `all` | Event, or all events + combined. |
| `--alpha` | `0.05` | Significance threshold on `p_zeta`. |
| `--min-n` | `5` | Drop regions with fewer pooled neurons. |
| `--csv` | off | Save contingency tables. |
| `--plot` | off | Save a regions × events heatmap. |
| `--window-end` | none | Analyse variable-window CSVs; requires a specific `--event`. |

### `analysis/outcome_direction.py` — direction of outcome effect (SI)

Appends `rate_*`, `SI`, `preference` columns to the `zeta2_*` CSVs and writes per-contrast SI-distribution PNGs to `results/direction/`. Reads `results/zeta_outcome/` (run `zeta_outcome.py` first).

- **Default:** `python -m analysis.outcome_direction`
- **Example:** `python -m analysis.outcome_direction --sessions 20250521 20250602 --window-ms 200 --alpha 0.01`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--sessions` | all `YYYYMMDD` dirs | Sessions to process. |
| `--window-ms` | `300` | Half-width of window around `zeta_t_s` (ms). |
| `--dur` | `2.0` | ZETA analysis window (s). |
| `--alpha` | `0.05` | Significance threshold for the histogram. |

### `analysis/build_neuron_table.py` — per-neuron summary CSV

Joins all ZETA CSVs into `results/neuron_summary_<session>.csv` (one row per neuron). Run the ZETA analyses first.

- **Default:** `python -m analysis.build_neuron_table`  (all discovered sessions)
- **Example:** `python -m analysis.build_neuron_table --session 20250714`
- **Parameters:** `--session SESSION [SESSION ...]` (default: all sessions with ZETA CSVs).

### `analysis/export_acg.py` — batch ACG PNGs for one session

One PNG per neuron (±75 ms and ±300 ms panels, 1 ms bins) → `acg_export/<session>/`.

- **Default / only form:** `python analysis/export_acg.py`
- **Example:** `python analysis/export_acg.py --session 20250605`
- **Parameters:** `--session`

### `analysis/batch_export_acg.py` — export ACGs for all remaining sessions

Runs `export_acg.py` for every `YYYYMMDD/` not already done. No parameters.

- **Default / only form:** `python analysis/batch_export_acg.py`

---

## 5. Behaviour & choice modelling (`analysis/`)

### `analysis/behaviour_overview.py` — raw choice behaviour + true reward probability

Per-session choice strips (gamble/safe, rewarded/unrewarded) over the scheduled P(reward) step line.

- **Default:** `python -m analysis.behaviour_overview`
- **Example (single):** `python -m analysis.behaviour_overview --session 20250714 --save`
- **Example (all):** `python -m analysis.behaviour_overview --all --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Single session. |
| `--all` | off | Plot all sessions in one grid (ignores `--session`). |
| `--save` | off | Save figure to `results/figures/…`. |

### `analysis/choice_timeline.py` — choices vs Rescorla–Wagner model

Participant choices vs an RW + softmax + perseveration model, with rolling match-rate.
`shadow` mode predicts the participant's choices; `simulate` mode free-runs.

- **Default:** `python -m analysis.choice_timeline`  (shadow mode, α=0.1, β=5, φ=0)
- **Example:** `python -m analysis.choice_timeline --session 20250602 --alpha 0.15 --beta 4 --phi 1.0 --window 20 --save`
- **Example (simulate):** `python -m analysis.choice_timeline --mode simulate --phi 1.0 --seed 42`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Session. |
| `--alpha` | `0.1` | RW learning rate (0–1). |
| `--beta` | `5.0` | Softmax inverse temperature (>0). |
| `--phi` | `0.0` | Perseveration strength (>0 = sticky). |
| `--window` | `20` | Rolling window for the match-rate metric. |
| `--mode` | `shadow` | `shadow` (predict participant) / `simulate` (free-run). |
| `--seed` | none | RNG seed for `simulate`. |
| `--save` | off | Save figure. |

### `analysis/fit_rw.py` — maximum-likelihood RW fit

Fits (α, β, φ) by maximising choice log-likelihood, prints a base-vs-full comparison table (AIC/BIC/LRT), and plots the best-fit timeline.

- **Default:** `python analysis/fit_rw.py`
- **Example:** `python analysis/fit_rw.py --session 20250602 --restarts 50 --save`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--session` | `20250714` | Session. |
| `--restarts` | `25` | Optimisation restarts. |
| `--no-plot` | off | Skip the timeline plot after fitting. |
| `--save` | off | Save the timeline figure. |

---

## 6. HGF latent-belief pipeline (`analysis/hgf/`)

### `analysis.hgf.run` — end-to-end HGF fit + trajectories + comparison

Fits a 3-level binary HGF (free: omega2, beta, bias), writes per-session
`results/hgf/trajectory_<session>.csv` (consumed by the `fr_vs_hgf_p` / `fr_vs_delta1`
viewers), plus model-comparison, parameter-recovery, power, and (optionally) hierarchical outputs.

- **Default:** `python -m analysis.hgf.run`  (all sessions, including the NUTS hierarchical fit)
- **Example (fast):** `python -m analysis.hgf.run --sessions 20250714 --no-hierarchical`
- **Example (extended params):** `python -m analysis.hgf.run --free omega3 kappa --out results/hgf_extended`
- **Parameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--sessions` | all available | Session IDs to include. |
| `--no-hierarchical` | off | Skip the PyMC/NUTS hierarchical fit (much faster). |
| `--n-recovery` | `24` | Simulate→refit cycles for parameter recovery. |
| `--n-power` | `20` | Simulated session pairs per (param, shift). |
| `--n-restarts` | `5` | MAP optimisation restarts per fit. |
| `--free` | none | Additionally free `omega3` and/or `kappa`. |
| `--out` | `results/hgf/` (or `results/hgf_extended/` when `--free` given) | Output directory. |

> Environment pins matter here — `numpyro`-latest breaks `pyhgf` (see the HGF env note).
> A quick core sanity check: `python scripts/hgf_smoke.py` (no parameters).

---

## 7. Streamlit UI

### `ui/app.py` — interactive explorer (all plot types in the browser)

PSTH, rasters, ACG, and the three FR-vs-regressor plots, with session/neuron/condition pickers.

- **Default / only form:** `streamlit run ui/app.py`  (run from the repo root)

---

## 8. Utility / one-off scripts (`scripts/`)

| Command | Purpose | Parameters |
|---|---|---|
| `python scripts/generate_test_data.py` | Write a synthetic `test_session/` for testing the loaders/plots. | none |
| `python scripts/mat_to_csv.py` | Convert all session `.mat` files to CSV under `csv/`. | none |
| `python scripts/hgf_smoke.py` | Smoke-test the HGF core (data, model, gradients, simulator). | none |
| `python scripts/report_inventory.py` | Cross-session data inventory → `results/report/session_inventory.csv`. | none |
| `python scripts/report_behaviour.py` | Behavioural learning metrics + figure → `results/report/`. | none |
| `python scripts/report_neural_aggregate.py` | Pool neuron-summary CSVs into report tables/figures. | none |
| `python scripts/report_neural_extra.py` | Extra neural enrichment stats/figures for the report. | none |
| `python scripts/report_to_html.py` | Render `results/report/REPORT.md` → portable `REPORT.html`. | none |
