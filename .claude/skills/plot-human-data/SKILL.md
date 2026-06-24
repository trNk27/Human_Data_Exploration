---
name: plot-human-data
description: >-
  Generate any figure the Human Data neural codebase can produce — PSTHs,
  rasters, autocorrelograms, ZETA responsiveness/outcome detections, cumulative
  spike-count + ZETA plots, firing-rate-vs-regressor scatters (behavioural
  P(reward), HGF belief p̂, prediction error δ₁), population heatmaps, and
  behaviour / Rescorla-Wagner / HGF model plots — from a plain-English request.
  Use whenever the user asks to plot, visualise, draw, or "make a PNG/figure"
  for a neuron, session, brain region, event, outcome, or analysis in this repo.
---

# Plotting the Human Data project from natural language

This repo turns a two-armed risky-choice task with intracranial recordings into
figures. Your job: map the user's plain-English ask to the correct command,
run it so a PNG is written, then tell them the path. **[COMMANDS.md](../../../COMMANDS.md)
is the exhaustive parameter reference — consult it for every flag.** This skill
is the router and the environment cheat-sheet; it does not repeat COMMANDS.md.

## Step 0 — environment (read this first, it WILL bite you)

`conda` and `python` are **not on PATH**. Always call the env interpreter directly:

```
C:\Users\mstammler\.conda\envs\humandata\python.exe
```

If that path ever moves, rediscover it:
`Get-ChildItem C:\Users\mstammler\.conda\envs -Recurse -Filter python.exe -Depth 2`

Run everything **from the repo root** with a headless Matplotlib backend so figures
save without opening a blocking window:

```powershell
$env:MPLBACKEND="Agg"; & "C:\Users\mstammler\.conda\envs\humandata\python.exe" -m viewers.psth --area ACC --event reward --save
```

- **A figure is only produced if you ask for it.** CLIs need `--save`; the fluent
  Python API needs `.save()`. Without these, nothing is written.
- **Suppress zetapy's noisy stderr** with `2>$null` in PowerShell — it logs heavily.
- Follow repo convention: **write Python to a `.py` file in `scripts/` and run it**,
  never `python -c "…"` or heredocs (see CLAUDE.md).
- **ZETA `boolParallel` trap:** `zetapy.zetatest` defaults to `boolParallel=True`,
  which on Windows re-imports the calling module in child processes. If you call
  `zetatest` directly in a script, pass `boolParallel=False` **and** guard the
  entry under `if __name__ == "__main__":`, or it fan-outs explosively. The
  `analysis/` ZETA CLIs already handle this — prefer them.

## Step 1 — pick the tool from what the user wants

| User asks for… | Use | Where the PNG lands |
|---|---|---|
| PSTH / firing rate over time, **one named neuron** | fluent API: `sess.neuron(i).psth(align=…).save()` | `results/figures/<session>/` |
| PSTH **across many neurons / a region** | `python -m viewers.psth --area … --event … --save` | `results/<prefix>_<session>.png` |
| Spike **raster** (aligned or full recording) | `.raster()` or `python -m viewers.raster_plot … --save` | as above |
| **Autocorrelogram / ACG** | `.acg()` or `python -m viewers.autocorrelogram --save`; batch → `analysis/export_acg.py` | as above / `acg_export/<session>/` |
| "Which neurons **respond to** cue/reward/… (ZETA)?" | `python analysis/zeta_analysis.py --event reward --csv --save` | `results/zeta_responsiveness/` |
| "A neuron that responds to reward, **with cumulative spike count + ZETA detection**" | `python scripts/zeta_reward_demo.py` (see recipe) | `results/figures/<session>/` |
| "Does the response **differ by outcome** (reward vs no-reward, gamble vs safe)?" | `python analysis/zeta_outcome.py --contrast reward --csv --save` | `results/zeta_outcome/` |
| Firing rate **vs perceived P(reward)** (behavioural) | `.fr_vs_p()` or `python -m viewers.firing_rate_vs_perc_p --save` | `results/figures/<session>/` / `results/` |
| Firing rate **vs HGF belief p̂** / **prediction error δ₁** | `.fr_vs_hgf_p()` / `.fr_vs_delta1()` or the matching viewer | needs HGF CSV first (below) |
| **Population heatmap** (neurons × time, z-scored) | `python -m analysis.population_heatmap --save` | `results/` |
| Is responsiveness **region-dependent**? | `python -m analysis.responsive_region --plot` (run ZETA first) | `results/zeta_responsiveness/` |
| Raw **choice behaviour** over the reward schedule | `python -m analysis.behaviour_overview --save` | `results/figures/…` |
| Choices vs a **Rescorla-Wagner** model / RW fit | `analysis.choice_timeline` / `analysis/fit_rw.py --save` | `results/figures/…` |
| Interactive browse (no PNG) | `python -m viewers.browser` or `streamlit run ui/app.py` | — |

If the request is ambiguous between "one neuron" and "many", default to the
**fluent API for a specific neuron index** and the **viewer CLI for a region/all**.

## Step 2 — translate the user's words into parameters

- **Session** `YYYYMMDD` — one of the dated dirs. Default is `20250714` (`utils.SESSION`); `--session` overrides. List them with `Glob 20*`.
- **Align event** (`--event` / `align=`): `cue`, `response`, `reward`, `trial_start`.
- **Outcome condition** (`condition=` / `--by-condition`): `G+R` gamble+rewarded, `G+N` gamble+no-reward, `S+R` safe+rewarded; `None`=all, `"all"`=overlay the three.
- **Brain region** (`--area`, substring match on the label): e.g. `ACC`, `MFG`, `IFG`, `AG`, `SMG`. `--list` prints every neuron index + label.
- **Firing-rate window** (`--window`): `trial`, `cue_to_reward`, `reward_to_end`.
- Time params are **milliseconds** (`pre_ms`/`post_ms`/`bin_ms`/`sigma_ms`, `--pre`/`--post`/`--bin`/`--sigma`); spike/event times stay in seconds.

## Step 3 — two ways to drive it

**A. CLI one-liner** — best for grids, regions, whole-session and aggregate analyses.
No code to write; just add `--save`. See the table above and COMMANDS.md for flags.

**B. Fluent Python API** — best for one neuron with precise control. Write a small
file in `scripts/` and run it:

```python
# scripts/my_plot.py
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session import Session

sess = Session("20250714")
sess.neuron(7).psth(condition="G+R", align="reward", sigma_ms=50).save()
sess.neuron(7).raster(align="reward").save()
# many neurons, same draw:
from explore import grid
grid(sess, "psth", area="ACC", align="reward", condition="all").save("acc_psth.png")
```

`NeuronView` methods: `.psth() .raster() .acg() .fr_vs_p() .fr_vs_hgf_p() .fr_vs_delta1()`
— full signatures in COMMANDS.md §2. `grid(sess, kind, neurons=…, area=…, **kw)`
tiles any of those across neurons.

## "Find a neuron that <does X>" requests

The user often won't name an index. Resolve it from data, don't guess:

- **Responds to an event** → the ZETA responsiveness CSV ranks neurons by p-value.
  Read `results/zeta_responsiveness/zeta_<event>_<session>.csv` if present (columns:
  `neuron_idx,label,p_zeta,zeta,latency_s,peak_onset_s`, already sorted), else run
  `zeta_analysis.py --event <event> --csv` first. The sign of the response
  (excitation vs inhibition) is **not** in the CSV — re-run `zetatest` on the
  candidate and check `dZETA["dblZETADeviation"]` (>0 = excitation).
- **Differs by outcome** → `results/zeta_outcome/zeta2_<contrast>_<session>.csv`.
- **Tracks a regressor** → eyeball the `fr_vs_*` grid for the region, or sort by the
  regression r in those plots.

## Worked recipe: reward response + cumulative spike count + ZETA detection

This exact request already has a reusable script — [scripts/zeta_reward_demo.py](../../../scripts/zeta_reward_demo.py).
It runs one-sample ZETA on reward onset for one neuron and writes **two** PNGs to
`results/figures/<session>/`: the reward-aligned PSTH, and a two-panel
cumulative-spike-count (`vecRealFrac` vs `vecRealFracLinear`) + ZETA-deviation
(`vecRealDeviation` with the jittered null and the ZETA peak) figure.

```powershell
$env:MPLBACKEND="Agg"; & "C:\Users\mstammler\.conda\envs\humandata\python.exe" scripts\zeta_reward_demo.py --session 20250521 --neuron 113 2>$null
```

`--neuron` defaults to a known strong reward unit (113 = `AG ele051`, single unit,
excitatory). Pick a different one from the ZETA CSV (above). Use this script as the
template for any "PSTH + the ZETA internals" figure for another event by swapping the
event passed to `zetatest`/`responding_event_times`.

## HGF-dependent plots need a trajectory CSV

`fr_vs_hgf_p` and `fr_vs_delta1` read `results/hgf/trajectory_<session>.csv`. If it's
missing, the plot shows a "no HGF trajectory" placeholder — build it first:
`python -m analysis.hgf.run --sessions <id> --no-hierarchical` (COMMANDS.md §6).
**Env pins matter:** `numpyro`-latest breaks `pyhgf` (see the HGF env memory note).

## After you produce a figure

State the **path(s)** written and one line on what the figure shows. If you had to
choose a session/neuron/window the user didn't specify, say so and offer to redo it
with different choices.
