# Project: Fit a Hierarchical Gaussian Filter to a SINGLE participant's
# two-armed bandit data across MULTIPLE SESSIONS

## Goal
I have behavioural data from ONE participant who completed a two-armed bandit
task across SEVERAL SESSIONS. The task has a SAFE arm and a VOLATILE "gamble"
arm; rewards are FIXED (deterministic magnitudes), and what changes over trials
is the gamble arm's probability of paying out. I want to "shadow" the
participant's choices with an HGF and extract their latent belief trajectories —
above all the PERCEIVED PROBABILITY of the gamble arm, trial by trial — for
every session. The reward amounts for safe and gamble are for now hardcoded to be 1 and 4 respectively.

Build this in Python using `pyhgf` (JAX-based HGF library). Because this is a
single participant measured repeatedly, the central design choice is how
parameters are shared across sessions (see below). Use MAP point estimates for
the default shared-parameter fit; use pyhgf's PyMC integration for the
hierarchical sessions-within-participant model.

## Important process notes
- Treat the `pyhgf` API as version-sensitive. Install the latest pyhgf, check
  the installed version, and verify the actual API from the package and the docs
  at https://computationalpsychiatry.github.io/pyhgf/ before writing model code.
- Do NOT guess my data format. check the README.md and Claude.md for info
- Pause for my confirmation on the modelling choices (below) before long fits.
- Reproducible: pin deps, set seeds, log versions.

## The data
Is in the session folders labelled by their date. Check out `README.md` and `CLAUDE.md` for more information


## The model

### Perceptual model
A 3-level BINARY HGF tracking the gamble arm's reward probability (confirm the
current pyhgf constructor, e.g. `from pyhgf.model import HGF`,
`HGF(n_levels=3, model_type="binary", ...)`). Since feedback is partial (no information about gamble side if not chosen), use
pyhgf's missing/unobserved-input support so beliefs predict forward on
unsampled trials.

### Response model ("shadowing")
A CUSTOM response function taking the predicted gamble-win probability
p̂ = s(μ̂₂) and the choices y, returning total SURPRISE = −Σ log P(choice).
Decision rule with fixed rewards:
  EV_gamble = p̂ · R_gamble ;  V_safe = R_safe (fixed)
  P(choose gamble) = sigmoid( β·(EV_gamble − V_safe) + bias )

## Parameter-sharing strategy across sessions 
(1) SHARED parameters. One parameter set for the whole participant,
    fit across ALL sessions jointly. Do NOT naively concatenate into one filter
    run — instead run a SEPARATE filter pass per session that SHARES the same
    parameters, and SUM the surprise across sessions for the objective. This
    keeps session boundaries clean. Fit by MAP. This gives the most reliable
    parameters and cleanest trajectories.


### Belief continuity at session boundaries (separate from the above)
Make this an explicit switch, set from the data inspection:
  - RESET beliefs to prior at each session start — default; use if schedules are
    re-randomized per session or sessions are far apart in time.
  - CARRY OVER end-of-session beliefs into the next — use only if sessions are
    continuous over the SAME contingencies and close in time.
  - PARTIAL: carry over belief means but inflate their variance (forgetting).
Recommend a default based on the data inspection and confirm
with me.

### Free parameters — start parsimonious
Begin with ω₂ (perceptual) and β (response) free; fix κ=1 and ω₃/θ to sensible
priors. Free more only if parameter recovery shows identifiability. Report priors.

## Validation
1. Parameter recovery on my actual trial schedule (simulate → refit → recover).
2. POWER CHECK specific to this design: given my per-session trial counts, can a
   plausible session-to-session change in ω₂ or β actually be DETECTED? Simulate
   sessions with a known parameter shift and show whether modes (2)/(3) recover
   it. This tells me if "parameters changed across sessions" is even answerable
   with my data.
3. Posterior predictive checks per session (reproduce choice patterns).
4. Model comparison: vs Rescorla wagner from choice_timeline.py ideally compare it against pure rw, rw with stickiness

## Outputs
1. `fitted_parameters.csv`
2. Per-session latent trajectory tables (tidy CSV), one row per trial:
   perceived_gamble_prob = s(μ₂), mu2, sa2, mu3, learning_rate (1/π₂),
   delta1, delta2, predicted P(choose gamble), actual choice.
3. Plots: per-session perceived gamble probability over trials with actual
   choices (and true schedule if available); learning-rate and volatility
   traces; AND a cross-session plot of the parameter estimates to visualise any
   drift (flat line = stable learner; trend = changing learner). save this in results and label everything accordingly

## Engineering
- Modular package; type hints; docstrings.
- A synthetic end-to-end test asserting recovery.
- README covering setup, how to run, a plain-language description of every latent
  variable, AND an explicit note that with N=1 all inferences are about THIS
  participant across sessions, not a population.