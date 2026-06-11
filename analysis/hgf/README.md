# HGF analysis — `analysis/hgf/`

Fits a **3-level binary Hierarchical Gaussian Filter** (pyhgf 0.2.12) to one
participant's two-armed bandit choices recorded across multiple sessions of an
intracranial EEG study (Simon Jacob's lab).

---

## Scientific rationale

The participant sees a gamble arm (win probability changes over time, unknown)
and a safe arm (fixed reward). Only gamble-choice trials yield outcome
information, so the belief update is **partial-feedback**. The HGF models
subjective learning: level 2 tracks the estimated gamble win-probability, level
3 tracks perceived environmental volatility.

The key latent variable is `perceived_gamble_prob` (= the HGF level-2
predicted mean `x_0_expected_mean` in pyhgf notation): the participant's
trial-by-trial internal estimate of `P(gamble pays big reward)`.

---

## Model

### Perceptual model — 3-level binary HGF
| Level | Quantity | Role |
|---|---|---|
| 1 | Binary input node | Gamble big-reward outcome (0/1, partial) |
| 2 | μ₂ — logit win-prob belief | **Key latent — the perceptual state** |
| 3 | μ₃ — log-volatility | Meta-volatility / learning-rate modulation |

**Fixed parameters:** κ = 1 (volatility coupling), ω₃ = −4 (meta-volatility).

### Response model
```
P(choose gamble) = σ(β · (p̂ · R_gamble − R_safe) + bias)
```
where `R_gamble = 4`, `R_safe = 1` (fixed per task design), and `p̂` is
the predicted level-2 mean (= perceived gamble win-probability).

### Free parameters
| Symbol | Space | Prior |
|---|---|---|
| ω₂ | unconstrained (log-volatility, typ. negative) | N(−3, 2) |
| β | log-space; `log_beta ~ N(log 2, 1)` ↔ β ∈ (0.05, 30) | |
| bias | unconstrained | N(0, 2) |

### Belief continuity
Beliefs are **reset** to the prior at each session boundary. Sessions are
weeks apart with re-randomised schedules, so continuity across sessions is
not warranted.

---

## Package layout

| Module | Contents |
|---|---|
| `config.py` | Constants, priors, bounds, `to_natural`, `log_prior` |
| `data.py` | `SessionData`, `load_session`, `list_sessions` |
| `model.py` | `build_hgf`, `SessionModel`, `simulate_session`, response fn |
| `fit.py` | `shared_map_fit`, `separate_map_fit`, `FitResult`, `bic` |
| `trajectories.py` | `session_trajectory` → tidy per-trial DataFrame |
| `comparison.py` | `fit_rw`, `compare_models` (HGF vs RW vs RW+stickiness) |
| `recovery.py` | `parameter_recovery`, `recovery_summary` |
| `power.py` | `power_check`, `power_summary` |
| `hierarchical.py` | `run_hierarchical` (PyMC/NUTS partial-pooling model) |
| `plots.py` | All figure functions — return `(fig, axes)`, optionally save |
| `run.py` | End-to-end pipeline CLI |

---

## Running the pipeline

```bash
# Full pipeline (all sessions, including hierarchical NUTS)
python -m analysis.hgf.run

# Skip slow hierarchical fit
python -m analysis.hgf.run --no-hierarchical

# Subset of sessions, custom output directory
python -m analysis.hgf.run --sessions 20250714 20250721 --out results/hgf_subset/

# Reduce stochastic validation draws for a quick check
python -m analysis.hgf.run --no-hierarchical --n-recovery 8 --n-power 6
```

---

## Outputs (`results/hgf/`)

| File | Content |
|---|---|
| `fitted_parameters.csv` | Shared + per-session MAP estimates, BIC, accuracy |
| `trajectory_<sid>.csv` | Per-trial latent variables (one file per session) |
| `model_comparison.csv` | HGF vs RW vs RW+stickiness — loglik, BIC, ΔBIC |
| `parameter_recovery.csv` | Raw simulate→refit records |
| `parameter_recovery_summary.csv` | Pearson r, RMSE, bias per free parameter |
| `power_check.csv` | Raw simulation pairs |
| `power_check_summary.csv` | Power (prop. correct sign) per (param, shift size) |
| `hierarchical_summary.csv` | Posterior mean/SD/HDI per parameter |
| `hierarchical_samples.csv` | Tidy posterior draws in natural-param space |
| `figures/trajectory_<sid>.png` | p̂ trace + choices + true schedule |
| `figures/learning_rate_<sid>.png` | Learning rate and uncertainty traces |
| `figures/volatility_<sid>.png` | μ₃ and μ₂ traces |
| `figures/parameter_drift.png` | Per-session vs shared MAP estimates |
| `figures/model_comparison.png` | ΔBIC bar chart |
| `figures/parameter_recovery.png` | True vs recovered scatter (3 panels) |
| `figures/power_curve.png` | Power vs shift magnitude |
| `figures/posterior_overview.png` | Violin posteriors per session (hierarchical) |

### Trajectory CSV columns

| Column | Meaning |
|---|---|
| `perceived_gamble_prob` | p̂ — predicted win-prob (the headline latent) |
| `mu2` | Posterior mean of level-2 belief (logit space) |
| `mu2_hat` | Predicted (prior) mean of level-2 before update |
| `sa2` | Predicted variance (1/π̂₂) — belief uncertainty |
| `mu3` | Posterior mean of level-3 (log-volatility) |
| `learning_rate` | 1/π₂ (inverse posterior precision) |
| `delta1` | Level-1 prediction error (outcome − p̂) |
| `delta2` | Level-2 prediction error (μ₂ − μ̂₂) |
| `p_choose_gamble` | Response model's P(choose gamble) |
| `actual_choice` | Participant's choice (1 = gamble, 0 = safe) |
| `gamble_observed` | 1 if gamble was chosen (outcome observed) |
| `gamble_outcome` | Observed big-reward outcome (NaN if unobserved) |
| `true_p_schedule` | Task's scheduled P(big reward) |
| `original_trial_index` | Pre-filter trial index |

---

## Validation checklist

| Check | Where |
|---|---|
| Gradient flows through HGF + response model | `scripts/hgf_smoke.py` |
| `scan_fn` single step ≡ full sequence | `scripts/hgf_validate_stepper.py` |
| Shared MAP fit on real data — sane params + accuracy | `scripts/hgf_test_fit.py` |
| Parameter recovery r > 0.7 for all params | `results/hgf/parameter_recovery_summary.csv` |
| Power: β detectable (power→1.0 at Δβ=1.0); ω₂ underpowered (≤0.6 even at Δω₂=2.0) — see note | `results/hgf/power_check_summary.csv` |
| HGF ΔBIC vs RW | `results/hgf/model_comparison.csv` |
| Hierarchical posterior R̂ ≈ 1 | `results/hgf/hierarchical_summary.csv` |

---

## Environment

```
pyhgf==0.2.12   jax==0.4.31   jaxlib==0.4.31
numpyro==0.15.3  pymc==5.28.5
```

> **Pin these versions.** numpyro ≥ 0.16 upgrades jax to ≥ 0.10 which breaks
> pyhgf's `HGF.input_data()` partial-feedback path.
