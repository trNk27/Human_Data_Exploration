"""HGF shadowing of a single participant's two-armed bandit choices.

A modular pipeline that fits a 3-level binary Hierarchical Gaussian Filter
(``pyhgf``) to ONE participant's choices across multiple sessions and extracts
the latent perceived gamble probability per trial.

Modules
-------
config        Constants, priors, fixed/free parameter spec, output paths.
data          Per-session sequence extraction (perceptual input, observed mask,
              choices) from ``Trials_Sync``.
model         HGF construction, the custom response function, per-session
              surprise, and the differentiable JAX log-likelihood.
fit           Shared-parameter (complete-pooling) MAP fit and separate
              per-session MAP fits.
trajectories  Latent trajectory tables (tidy, one row per trial).
hierarchical  Sessions-within-participant Bayesian model (PyMC + JAX-NUTS).
recovery      Parameter recovery (simulate -> refit).
power         Power check for a session-to-session parameter shift.
comparison    Model comparison vs Rescorla-Wagner (+ stickiness).
plots         Per-session and cross-session figures.
run           End-to-end orchestration CLI.

Because N = 1, all inferences are about THIS participant across sessions, not a
population.
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "model",
    "fit",
    "trajectories",
    "hierarchical",
    "recovery",
    "power",
    "comparison",
    "plots",
]
