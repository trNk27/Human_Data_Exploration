"""Configuration for the HGF shadowing pipeline.

Single source of truth for: the fixed reward magnitudes, the perceptual/response
model's fixed and free parameters, the priors used for the MAP fit and the
hierarchical model, the random seed, and the output directory.

Parameter conventions
---------------------
The model has three FREE parameters, listed in :data:`FREE_PARAMS`:

  * ``omega2`` — tonic (log-)volatility of HGF level 2 (perceptual learning).
  * ``beta``   — choice slope / inverse temperature (response model), > 0.
  * ``bias``   — additive gamble-vs-safe preference (response model).

Optimisation works in an UNCONSTRAINED vector ``theta`` of length 3:

  ``theta = [omega2, log_beta, bias]``  with ``beta = exp(log_beta)``.

The ``exp`` link keeps ``beta`` positive. :func:`to_natural` / :func:`to_theta`
convert between the two representations, and :func:`log_prior` evaluates the
(independent Gaussian) prior in ``theta`` space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Repo root = three levels up from this file (analysis/hgf/config.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Fixed task structure
# ---------------------------------------------------------------------------

#: Reward magnitudes are NOT stored in the data (columns 8/9 are empty), so they
#: are hardcoded per the project spec.
REWARD_GAMBLE: float = 4.0
REWARD_SAFE: float = 1.0

#: Perceptual model depth (3-level binary HGF).
N_LEVELS: int = 3

# ---------------------------------------------------------------------------
# Fixed HGF parameters (not inferred)
# ---------------------------------------------------------------------------

OMEGA3: float = -4.0          # level-3 tonic volatility (meta-volatility)
KAPPA: float = 1.0            # volatility coupling between levels 2 and 3
TONIC_VOLATILITY_1: float = 0.0  # binary state node (unused for binary HGF)

# Initial beliefs (reset at the start of every session).
INIT_MEAN: dict = {"1": 0.0, "2": 0.0, "3": 0.0}
INIT_PRECISION: dict = {"1": 1.0, "2": 1.0, "3": 1.0}

# ---------------------------------------------------------------------------
# Free parameters and their priors (reported in the README)
# ---------------------------------------------------------------------------

FREE_PARAMS: tuple[str, ...] = ("omega2", "beta", "bias")
THETA_NAMES: tuple[str, ...] = ("omega2", "log_beta", "bias")


@dataclass(frozen=True)
class GaussianPrior:
    """A 1-D Gaussian prior (mean, sd) in the unconstrained ``theta`` space."""

    mean: float
    sd: float

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return -0.5 * np.log(2 * np.pi * self.sd ** 2) - 0.5 * ((x - self.mean) / self.sd) ** 2


#: Priors in theta-space: omega2, log_beta, bias (independent Gaussians).
PRIORS: dict[str, GaussianPrior] = {
    "omega2": GaussianPrior(mean=-3.0, sd=2.0),
    "log_beta": GaussianPrior(mean=float(np.log(2.0)), sd=1.0),
    "bias": GaussianPrior(mean=0.0, sd=2.0),
}

#: A reasonable starting point for optimisation (theta space).
THETA_INIT: np.ndarray = np.array(
    [PRIORS["omega2"].mean, PRIORS["log_beta"].mean, PRIORS["bias"].mean],
    dtype=float,
)

#: Box bounds on theta = [omega2, log_beta, bias], to keep the optimiser out of
#: pathological high-volatility / extreme-slope regions where the filter degenerates.
#: omega2 is a log-volatility; > 2 implies implausibly fast belief swings.
THETA_BOUNDS: list[tuple[float, float]] = [
    (-10.0, 2.0),                       # omega2
    (float(np.log(0.05)), float(np.log(30.0))),  # log_beta -> beta in [0.05, 30]
    (-10.0, 10.0),                      # bias
]

# ---------------------------------------------------------------------------
# Reproducibility & output
# ---------------------------------------------------------------------------

SEED: int = 20250714
RESULTS_HGF: str = os.path.join(REPO_ROOT, "results", "hgf")


# ---------------------------------------------------------------------------
# Parameter transforms
# ---------------------------------------------------------------------------

def to_natural(theta: np.ndarray) -> dict[str, float]:
    """Map an unconstrained ``theta`` vector to natural parameters.

    ``theta = [omega2, log_beta, bias] -> {omega2, beta, bias}``.
    """
    theta = np.asarray(theta, dtype=float)
    return {"omega2": float(theta[0]), "beta": float(np.exp(theta[1])), "bias": float(theta[2])}


def to_theta(natural: dict[str, float]) -> np.ndarray:
    """Inverse of :func:`to_natural`."""
    return np.array(
        [natural["omega2"], np.log(natural["beta"]), natural["bias"]],
        dtype=float,
    )


def log_prior(theta: np.ndarray) -> float:
    """Sum of independent Gaussian log-priors over ``theta``."""
    theta = np.asarray(theta, dtype=float)
    return float(
        PRIORS["omega2"].logpdf(theta[0])
        + PRIORS["log_beta"].logpdf(theta[1])
        + PRIORS["bias"].logpdf(theta[2])
    )


def ensure_results_dir() -> str:
    """Create (if needed) and return the HGF results directory."""
    os.makedirs(RESULTS_HGF, exist_ok=True)
    return RESULTS_HGF
