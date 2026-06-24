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
from dataclasses import dataclass, field

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
# Free-parameter SPACES — let the model free extra HGF parameters (omega3
# meta-volatility, kappa volatility coupling) without rewiring the fitter.
#
# A ParamSpace is an ordered list of FREE parameters plus fixed values for the
# rest. theta is the unconstrained optimisation vector (one entry per free
# parameter); natural dicts always carry all five model parameters, so the rest
# of the pipeline can read natural["omega3"] etc. regardless of what was fit.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """One freeable parameter: natural name, theta-space transform, prior, bounds."""

    name: str            # natural-space name (omega2, omega3, kappa, beta, bias)
    log_space: bool      # theta = log(natural)  (keeps a positive natural value)
    prior: GaussianPrior # Gaussian prior in THETA space
    bounds: tuple        # (lo, hi) box bounds in THETA space
    init: float          # theta-space restart centre
    jitter: float        # theta-space sd for random restarts

    def to_theta(self, natural_value: float) -> float:
        return float(np.log(natural_value)) if self.log_space else float(natural_value)

    def to_natural(self, theta_value: float) -> float:
        return float(np.exp(theta_value)) if self.log_space else float(theta_value)


#: Every parameter the perceptual + response model can free, keyed by natural name.
#: omega2/beta/bias reuse the baseline PRIORS/THETA_BOUNDS so there is one source
#: of truth; omega3/kappa add the meta-volatility and coupling parameters.
PARAM_SPECS: dict[str, "ParamSpec"] = {
    "omega2": ParamSpec("omega2", False, PRIORS["omega2"],   THETA_BOUNDS[0], PRIORS["omega2"].mean,   1.5),
    "omega3": ParamSpec("omega3", False, GaussianPrior(OMEGA3, 2.0), (-10.0, 2.0), OMEGA3,             1.5),
    "kappa":  ParamSpec("kappa",  True,  GaussianPrior(0.0, 0.5), (float(np.log(0.1)), float(np.log(3.0))), 0.0, 0.4),
    "beta":   ParamSpec("beta",   True,  PRIORS["log_beta"], THETA_BOUNDS[1], PRIORS["log_beta"].mean, 0.7),
    "bias":   ParamSpec("bias",   False, PRIORS["bias"],     THETA_BOUNDS[2], PRIORS["bias"].mean,     1.0),
}

#: Natural-space defaults — fill in any parameter that is held fixed.
NATURAL_DEFAULTS: dict[str, float] = {
    "omega2": -3.0,
    "omega3": OMEGA3,
    "kappa": KAPPA,
    "beta": float(np.exp(PRIORS["log_beta"].mean)),
    "bias": 0.0,
}


@dataclass(frozen=True)
class ParamSpace:
    """An ordered set of FREE parameters + fixed natural values for the rest."""

    free: tuple
    fixed: dict = field(default_factory=dict)

    @property
    def specs(self) -> list:
        return [PARAM_SPECS[n] for n in self.free]

    @property
    def k(self) -> int:
        return len(self.free)

    def to_natural(self, theta) -> dict:
        """theta vector -> full natural dict (free from theta, rest from defaults/fixed)."""
        nat = dict(NATURAL_DEFAULTS)
        nat.update(self.fixed)
        for spec, t in zip(self.specs, np.asarray(theta, dtype=float)):
            nat[spec.name] = spec.to_natural(t)
        return nat

    def to_theta(self, natural) -> np.ndarray:
        return np.array([s.to_theta(natural[s.name]) for s in self.specs], dtype=float)

    def perceptual(self, natural) -> tuple:
        """(omega2, omega3, kappa) for the HGF filter, from a natural dict."""
        return (natural["omega2"], natural["omega3"], natural["kappa"])

    def bounds(self) -> list:
        return [s.bounds for s in self.specs]

    def init(self) -> np.ndarray:
        return np.array([s.init for s in self.specs], dtype=float)

    def jitter(self) -> np.ndarray:
        return np.array([s.jitter for s in self.specs], dtype=float)

    def log_prior(self, theta) -> float:
        theta = np.asarray(theta, dtype=float)
        return float(sum(s.prior.logpdf(t) for s, t in zip(self.specs, theta)))


#: Project default — three free parameters (omega2, beta, bias); omega3/kappa fixed.
BASELINE: ParamSpace = ParamSpace(free=("omega2", "beta", "bias"),
                                  fixed={"omega3": OMEGA3, "kappa": KAPPA})
#: Extended — additionally free meta-volatility omega3 and volatility coupling kappa.
EXTENDED: ParamSpace = ParamSpace(free=("omega2", "omega3", "kappa", "beta", "bias"))


def make_param_space(extra_free: list[str] | None) -> ParamSpace:
    """Build a ParamSpace freeing omega2/beta/bias plus any of {omega3, kappa}."""
    extra_free = list(extra_free or [])
    if not extra_free:
        return BASELINE
    free = ("omega2", *extra_free, "beta", "bias")
    fixed = {p: NATURAL_DEFAULTS[p] for p in ("omega3", "kappa") if p not in extra_free}
    return ParamSpace(free=free, fixed=fixed)


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
