"""Physical design space and default GP/MOE settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any


FEATURE_COLUMNS = ("Er_at_pct", "V_Ru_mL", "T_C", "t_h")


@dataclass(frozen=True)
class FeatureSpec:
    """Authoritative physical design space used by every surrogate and prior."""

    version: str = "oer-feature-space-v3"
    columns: tuple[str, str, str, str] = FEATURE_COLUMNS
    lower: tuple[float, float, float, float] = (1.0, 1.0, 200.0, 1.0)
    upper: tuple[float, float, float, float] = (20.0, 2.0, 600.0, 10.0)
    units: tuple[str, str, str, str] = ("at.%", "mL", "degC", "h")

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "columns": list(self.columns),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "units": list(self.units),
            "normalization": "fixed_bounds_affine_to_unit_hypercube",
        }


FEATURE_SPEC = FeatureSpec()


@dataclass(frozen=True)
class GPConfig:
    """Fixed noise and ARD on observed-training-normalized inputs per GP."""
    noise_variance: float = .03
    ard: bool = True

    def __post_init__(self):
        if (isinstance(self.noise_variance, bool) or not isinstance(self.noise_variance, Real)
                or not math.isfinite(self.noise_variance) or self.noise_variance <= 0):
            raise ValueError('noise_variance must be finite and strictly positive.')
        if not isinstance(self.ard, bool):
            raise ValueError('ard must be a boolean.')

    def metadata(self) -> dict:
        """Return fresh metadata preserving the frozen experiment schema."""
        return {'noise_variance': float(self.noise_variance), 'ard': self.ard,
                'lengthscale_regularization': 'none', 'lengthscale_prior': None}


@dataclass(frozen=True)
class MoeProfile:
    confidence_threshold_eta: float = .9
    confidence_threshold_t80: float = .7
    prior_strength: float = 4.
    gate_beta: float = .1
    gating_mode: str = "per_target"
    confidence_mode: str = "per_target"
    ucb_beta: float = 1.5
    gp_maxiter: int = 300
    kernel: str = "matern15"
    noise_variance: float = .03
    uncertainty_mode: str = "diagonal"
    input_transform: str = "observed_physical_normalize"
    outcome_transform: str = "none"
    prior_coordinates: tuple[str, ...] = ("Er_at_pct", "V_Ru_mL", "T_C", "t_h")
    prior_functions: int = 49
    standard_ucb_beta: float = 1.5
    acquisition: str = "ucb_hypervolume_improvement"
    acquisition_fallback: bool = False

    def metadata(self) -> dict:
        return asdict(self)


MOE_PROFILE = MoeProfile()
DEFAULT_PRIORS_PATH = "src/prior/priors.json"
DEFAULT_GATE_BETA = MOE_PROFILE.gate_beta
DEFAULT_UCB_BETA = MOE_PROFILE.ucb_beta
DEFAULT_GP_MAXITER = MOE_PROFILE.gp_maxiter
DEFAULT_PRIOR_STRENGTH = MOE_PROFILE.prior_strength
DEFAULT_STANDARD_UCB_BETA = MOE_PROFILE.standard_ucb_beta
