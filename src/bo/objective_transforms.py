"""Canonical feature and objective semantics for the OER BO workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from botorch.utils.transforms import normalize, unnormalize

from src.bo.config import FEATURE_COLUMNS, FEATURE_SPEC

PHYSICAL_TARGET_COLUMNS = ("eta500_V", "T80_h")
OBJECTIVE_COLUMNS = ("neg_eta500_V", "log1p_T80_h")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Versioned definition shared by training, acquisition, metrics, and plots."""

    version: str = "oer-objectives-v1"
    reference_potential_v: float = 3.0
    reference_lifetime_h: float = 0.0

    @property
    def reference_physical(self) -> np.ndarray:
        return np.array(
            [self.reference_potential_v, self.reference_lifetime_h],
            dtype=np.float64,
        )

    @property
    def reference_objectives(self) -> np.ndarray:
        return np.array(
            [-self.reference_potential_v, np.log1p(self.reference_lifetime_h)],
            dtype=np.float64,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_space": FEATURE_SPEC.metadata(),
            "physical_target_columns": list(PHYSICAL_TARGET_COLUMNS),
            "objective_columns": list(OBJECTIVE_COLUMNS),
            "activity_transform": "negate",
            "lifetime_transform": "natural_log1p_hours",
            "reference_point_physical": self.reference_physical.tolist(),
            "reference_point_objectives": self.reference_objectives.tolist(),
        }


OBJECTIVE_SPEC = ObjectiveSpec()


@dataclass(frozen=True)
class TargetScaler:
    """Immutable population mean/std scaler for the two model objectives."""

    mean: tuple[float, float]
    std: tuple[float, float]
    eps: float = 1e-8

    @classmethod
    def fit(cls, objectives: np.ndarray | torch.Tensor, eps: float = 1e-8) -> "TargetScaler":
        values = _as_numpy_objectives(objectives)
        fitted = StandardScaler().fit(values)
        mean = fitted.mean_
        # scale_ uses 1 for constants; our frozen contract uses eps.
        std = np.ones(values.shape[-1]) if len(values) == 1 else np.sqrt(fitted.var_)
        std = np.where(std < eps, eps, std)
        return cls(tuple(float(x) for x in mean), tuple(float(x) for x in std), eps)

    def transform(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return _affine(values, self.mean, self.std, inverse=False)

    def inverse(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return _affine(values, self.mean, self.std, inverse=True)

    def variance_to_objectives(
        self,
        variance: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(variance, torch.Tensor):
            scale = torch.as_tensor(self.std, dtype=variance.dtype, device=variance.device)
            return variance * scale.square()
        return np.asarray(variance, dtype=np.float64) * np.square(np.asarray(self.std))

    def reference_standardized(self) -> np.ndarray:
        return np.asarray(self.transform(OBJECTIVE_SPEC.reference_objectives))

    def as_expert_scalers(self) -> dict[str, dict[str, float]]:
        """Return the per-target scaler structure consumed by expert GPs."""
        return {
            "eta": {"mean": self.mean[0], "std": self.std[0]},
            "t80": {"mean": self.mean[1], "std": self.std[1]},
        }

    def metadata(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


def physical_reference_point() -> np.ndarray:
    return OBJECTIVE_SPEC.reference_physical.copy()


def normalize_features(
    physical: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Map physical design variables to the fixed unit hypercube."""
    _validate_feature_array(physical, physical_space=True)
    return _feature_affine(physical, inverse=False)


def denormalize_features(
    normalized: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Map fixed-bound unit-hypercube coordinates to physical variables."""
    _validate_feature_array(normalized, physical_space=False)
    return _feature_affine(normalized, inverse=True)


def dataframe_to_features(dataframe, *, normalized: bool = True) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in dataframe]
    if missing:
        raise KeyError(f"Missing physical feature columns: {missing}")
    physical = dataframe.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not normalized:
        _validate_feature_array(physical, physical_space=True)
        return physical
    return np.asarray(normalize_features(physical))


def physical_to_objectives(
    physical: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Map ``[potential_V, lifetime_h]`` to two maximization objectives."""
    _validate_last_dimension(physical)
    if isinstance(physical, torch.Tensor):
        if not torch.isfinite(physical).all():
            raise ValueError("Physical targets must be finite.")
        if torch.any(physical[..., 1] < 0):
            raise ValueError("Lifetime must be non-negative hours.")
        return torch.stack((-physical[..., 0], torch.log1p(physical[..., 1])), dim=-1)

    values = np.asarray(physical, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Physical targets must be finite.")
    if np.any(values[..., 1] < 0):
        raise ValueError("Lifetime must be non-negative hours.")
    return np.stack((-values[..., 0], np.log1p(values[..., 1])), axis=-1)


def objectives_to_physical(
    objectives: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Invert model objectives to ``[potential_V, lifetime_h]``."""
    _validate_last_dimension(objectives)
    if isinstance(objectives, torch.Tensor):
        return torch.stack((-objectives[..., 0], torch.expm1(objectives[..., 1])), dim=-1)
    values = np.asarray(objectives, dtype=np.float64)
    return np.stack((-values[..., 0], np.expm1(values[..., 1])), axis=-1)


def dataframe_to_objectives(dataframe) -> np.ndarray:
    missing = [column for column in PHYSICAL_TARGET_COLUMNS if column not in dataframe]
    if missing:
        raise KeyError(f"Missing physical target columns: {missing}")
    physical = dataframe.loc[:, list(PHYSICAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    return np.asarray(physical_to_objectives(physical))


def _validate_last_dimension(values: np.ndarray | torch.Tensor) -> None:
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError(f"Expected targets with final dimension 2, got shape {tuple(values.shape)}.")


def _validate_feature_array(
    values: np.ndarray | torch.Tensor,
    *,
    physical_space: bool,
) -> None:
    if values.ndim == 0 or values.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected features with final dimension {len(FEATURE_COLUMNS)}, "
            f"got shape {tuple(values.shape)}."
        )
    if isinstance(values, torch.Tensor):
        if not torch.isfinite(values).all():
            raise ValueError("Feature values must be finite.")
        lower = torch.as_tensor(
            FEATURE_SPEC.lower if physical_space else (0.0,) * len(FEATURE_COLUMNS),
            dtype=values.dtype,
            device=values.device,
        )
        upper = torch.as_tensor(
            FEATURE_SPEC.upper if physical_space else (1.0,) * len(FEATURE_COLUMNS),
            dtype=values.dtype,
            device=values.device,
        )
        outside = torch.any((values < lower) | (values > upper))
        if bool(outside.item()):
            space = "authoritative design space" if physical_space else "unit hypercube"
            raise ValueError(f"Feature values fall outside the {space}.")
        return

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Feature values must be finite.")
    lower = np.asarray(FEATURE_SPEC.lower if physical_space else (0.0,) * len(FEATURE_COLUMNS))
    upper = np.asarray(FEATURE_SPEC.upper if physical_space else (1.0,) * len(FEATURE_COLUMNS))
    if np.any((array < lower) | (array > upper)):
        space = "authoritative design space" if physical_space else "unit hypercube"
        raise ValueError(f"Feature values fall outside the {space}.")


def _as_numpy_objectives(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    _validate_last_dimension(array)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("TargetScaler.fit requires a non-empty 2D objective array.")
    if not np.isfinite(array).all():
        raise ValueError("Objectives used to fit a scaler must be finite.")
    return array


def _affine(
    values: np.ndarray | torch.Tensor,
    mean: tuple[float, float],
    std: tuple[float, float],
    *,
    inverse: bool,
) -> np.ndarray | torch.Tensor:
    _validate_last_dimension(values)
    if isinstance(values, torch.Tensor):
        center = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(std, dtype=values.dtype, device=values.device)
    else:
        values = np.asarray(values, dtype=np.float64)
        center = np.asarray(mean, dtype=np.float64)
        scale = np.asarray(std, dtype=np.float64)
    return values * scale + center if inverse else (values - center) / scale


def _feature_affine(
    values: np.ndarray | torch.Tensor,
    *,
    inverse: bool,
) -> np.ndarray | torch.Tensor:
    is_tensor = isinstance(values, torch.Tensor)
    tensor = values if is_tensor else torch.tensor(np.asarray(values, dtype=np.float64))
    bounds = torch.as_tensor(
        (FEATURE_SPEC.lower, FEATURE_SPEC.upper), dtype=tensor.dtype, device=tensor.device
    )
    operation = unnormalize if inverse else normalize
    result = operation(tensor, bounds, update_constant_bounds=False)
    return result if is_tensor else result.numpy()
