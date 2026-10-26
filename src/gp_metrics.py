"""Posterior conversion and scale-explicit predictive metrics."""

from __future__ import annotations

import torch
from torch.nn.functional import gaussian_nll_loss
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.bo.objective_transforms import (
    TargetScaler,
    objectives_to_physical,
    physical_to_objectives,
)


def posterior_metrics(
    mean_standardized: torch.Tensor,
    variance_standardized: torch.Tensor,
    true_physical: torch.Tensor,
    scaler: TargetScaler,
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    """Return scale-explicit errors and objective-space Gaussian NLL."""
    if mean_standardized.shape != variance_standardized.shape:
        raise ValueError("Posterior mean and variance must have identical shapes.")
    if mean_standardized.ndim != 2 or mean_standardized.shape[-1] != 2:
        raise ValueError("Posterior tensors must have shape [N, 2].")
    if true_physical.shape != mean_standardized.shape:
        raise ValueError("Physical truth must have shape [N, 2].")

    objective_mean = scaler.inverse(mean_standardized)
    objective_variance = scaler.variance_to_objectives(variance_standardized)
    physical_mean = objectives_to_physical(objective_mean)
    true_objectives = physical_to_objectives(true_physical)

    eta = _single_target_metrics(
        physical_mean[:, 0],
        true_physical[:, 0],
        objective_mean[:, 0],
        objective_variance[:, 0],
        true_objectives[:, 0],
        error_scale="V",
        nll_scale="negative_V",
    )
    lifetime = _single_target_metrics(
        objective_mean[:, 1],
        true_objectives[:, 1],
        objective_mean[:, 1],
        objective_variance[:, 1],
        true_objectives[:, 1],
        error_scale="log1p_h",
        nll_scale="log1p_h",
        supplementary_physical=(physical_mean[:, 1], true_physical[:, 1]),
    )
    return eta, lifetime


def _single_target_metrics(
    report_mean: torch.Tensor,
    report_truth: torch.Tensor,
    objective_mean: torch.Tensor,
    objective_variance: torch.Tensor,
    objective_truth: torch.Tensor,
    *,
    error_scale: str,
    nll_scale: str,
    supplementary_physical: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float | int | str]:
    report_error = report_mean - report_truth
    variance = torch.clamp(objective_variance, min=1e-10)
    nll = gaussian_nll_loss(
        objective_mean, objective_truth, variance, full=True, eps=1e-10, reduction='mean'
    )
    prediction = report_mean.detach().cpu().numpy()
    truth = report_truth.detach().cpu().numpy()
    metrics: dict[str, float | int | str] = {
        "rmse": float(root_mean_squared_error(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "nll": float(nll.item()),
        "error_scale": error_scale,
        "nll_scale": nll_scale,
        "r2_n": report_truth.numel(),
        "r2_truth_sum": float(report_truth.sum().item()),
        "r2_truth_sum_squares": float(report_truth.square().sum().item()),
        "r2_sse": float(report_error.square().sum().item()),
    }
    if supplementary_physical is not None:
        physical_mean, physical_truth = supplementary_physical
        physical_prediction = physical_mean.detach().cpu().numpy()
        physical_truth_values = physical_truth.detach().cpu().numpy()
        metrics = {
            **metrics,
            "physical_rmse_h": float(root_mean_squared_error(physical_truth_values, physical_prediction)),
            "physical_mae_h": float(mean_absolute_error(physical_truth_values, physical_prediction)),
        }
    return metrics
