"""Shared model evaluation for GP validation workflows."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.bo.objective_transforms import PHYSICAL_TARGET_COLUMNS, denormalize_features
from src.bo.config import MOE_PROFILE
from src.experts_and_moe import build_models_bundle
from src.gp_comparison import (
    SUPPORTED_KERNELS,
    physical_truth_tensor,
    predict_moe_model,
    predict_standard_models,
    prepare_prediction_features,
    prepare_training_split,
    train_standard_gp,
)


ETA_LABEL, T80_LABEL = PHYSICAL_TARGET_COLUMNS
TARGETS = (ETA_LABEL, T80_LABEL)
STANDARD_GP = "Standard GP"
MOE_GP = "MOE GP"
MODELS = (STANDARD_GP, MOE_GP)
OOF_R2_STAT_COLUMNS = (
    "r2_n",
    "r2_truth_sum",
    "r2_truth_sum_squares",
    "r2_sse",
)
KFOLD_SEED_METRICS = ("rmse", "mae", "nll", "r2")


def validate_model_options(kernels):
    """Validate kernel names supplied by comparison commands."""
    if not kernels or any(kernel not in SUPPORTED_KERNELS for kernel in kernels):
        raise ValueError(f"kernels must be selected from {SUPPORTED_KERNELS}.")


def evaluate_standard_split(
    train_dataframe,
    test_dataframe,
    *,
    gp_maxiter: int,
    kernel: str = MOE_PROFILE.kernel,
    observation_noise: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fit and evaluate the two independent Standard GP targets."""
    validate_model_options((kernel,))
    train_x, train_y, target_scaler = prepare_training_split(train_dataframe)
    test_x = prepare_prediction_features(test_dataframe)
    train_x = denormalize_features(train_x)
    test_x = denormalize_features(test_x)
    true_physical = physical_truth_tensor(test_dataframe)
    eta_model = train_standard_gp(
        train_x,
        train_y[:, 0:1],
        gp_maxiter=gp_maxiter,
        kernel=kernel,
        context=f"standard GP/{ETA_LABEL}",
    )
    t80_model = train_standard_gp(
        train_x,
        train_y[:, 1:2],
        gp_maxiter=gp_maxiter,
        kernel=kernel,
        context=f"standard GP/{T80_LABEL}",
    )
    eta_metrics, t80_metrics = predict_standard_models(
        eta_model,
        t80_model,
        test_x,
        true_physical,
        target_scaler,
        observation_noise=observation_noise,
    )
    return {ETA_LABEL: eta_metrics, T80_LABEL: t80_metrics}


def evaluate_moe_split(
    train_dataframe,
    test_dataframe,
    priors,
    *,
    beta: float,
    gp_maxiter: int,
    kernel: str = MOE_PROFILE.kernel,
    observation_noise: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate the MOE using one common covariance kernel for every expert."""
    validate_model_options((kernel,))
    test_x = prepare_prediction_features(test_dataframe)
    true_physical = physical_truth_tensor(test_dataframe)
    bundle = build_models_bundle(
        train_dataframe,
        priors,
        beta=beta,
        gp_maxiter=gp_maxiter,
        kernel=kernel,
    )
    eta_metrics, t80_metrics = predict_moe_model(
        bundle["moe_model"],
        test_x,
        true_physical,
        bundle["target_scaler"],
        observation_noise=observation_noise,
    )
    return {ETA_LABEL: eta_metrics, T80_LABEL: t80_metrics}


def metric_records(
    model: str,
    metrics_by_target: Mapping[str, Mapping[str, Any]],
    **context: Any,
) -> list[dict[str, Any]]:
    """Convert target metrics into the stable long-form result schema."""
    return [
        {
            **context,
            "model": model,
            "target": target,
            **dict(metrics_by_target[target]),
        }
        for target in TARGETS
    ]


def pooled_oof_r2(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate exact R2 after pooling held-out predictions within each seed."""
    required = {"model", "target", "error_scale", *OOF_R2_STAT_COLUMNS}
    missing = sorted(required.difference(results_dataframe.columns))
    if missing:
        raise ValueError(f"K-fold results are missing pooled-R2 statistics: {missing}")

    group_columns = ["model", "target"]
    if "seed" in results_dataframe.columns:
        group_columns.insert(0, "seed")

    records: list[dict[str, Any]] = []
    for group_key, group in results_dataframe.groupby(group_columns, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        context = dict(zip(group_columns, key_values))
        model = context["model"]
        target = context["target"]
        n_oof = int(group["r2_n"].sum())
        if n_oof < 1:
            raise ValueError(f"No out-of-fold samples for {model}/{target}.")
        truth_sum = float(group["r2_truth_sum"].sum())
        truth_sum_squares = float(group["r2_truth_sum_squares"].sum())
        sse = float(group["r2_sse"].sum())
        sst = truth_sum_squares - truth_sum**2 / n_oof
        r2 = float("nan") if sst <= 0.0 else 1.0 - sse / sst
        scales = group["error_scale"].dropna().unique()
        if len(scales) != 1:
            raise ValueError(f"Inconsistent error scales for {model}/{target}.")
        records.append(
            {
                **context,
                "r2": r2,
                "error_scale": scales[0],
                "n_oof": n_oof,
            }
        )
    return pd.DataFrame(records)


def aggregate_kfold_seed_metrics(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Pool fold metrics into one complete out-of-fold result per seed."""
    required = {
        "seed",
        "model",
        "target",
        "rmse",
        "mae",
        "nll",
        "error_scale",
        "nll_scale",
        "r2_n",
        *OOF_R2_STAT_COLUMNS,
    }
    missing = sorted(required.difference(results_dataframe.columns))
    if missing:
        raise ValueError(f"K-fold results are missing seed metrics: {missing}")

    r2_results = pooled_oof_r2(results_dataframe).set_index(
        ["seed", "model", "target"]
    )
    records: list[dict[str, Any]] = []
    group_columns = ["seed", "model", "target"]
    for (seed, model, target), group in results_dataframe.groupby(
        group_columns,
        sort=False,
    ):
        weights = group["r2_n"].to_numpy(dtype=float)
        n_oof = int(weights.sum())
        if n_oof < 1:
            raise ValueError(f"No out-of-fold samples for seed {seed}, {model}/{target}.")
        error_scales = group["error_scale"].dropna().unique()
        nll_scales = group["nll_scale"].dropna().unique()
        if len(error_scales) != 1 or len(nll_scales) != 1:
            raise ValueError(
                f"Inconsistent metric scales for seed {seed}, {model}/{target}."
            )

        r2_row = r2_results.loc[(seed, model, target)]
        record: dict[str, Any] = {
            "seed": int(seed),
            "model": model,
            "target": target,
            "rmse": float(np.sqrt(group["r2_sse"].sum() / n_oof)),
            "mae": float(np.average(group["mae"], weights=weights)),
            "nll": float(np.average(group["nll"], weights=weights)),
            "r2": float(r2_row["r2"]),
            "error_scale": error_scales[0],
            "nll_scale": nll_scales[0],
            "n_oof": n_oof,
        }
        physical_columns = ("physical_rmse_h", "physical_mae_h")
        if all(
            column in group.columns and group[column].notna().all()
            for column in physical_columns
        ):
            record = {
                **record,
                "physical_rmse_h": float(
                    np.sqrt(
                        np.average(
                            group["physical_rmse_h"].to_numpy(dtype=float) ** 2,
                            weights=weights,
                        )
                    )
                ),
                "physical_mae_h": float(
                    np.average(group["physical_mae_h"], weights=weights)
                ),
            }
        records.append(record)
    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


def summarize_kfold_seeds(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return means and sample standard deviations across repeated-CV seeds."""
    required = {"seed", "model", "target", *KFOLD_SEED_METRICS}
    missing = sorted(required.difference(seed_metrics.columns))
    if missing:
        raise ValueError(f"Seed-level K-fold results are missing columns: {missing}")

    group_columns = ["model", "target"]
    aggregate = (
        seed_metrics.groupby(group_columns, sort=False)[list(KFOLD_SEED_METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = group_columns + [
        f"{metric}_{statistic}"
        for metric in KFOLD_SEED_METRICS
        for statistic in ("mean", "std")
    ]
    seed_counts = (
        seed_metrics.groupby(group_columns, sort=False)["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    return seed_counts.merge(aggregate, on=group_columns, validate="one_to_one")
