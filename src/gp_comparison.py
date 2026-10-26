"""Shared Standard-GP and MOE-GP preparation for retrospective comparisons."""

from __future__ import annotations

from copy import deepcopy
import math
import warnings

import gpytorch
import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean

from src.gp_metrics import posterior_metrics
from src.bo.config import GPConfig, MOE_PROFILE
from src.bo.gp_kernel_architectures import GP_ARCHITECTURES, build_kernel_architecture, initialize_lengthscales
from src.bo.objective_transforms import (
    PHYSICAL_TARGET_COLUMNS,
    TargetScaler,
    dataframe_to_objectives,
    dataframe_to_features,
)


SUPPORTED_KERNELS = ("rbf", "matern15", "matern25")


MAX_GP_MAXITER = 10_000
FIT_LENGTHSCALE_STARTS = (None, 0.1, 0.5, 2.0)
# Log-evidence returned when every deterministic start fails to fit. It is far
# below any achievable training log marginal likelihood, so the affected expert
# is down-weighted to ~zero by the Bayesian gating instead of crashing the run.
FAILED_FIT_LOG_EVIDENCE = -1e6


def validate_gp_maxiter(value: int) -> int:
    """Validate the bounded SciPy iteration budget used by MOE expert fits."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_GP_MAXITER
    ):
        raise ValueError(
            f"gp_maxiter must be an integer between 1 and {MAX_GP_MAXITER}."
        )
    return value


def prepare_training_split(dataframe) -> tuple[torch.Tensor, torch.Tensor, TargetScaler]:
    features = prepare_prediction_features(dataframe)
    objectives = dataframe_to_objectives(dataframe)
    scaler = TargetScaler.fit(objectives)
    targets = torch.tensor(scaler.transform(objectives), dtype=torch.double)
    return features, targets, scaler


def prepare_prediction_features(dataframe) -> torch.Tensor:
    """Return fixed-bound normalized features for any train or test dataframe."""
    return torch.tensor(dataframe_to_features(dataframe), dtype=torch.double)


def physical_truth_tensor(dataframe) -> torch.Tensor:
    return torch.tensor(
        dataframe.loc[:, list(PHYSICAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64),
        dtype=torch.double,
    )


def initialized_constant_mean(train_Y: torch.Tensor) -> ConstantMean:
    """Return a learnable constant mean initialized near the observed targets."""
    mean_module = ConstantMean()
    mean_module.initialize(constant=train_Y.detach().mean())
    return mean_module


def build_moe_gp(train_X: torch.Tensor, train_Y: torch.Tensor, *, kernel: str = "matern15", gp_config: GPConfig | None = None) -> SingleTaskGP:
    """Build one common expert architecture on physical training coordinates."""
    supported = SUPPORTED_KERNELS if gp_config is None else GP_ARCHITECTURES
    if kernel not in supported:
        raise ValueError(f"Unknown kernel {kernel!r}; choose from {supported}.")
    if gp_config is not None and not isinstance(gp_config, GPConfig):
        raise TypeError("gp_config must be a GPConfig or None.")
    base_kernel = build_kernel_architecture(kernel, input_dim=train_X.shape[-1],
                                            ard=True if gp_config is None else gp_config.ard)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputDataWarning)
        return SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=torch.full_like(train_Y, MOE_PROFILE.noise_variance if gp_config is None else gp_config.noise_variance),
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=None,
            mean_module=initialized_constant_mean(train_Y),
            covar_module=ScaleKernel(base_kernel),
        )


def fit_exact_gp(
    model: SingleTaskGP,
    *,
    gp_maxiter: int,
    context: str,
) -> float:
    """Fit a shared exact GP from deterministic starts and return its evidence.

    If every deterministic start fails, the model is restored to its
    initialized state and a large negative log-evidence is returned so the
    caller can down-weight the expert instead of aborting.
    """
    maxiter = validate_gp_maxiter(gp_maxiter)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    initial_state = deepcopy(model.state_dict())
    failures: list[Exception] = []

    for lengthscale_start in FIT_LENGTHSCALE_STARTS:
        model.load_state_dict(initial_state)
        if lengthscale_start is not None:
            initialize_lengthscales(model.covar_module.base_kernel, lengthscale_start)
            model.covar_module.initialize(outputscale=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(
                    mll,
                    optimizer_kwargs={"options": {"maxiter": maxiter}},
                    max_attempts=1,
                )
                log_evidence = _training_log_evidence(model, mll)
            except (ModelFittingError, RuntimeError) as exc:
                failures.append(exc)
                continue

        if math.isfinite(log_evidence):
            model.eval()
            return log_evidence
        failures.append(FloatingPointError("non-finite log marginal likelihood"))

    # Every deterministic start failed (e.g. numerical instability on very
    # small training sets). Keep the initialized, prior-centred GP so the
    # expert still yields a valid posterior, and down-weight it in gating
    # rather than crashing the enclosing run.
    model.load_state_dict(initial_state)
    model.eval()
    warnings.warn(
        f"GP fitting failed for {context} after "
        f"{len(FIT_LENGTHSCALE_STARTS)} deterministic starts; "
        "falling back to the initialized prior-centred model.",
        RuntimeWarning,
        stacklevel=2,
    )
    return FAILED_FIT_LOG_EVIDENCE


def _training_log_evidence(
    model: SingleTaskGP,
    mll: gpytorch.mlls.ExactMarginalLogLikelihood,
) -> float:
    """Evaluate training log marginal likelihood for a fitted model."""
    # eval() may replace raw stored inputs with Normalize output. Restore raw
    # inputs before reading them, so training-mode evidence uses one transform.
    model.train()
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(mll(model(train_x), train_y).item())


def train_standard_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    gp_maxiter: int,
    context: str = "standard GP",
    kernel: str = MOE_PROFILE.kernel,
) -> SingleTaskGP:
    """Fit an independent target GP with the same architecture as the experts."""
    model = build_moe_gp(train_X, train_Y, kernel=kernel)
    fit_exact_gp(model, gp_maxiter=gp_maxiter, context=context)
    return model


def predict_standard_models(
    eta_model: SingleTaskGP,
    lifetime_model: SingleTaskGP,
    test_X: torch.Tensor,
    true_physical: torch.Tensor,
    scaler: TargetScaler,
    *, observation_noise: bool = False,
):
    eta_mean, eta_variance = _posterior(eta_model, test_X)
    life_mean, life_variance = _posterior(lifetime_model, test_X)
    mean = torch.stack((eta_mean, life_mean), dim=-1)
    variance = torch.stack((eta_variance, life_variance), dim=-1)
    if observation_noise:
        noise = torch.stack((
            eta_model.likelihood.noise.detach().flatten()[0],
            lifetime_model.likelihood.noise.detach().flatten()[0],
        ))
        variance = variance + noise
    return posterior_metrics(mean, variance, true_physical, scaler)


def predict_moe_model(
    moe_model, test_X, true_physical, scaler: TargetScaler,
    *, observation_noise: bool = False,
):
    posterior = moe_model.posterior(test_X)
    return posterior_metrics(
        posterior.mean,
        posterior.variance + (MOE_PROFILE.noise_variance if observation_noise else 0.0),
        true_physical,
        scaler,
    )


def _posterior(model: SingleTaskGP, test_X: torch.Tensor):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(test_X)
    return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
