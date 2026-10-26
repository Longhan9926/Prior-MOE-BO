"""Prior-centred expert GPs with Bayesian evidence-and-confidence gating."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch

import gpytorch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from linear_operator.operators import RootLinearOperator

from src.bo.objective_transforms import (
    FEATURE_COLUMNS,
    OBJECTIVE_SPEC,
    PHYSICAL_TARGET_COLUMNS,
    TargetScaler,
    dataframe_to_features,
    dataframe_to_objectives,
    denormalize_features,
)
from src.bo.config import (
    DEFAULT_GATE_BETA,
    DEFAULT_GP_MAXITER,
    DEFAULT_PRIOR_STRENGTH,
    GPConfig,
    MOE_PROFILE,
)
from src.gp_comparison import (
    build_moe_gp,
    fit_exact_gp,
    validate_gp_maxiter,
)
from src.prior.functions import (
    load_priors,
    prior_mean_eta_Pk,
    prior_mean_t80_Pk,
    scale_prior_amplitudes,
)


@dataclass
class Expert:
    """单个先验专家，包含两套 GP 和门控所需信息。"""

    key: str
    gp_eta: SingleTaskGP
    gp_t80: SingleTaskGP
    confidence: float
    priors: Dict[str, Dict[str, float]]
    scalers: Dict[str, Dict[str, float]]
    log_evidence_eta: float
    log_evidence_t80: float
    input_space: str = "normalized"
    gp_input_space: str = "normalized"
    confidence_eta: Optional[float] = None
    confidence_t80: Optional[float] = None

    def target_confidence(self, target_index: int) -> float:
        """Use the target-specific assessment when present, otherwise shared confidence."""
        value = self.confidence_eta if target_index == 0 else self.confidence_t80
        return self.confidence if value is None else value

    def physical_features(self, X: torch.Tensor) -> torch.Tensor:
        """Map the public input convention to canonical physical prior inputs."""
        return denormalize_features(X) if self.input_space == "normalized" else X

    def gp_features(self, X: torch.Tensor) -> torch.Tensor:
        return self.physical_features(X) if self.gp_input_space == "physical" else X

    def posterior_eta(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回均值与方差（已叠加先验均值）。"""

        X = _ensure_2d_tensor(X)
        # 训练时我们使用的是 (标准化后的目标) - (标准化先验) 作为 y
        # 因此预测时需要将 GP 的后验（处于“中心化标准化”空间）加回“标准化先验”以回到标准化目标空间。
        physical_X = self.physical_features(X)
        prior_raw = cast(torch.Tensor, prior_mean_eta_Pk(physical_X, self.priors, self.key))
        prior_std = ((prior_raw - self.scalers["eta"]["mean"]) / (self.scalers["eta"]["std"] + 1e-8)).unsqueeze(-1)
        with (nullcontext() if X.requires_grad else torch.no_grad()), gpytorch.settings.fast_pred_var():
            mvn = self.gp_eta.posterior(self.gp_features(X))
        mean = mvn.mean
        var = mvn.variance
        # Coerce to shape (N, 1)
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        elif mean.dim() >= 2 and mean.size(-1) != 1:
            mean = mean[..., :1]
        if var.dim() == 1:
            var = var.unsqueeze(-1)
        elif var.dim() >= 2 and var.size(-1) != 1:
            var = var[..., :1]
        mean = mean + prior_std
        return mean, var

    def posterior_t80(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 log1p(T80) 的均值与方差（含先验）。"""

        X = _ensure_2d_tensor(X)
        physical_X = self.physical_features(X)
        prior_raw = cast(torch.Tensor, prior_mean_t80_Pk(physical_X, self.priors, self.key))
        prior_std = ((prior_raw - self.scalers["t80"]["mean"]) / (self.scalers["t80"]["std"] + 1e-8)).unsqueeze(-1)
        with (nullcontext() if X.requires_grad else torch.no_grad()), gpytorch.settings.fast_pred_var():
            mvn = self.gp_t80.posterior(self.gp_features(X))
        mean = mvn.mean
        var = mvn.variance
        # Coerce to shape (N, 1)
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        elif mean.dim() >= 2 and mean.size(-1) != 1:
            mean = mean[..., :1]
        if var.dim() == 1:
            var = var.unsqueeze(-1)
        elif var.dim() >= 2 and var.size(-1) != 1:
            var = var[..., :1]
        mean = mean + prior_std
        return mean, var


def build_expert_models(
    X: torch.Tensor,
    Y_eta: torch.Tensor,
    Y_t80: torch.Tensor,
    kernel_cfg: Dict,
) -> List[Expert]:
    """根据先验配置构建多名 GP 专家（不使用伪观测）。

    kernel_cfg 至少需要包含 `priors` 字段。可额外传入：
    - maxiter: SciPy GP 优化器的最大迭代数。
    """

    priors = kernel_cfg.get("priors")
    if priors is None:
        raise ValueError("kernel_cfg 必须包含 'priors' 字段。")

    scalers = kernel_cfg.get("scalers")
    if scalers is None:
        raise ValueError("kernel_cfg 必须包含 'scalers' 字段。")
    eta_scaler = scalers["eta"]
    t80_scaler = scalers["t80"]

    maxiter = kernel_cfg.get("maxiter", DEFAULT_GP_MAXITER)
    maxiter = validate_gp_maxiter(maxiter)
    kernel = kernel_cfg.get("kernel", MOE_PROFILE.kernel)
    input_space = kernel_cfg.get("input_space", "normalized")
    if input_space not in ("normalized", "physical"):
        raise ValueError("input_space must be normalized or physical")
    selected_target = kernel_cfg.get("selected_target")
    if selected_target not in (None, 0, 1):
        raise ValueError("selected_target must be None, 0 or 1.")
    gp_config = kernel_cfg.get("gp_config")
    if gp_config is not None and not isinstance(gp_config, GPConfig):
        raise TypeError("gp_config must be a GPConfig or None.")
    gp_options = {} if gp_config is None else {"gp_config": gp_config}

    experts: List[Expert] = []

    # 没有伪观测时，基于 priors 中的每个专家 key 构建一名专家，训练数据仅来自真实观测
    for key, params in priors.items():
        confidence = float(params.get("confidence", 0.5))

        train_X = denormalize_features(X) if input_space == "normalized" else X
        train_eta = Y_eta
        train_t80 = Y_t80

        if selected_target is None:
            physical_train_X = train_X
            prior_eta_raw = prior_mean_eta_Pk(physical_train_X, priors, key)
            prior_t80_raw = prior_mean_t80_Pk(physical_train_X, priors, key)

            prior_eta = ((prior_eta_raw - eta_scaler["mean"]) / (eta_scaler["std"] + 1e-8)).unsqueeze(-1)
            prior_t80 = ((prior_t80_raw - t80_scaler["mean"]) / (t80_scaler["std"] + 1e-8)).unsqueeze(-1)

            centered_eta = train_eta - prior_eta
            centered_t80 = train_t80 - prior_t80

            gp_eta = build_moe_gp(train_X, centered_eta, kernel=kernel, **gp_options)
            gp_t80 = build_moe_gp(train_X, centered_t80, kernel=kernel, **gp_options)

            log_evidence_eta = fit_exact_gp(
                gp_eta,
                gp_maxiter=maxiter,
                context=f"{key}/eta",
            )
            log_evidence_t80 = fit_exact_gp(
                gp_t80,
                gp_maxiter=maxiter,
                context=f"{key}/t80",
            )
        else:
            target_name = "eta" if selected_target == 0 else "t80"
            prior_fn = prior_mean_eta_Pk if selected_target == 0 else prior_mean_t80_Pk
            target_scaler = eta_scaler if selected_target == 0 else t80_scaler
            target_y = train_eta if selected_target == 0 else train_t80
            prior = ((prior_fn(train_X, priors, key) - target_scaler["mean"])
                     / (target_scaler["std"] + 1e-8)).unsqueeze(-1)
            selected_gp = build_moe_gp(train_X, target_y - prior, kernel=kernel, **gp_options)
            evidence = fit_exact_gp(selected_gp, gp_maxiter=maxiter, context=f"{key}/{target_name}")
            # The unused slot is never fitted or queried by single-output MOE.
            unused_gp = build_moe_gp(train_X, torch.zeros_like(target_y), kernel=kernel, **gp_options)
            gp_eta, gp_t80 = ((selected_gp, unused_gp) if selected_target == 0 else (unused_gp, selected_gp))
            log_evidence_eta, log_evidence_t80 = ((evidence, 0.) if selected_target == 0 else (0., evidence))

        expert = Expert(
            key=key,
            gp_eta=gp_eta,
            gp_t80=gp_t80,
            confidence=confidence,
            priors=priors,
            scalers=scalers,
            log_evidence_eta=log_evidence_eta,
            log_evidence_t80=log_evidence_t80,
            input_space=input_space,
            gp_input_space="physical",
            confidence_eta=float(params["confidence_eta"]) if "confidence_eta" in params else None,
            confidence_t80=float(params["confidence_t80"]) if "confidence_t80" in params else None,
        )

        experts.append(expert)

    return experts


def _ensure_2d_tensor(x: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.to(dtype=torch.double)
    else:
        tensor = torch.tensor(list(x), dtype=torch.double)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _compute_bayesian_weights(
    experts: List[Expert],
    X: torch.Tensor,
    beta: float = DEFAULT_GATE_BETA,
    *,
    target_index: int,
) -> torch.Tensor:
    """Weight experts by each response's evidence and prior confidence."""
    evidence_field = "log_evidence_eta" if target_index == 0 else "log_evidence_t80"
    evidence = X.new_tensor([getattr(expert, evidence_field) for expert in experts])
    confidence = X.new_tensor([expert.target_confidence(target_index) for expert in experts])
    logits = beta * (evidence - evidence.max()) + torch.log(confidence + 1e-6)
    return torch.softmax(logits, dim=0).unsqueeze(-1).expand(-1, X.shape[0])


class MOEModel(Model):
    """Response-specific GP means and weighted within-expert latent variances."""

    def __init__(self, experts: List[Expert], beta: float = DEFAULT_GATE_BETA) -> None:
        super().__init__()
        self.experts = list(experts)
        self.beta = beta

    @property
    def num_outputs(self) -> int:
        return 2

    def set_beta(self, beta: float) -> None:
        self.beta = beta

    def _target_weights(self, X: torch.Tensor, target: int) -> torch.Tensor:
        return _compute_bayesian_weights(self.experts, X, self.beta, target_index=target)

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform=None,
    ) -> Posterior:
        if isinstance(observation_noise, torch.Tensor) or observation_noise:
            raise NotImplementedError("Use latent MOE predictions; add measurement noise when evaluating observations.")
        X = X.to(dtype=torch.double)
        flat = X.reshape(-1, X.shape[-1])
        means, variances = [], []
        for target in (range(2) if output_indices is None else output_indices):
            weights = self._target_weights(flat, target).unsqueeze(-1)
            moments = [(expert.posterior_eta(flat) if target == 0 else expert.posterior_t80(flat))
                       for expert in self.experts]
            means.append((weights * torch.stack([moment[0] for moment in moments])).sum(0))
            variances.append((weights * torch.stack([moment[1] for moment in moments])).sum(0).clamp(min=1e-10))
        mean = torch.cat(means, -1).reshape(*X.shape[:-1], len(means))
        variance = torch.cat(variances, -1).reshape_as(mean)
        root = RootLinearOperator(torch.diag_embed(variance.flatten(-2).sqrt()))
        posterior = GPyTorchPosterior(MultitaskMultivariateNormal(mean, root, interleaved=True))
        return posterior_transform(posterior) if posterior_transform is not None else posterior

    def condition_on_observations(self, X, Y, **kwargs):
        raise NotImplementedError("MOEModel does not implement fantasy conditioning.")


def _training_tensors(df: pd.DataFrame) -> tuple[dict[str, torch.Tensor], TargetScaler]:
    """Transform physical data once, fitting target moments on observed rows."""
    X_cols = list(FEATURE_COLUMNS)
    for c in X_cols + list(PHYSICAL_TARGET_COLUMNS):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' missing in experiments dataframe")

    X = torch.tensor(dataframe_to_features(df), dtype=torch.double)
    objectives = dataframe_to_objectives(df)
    target_scaler = TargetScaler.fit(objectives)
    standardized = np.asarray(target_scaler.transform(objectives))
    Y_eta = torch.tensor(standardized[:, 0:1], dtype=torch.double)
    Y_t80 = torch.tensor(standardized[:, 1:2], dtype=torch.double)

    return {"X": X, "Y_eta": Y_eta, "Y_t80": Y_t80}, target_scaler


def build_models_bundle(
    df: pd.DataFrame,
    priors: Dict[str, Dict[str, float]] | None = None,
    beta: float = DEFAULT_GATE_BETA,
    gp_maxiter: int = DEFAULT_GP_MAXITER,
    kernel: str = MOE_PROFILE.kernel,
    gp_config: GPConfig | None = None,
) -> Dict:
    """Build experts and their MOE using train-only objective moments.

    Explicit priors are effective coefficients; default priors are scaled once.
    """
    if gp_config is not None and not isinstance(gp_config, GPConfig):
        raise TypeError("gp_config must be a GPConfig or None.")
    effective_priors = (
        scale_prior_amplitudes(load_priors(), DEFAULT_PRIOR_STRENGTH)
        if priors is None else priors
    )
    train_tensors, target_scaler = _training_tensors(df)
    X, Y_eta, Y_t80 = (train_tensors[key] for key in ("X", "Y_eta", "Y_t80"))
    scalers = target_scaler.as_expert_scalers()

    kernel_cfg = {
        "priors": effective_priors,
        "scalers": scalers,
        "maxiter": gp_maxiter,
        "kernel": kernel,
    }
    if gp_config is not None:
        kernel_cfg = {**kernel_cfg, "gp_config": gp_config}
    experts = build_expert_models(X, Y_eta, Y_t80, kernel_cfg)
    moe_model = MOEModel(experts, beta=beta)

    result = {
        "experts": experts,
        "moe_model": moe_model,
        "scalers": scalers,
        "target_scaler": target_scaler,
        "objective_spec": OBJECTIVE_SPEC,
        "observation_noise": {
            "mode": "fixed",
            "variance": MOE_PROFILE.noise_variance if gp_config is None else gp_config.noise_variance,
            "scale": "standardized_objective",
        },
        "train_tensors": train_tensors,
        "moe_profile": MOE_PROFILE.metadata(),
    }
    if gp_config is not None:
        result = {**result, "gp_config": gp_config.metadata()}

    return result
