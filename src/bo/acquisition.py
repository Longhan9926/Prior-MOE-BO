"""Hypervolume metrics and UCB acquisition for finite candidate pools."""
from __future__ import annotations

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import t_batch_mode_transform


def hypervolume(objectives, reference_point) -> float:
    values = torch.as_tensor(objectives, dtype=torch.double)
    reference = torch.as_tensor(reference_point, dtype=torch.double)
    if values.ndim != 2 or values.shape[-1] != reference.numel():
        raise ValueError("Objectives and reference point have incompatible shapes.")
    if values.shape[0] == 0:
        return 0.0
    pareto = values[is_non_dominated(values)]
    return float(Hypervolume(ref_point=reference).compute(pareto))


def hypervolume_2d(points, reference):
    """Exact dominated area, ignoring points outside the reference orthant."""
    if points.ndim != 2 or points.shape[-1] != 2 or reference.shape != (2,):
        raise ValueError('Hypervolume requires two objectives and a two-element reference')
    if not torch.isfinite(points).all() or not torch.isfinite(reference).all():
        raise ValueError('Hypervolume inputs must be finite')
    shifted = points - reference
    positive = shifted[(shifted > 0).all(dim=-1)]
    if not len(positive):
        return points.new_zeros(())
    ordered = positive[torch.argsort(positive[:, 0], descending=True)]
    heights = torch.cummax(ordered[:, 1], dim=0).values
    next_x = torch.cat((ordered[1:, 0], ordered.new_zeros(1)))
    return ((ordered[:, 0] - next_x) * heights).sum()


class UCBHypervolumeImprovement(AcquisitionFunction):
    """Optimistic joint hypervolume gain with BoTorch batch and gradient support.

    For q greater than one, scores use the union of optimistic candidates. The
    observed objectives and reference live in the posterior's target units.
    """

    def __init__(self, model, observed, reference, beta=1.5):
        super().__init__(model=model)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('UCB beta must be finite and nonnegative')
        # Reuse the complete shape/finite validation for observed objectives.
        hypervolume_2d(observed, reference)
        self.register_buffer('observed', observed.detach().clone())
        self.register_buffer('reference', reference.detach().clone())
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X):
        if not X.shape[-2] or not X.shape[-1] or not torch.isfinite(X).all():
            raise ValueError('Acquisition candidates must be nonempty and finite')
        posterior = self.model.posterior(X)
        mean, variance = posterior.mean, posterior.variance
        if mean.shape != (*X.shape[:-1], 2) or variance.shape != mean.shape:
            raise ValueError('Joint posterior moments must have shape (..., q, 2)')
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('Posterior moments must be finite and variance nonnegative')
        positive = variance > 0
        std = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        std = torch.where(positive, std, torch.zeros_like(std))
        prediction = mean + math.sqrt(self.beta) * std
        observed, reference = self.observed.to(prediction), self.reference.to(prediction)
        base = hypervolume_2d(observed, reference)
        scores = torch.stack([
            hypervolume_2d(torch.cat((observed, points)), reference) - base
            for points in prediction.reshape(-1, X.shape[-2], 2)
        ]).reshape(X.shape[:-2])
        if not torch.isfinite(scores).all():
            raise ValueError('Joint hypervolume scores must be finite')
        # A flat, zero-gain region still has a valid zero input gradient.
        return scores + X.sum(dim=(-2, -1)) * 0.


def select_ucb_candidates(model, candidates, q, *, beta=1.5, observed=None,
                          reference=None):
    """Rank optimistic hypervolume gain using observed standardized outcomes.

    Tied or zero hypervolume gains retain ``torch.topk`` ordering.
    """
    if candidates.ndim != 2 or not len(candidates) or not candidates.shape[1]:
        raise ValueError('Candidates must be a nonempty candidate x feature tensor')
    if type(q) is not int or not 1 <= q <= min(20, len(candidates)):
        raise ValueError('q must be between one and min(20, candidate count)')
    if not math.isfinite(beta) or beta < 0 or not torch.isfinite(candidates).all():
        raise ValueError('Acquisition beta and candidates must be finite; beta must be nonnegative')
    with torch.no_grad():
        posterior = model.posterior(candidates)
        mean, variance = posterior.mean, posterior.variance
        if mean.ndim != 2 or mean.shape[0] != len(candidates) or variance.shape != mean.shape:
            raise ValueError('Posterior moments must be candidate x objective tensors')
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('Posterior moments must be finite and variance nonnegative')
        prediction = mean + math.sqrt(beta) * variance.sqrt()
        if observed is None or reference is None or prediction.shape[1] != 2:
            raise ValueError('Joint UCB needs observed objectives and a two-objective reference')
        base = hypervolume_2d(observed, reference)
        scores = torch.stack([
            hypervolume_2d(torch.cat((observed, point.unsqueeze(0))), reference) - base
            for point in prediction
        ])
        if scores.shape != (len(candidates),) or not torch.isfinite(scores).all():
            raise ValueError('Candidate scores must be finite and have candidate-count shape')
        selected = torch.topk(scores, q).indices
    return selected, {'mode': 'ucb_hv', 'policy': 'topk',
                      'score_min': float(scores.min()), 'score_max': float(scores.max())}
