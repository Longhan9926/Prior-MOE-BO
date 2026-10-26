"""Finite-pool BO replay with observed-data-only scaling."""
from __future__ import annotations

import numpy as np
import torch
from src.bo.acquisition import hypervolume, select_ucb_candidates
from src.bo.objective_transforms import OBJECTIVE_SPEC, TargetScaler


def _initial_state(seed, n_all, n_init):
    if not 1 <= n_init < n_all:
        raise ValueError('Initial budget must be positive and smaller than candidate pool.')
    rng = np.random.default_rng(seed)
    observed = rng.choice(n_all, n_init, replace=False).tolist()
    return rng, observed, [i for i in range(n_all) if i not in observed]


def run_joint_replay(seed, features, objectives, reference, fit_model,
                     n_init, n_rounds, q, n_all, ucb_beta):
    """Joint UCB replay; the callback fits a model on standardized observations.

    Both surrogate strategies share initialization, candidate ordering and target
    scaling. The fitter retains each strategy's original model construction order.
    """
    rng = np.random.default_rng(seed)
    candidates = list(range(n_all))
    observed = rng.choice(candidates, size=n_init, replace=False).tolist()
    available = [i for i in candidates if i not in observed]
    curve = [hypervolume(objectives[observed], reference)]

    for _ in range(n_rounds):
        if not available:
            break
        x = torch.tensor(features[observed], dtype=torch.double)
        scaler = TargetScaler.fit(objectives[observed])
        y = torch.tensor(scaler.transform(objectives[observed]), dtype=torch.double)
        model = fit_model(x, y, scaler)
        model.eval()
        transformed_reference = torch.tensor(
            scaler.transform(np.asarray(reference)), dtype=torch.double)
        available_x = torch.tensor(features[available], dtype=torch.double)
        local, _ = select_ucb_candidates(
            model, available_x, min(q, len(available)), beta=ucb_beta,
            observed=y, reference=transformed_reference)
        selected = [available[i] for i in local]
        observed = observed + selected
        available = [i for i in available if i not in selected]
        curve = curve + [hypervolume(objectives[observed], reference)]
    return curve, observed


def run_random_replay(seed, objectives, n_init, n_rounds, q):
    """Random sampling shares exactly the BO initialization for the same seed."""
    if q < 1 or n_rounds < 0:
        raise ValueError('q must be positive and rounds non-negative.')
    rng, observed, available = _initial_state(seed, len(objectives), n_init)
    curve, budget = [hypervolume(objectives[observed], OBJECTIVE_SPEC.reference_objectives)], [n_init]
    for _ in range(n_rounds):
        if not available:
            break
        selected = rng.choice(available, min(q, len(available)), replace=False).tolist()
        observed = observed + selected
        available = [i for i in available if i not in selected]
        curve = curve + [hypervolume(objectives[observed], OBJECTIVE_SPEC.reference_objectives)]
        budget = budget + [len(observed)]
    return {'trajectory': np.asarray(curve), 'physical': np.asarray(curve),
            'utility': np.asarray(curve), 'budget': np.asarray(budget),
            'observed_indices': np.asarray(observed)}
