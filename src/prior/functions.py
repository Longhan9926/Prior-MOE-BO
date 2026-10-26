"""Prior schema, loading, and evaluation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import torch

from src.prior.schema import OPTIONAL_PRIOR_FIELDS, PRIOR_FIELDS, TERM_FIELDS, TERM_TYPES

LEGACY_FIELD = "applicability" + "_terms"

def _require_number(term: Dict[str, Any], field: str) -> float:
    if isinstance(term.get(field), bool):
        raise ValueError(f"Prior term '{field}' must be numeric, not boolean: {term}")
    try:
        value = float(term[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Prior term requires numeric '{field}': {term}") from exc
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"Prior term '{field}' must be finite: {term}")
    return value


def validate_priors_schema(priors: Any) -> Dict[str, Dict[str, Any]]:
    """Validate priors, including optional response-specific expert confidence."""
    if not isinstance(priors, dict) or not priors:
        raise ValueError("Priors must be a non-empty JSON object keyed by expert ID.")

    for key, params in priors.items():
        if not isinstance(params, dict):
            raise ValueError(f"Prior {key} must be a JSON object.")
        if LEGACY_FIELD in params:
            raise ValueError(
                "Legacy prior schema is unsupported; supply the frozen publication priors."
            )
        fields = set(params)
        optional_fields = OPTIONAL_PRIOR_FIELDS
        if not PRIOR_FIELDS <= fields or fields - PRIOR_FIELDS - optional_fields:
            raise ValueError(
                f"Prior {key} fields must be {sorted(PRIOR_FIELDS)}; got {sorted(fields)}."
            )
        confidence = _require_number(params, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Prior {key} confidence must lie in [0, 1].")
        for field in optional_fields.intersection(fields):
            if not 0.0 <= _require_number(params, field) <= 1.0:
                raise ValueError(f"Prior {key} {field} must lie in [0, 1].")
        for field in TERM_FIELDS:
            terms = params[field]
            if not isinstance(terms, list):
                raise ValueError(f"Prior {key}.{field} must be a list.")
            for term in terms:
                _validate_term_schema(key, field, term)
    return cast(Dict[str, Dict[str, Any]], priors)


def _validate_term_schema(key: str, field: str, term: Any) -> None:
    if not isinstance(term, dict) or term.get("type") not in TERM_TYPES:
        raise ValueError(f"Unsupported term in {key}.{field}: {term}")
    _require_number(term, "amplitude")
    term_type = term["type"]
    if term_type == "product_gauss":
        dims = term.get("dims")
        if not isinstance(dims, list) or len(dims) != 2:
            raise ValueError(f"product_gauss requires two dims: {term}")
        if any(not isinstance(dim, int) or not 0 <= dim <= 3 for dim in dims):
            raise ValueError(f"Term dimensions must lie in [0, 3]: {term}")
        for name in ("center0", "width0", "center1", "width1"):
            _require_number(term, name)
        if float(term["width0"]) <= 0 or float(term["width1"]) <= 0:
            raise ValueError(f"Term widths must be positive: {term}")
        return

    dim = term.get("dim")
    if not isinstance(dim, int) or not 0 <= dim <= 3:
        raise ValueError(f"Term dimension must lie in [0, 3]: {term}")
    _require_number(term, "center")
    if _require_number(term, "width") <= 0:
        raise ValueError(f"Term width must be positive: {term}")


def load_priors(priors_path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Read prior coefficients from JSON."""
    path = Path(priors_path) if priors_path is not None else Path(__file__).with_name("priors.json")
    with path.open("r", encoding="utf-8") as handle:
        return validate_priors_schema(json.load(handle))


def _ensure_tensor(x: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.double)
    return torch.tensor(list(x), dtype=torch.double)


def _eval_single_term(x: torch.Tensor, term: Dict[str, Any]) -> torch.Tensor:
    term_type = term["type"]
    amplitude = float(term["amplitude"])
    if term_type in ("gauss", "sigmoid_rise", "sigmoid_fall"):
        value = x[..., int(term["dim"])]
        center = float(term["center"])
        width = float(term["width"])
        if term_type == "gauss":
            return amplitude * torch.exp(-0.5 * ((value - center) / width) ** 2)
        if term_type == "sigmoid_rise":
            return amplitude * torch.sigmoid((value - center) / width)
        return amplitude * torch.sigmoid((center - value) / width)

    if term_type == "product_gauss":
        dim0, dim1 = (int(dim) for dim in term["dims"])
        z0 = (x[..., dim0] - float(term["center0"])) / float(term["width0"])
        z1 = (x[..., dim1] - float(term["center1"])) / float(term["width1"])
        return amplitude * torch.exp(-0.5 * z0**2) * torch.exp(-0.5 * z1**2)
    raise ValueError(f"Unsupported prior term type: {term_type}")


def _eval_components(x: torch.Tensor, terms: List[Dict[str, Any]]) -> torch.Tensor:
    result = torch.zeros(x.shape[:-1], dtype=torch.double, device=x.device)
    for term in terms:
        result = result + _eval_single_term(x, term)
    return result


def prior_mean_eta_Pk(
    x: torch.Tensor,
    priors: Dict[str, Any],
    key: Optional[str] = None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Evaluate activity prior means; larger values indicate lower overpotential."""
    x = _ensure_tensor(x)
    if key is None:
        return {name: cast(torch.Tensor, prior_mean_eta_Pk(x, priors, name)) for name in priors}
    return _eval_components(x, priors[key]["eta_terms"])


def prior_mean_t80_Pk(
    x: torch.Tensor,
    priors: Dict[str, Any],
    key: Optional[str] = None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Evaluate durability prior means; larger values indicate longer lifetime."""
    x = _ensure_tensor(x)
    if key is None:
        return {name: cast(torch.Tensor, prior_mean_t80_Pk(x, priors, name)) for name in priors}
    return _eval_components(x, priors[key]["t80_terms"])


def validate_prior_strength(strength: float) -> float:
    """Require a finite positive scalar so every case retains its prior."""
    value = float(strength)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("prior-strength must be finite and strictly positive")
    return value


def scale_prior_amplitudes(priors: dict, strength: float = 1.0) -> dict:
    """Return a deep copy with both target amplitudes multiplied, without rounding."""
    strength = validate_prior_strength(strength)
    def scaled_term(term, context):
        if "amplitude" not in term:
            return copy.deepcopy(term)
        amplitude = float(term["amplitude"]) * strength
        if not math.isfinite(amplitude):
            raise ValueError(f"Nonfinite scaled amplitude in {context}")
        return {**copy.deepcopy(term), "amplitude": amplitude}

    return {
        key: {
            field: [scaled_term(term, f"{key}/{field}") for term in value]
            if field in ("eta_terms", "t80_terms") else copy.deepcopy(value)
            for field, value in expert.items()
        } for key, expert in priors.items()
    }


def _effective_prior_bytes(priors: dict) -> bytes:
    return (json.dumps(priors, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def effective_prior_metadata(priors: dict, strength: float = 1.0) -> dict:
    """Describe effective amplitudes with the hash of their exact artifact bytes."""
    return {
        "prior_strength": validate_prior_strength(strength),
        "prior_strength_transformation": "multiply_eta_and_t80_amplitudes_once_without_renormalization_or_rounding",
        "effective_priors_file": "effective_priors.json",
        "effective_priors_sha256": hashlib.sha256(_effective_prior_bytes(priors)).hexdigest(),
    }


def save_effective_priors(priors: dict, output_dir: str | Path, strength: float = 1.0) -> dict:
    """Write the already scaled prior; this function never scales it again."""
    metadata = effective_prior_metadata(priors, strength)
    (Path(output_dir) / metadata["effective_priors_file"]).write_bytes(_effective_prior_bytes(priors))
    return metadata
