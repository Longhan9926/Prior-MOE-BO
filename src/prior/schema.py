"""Canonical expert mapping and immutable prior-schema sanitization."""

from __future__ import annotations

from typing import Any, Mapping
import math

from src.bo.config import FEATURE_SPEC


EXPERT_TO_PRIOR = {
    "materials_science_expert": "P1",
    "electrochemistry_expert": "P2",
    "crystal_engineering_expert": "P3",
    "interface_science_expert": "P4",
    "transport_phenomena_expert": "P5",
}
PRIOR_TO_EXPERT = {prior: expert for expert, prior in EXPERT_TO_PRIOR.items()}
PRIOR_KEYS = tuple(PRIOR_TO_EXPERT)
PRIOR_FIELDS = frozenset({"confidence", "eta_terms", "t80_terms"})
OPTIONAL_PRIOR_FIELDS = frozenset({"confidence_eta", "confidence_t80"})
TERM_FIELDS = ("eta_terms", "t80_terms")
TERM_TYPES = frozenset({"gauss", "sigmoid_rise", "sigmoid_fall", "product_gauss"})

AMPLITUDE_LIMIT = 5.0
MIN_WIDTH_FRACTION = 1e-3


def _clamp(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Prior numeric values must be finite.")
    return max(lower, min(upper, value))


def sanitize_prior_term(term: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded copy of one prior term without changing its schema."""
    item = dict(term)
    numeric_fields = ("amplitude", "center", "width", "threshold", "ref",
                      "center0", "width0", "center1", "width1")
    for field in numeric_fields:
        if field in item and not math.isfinite(float(item[field])):
            raise ValueError(f"Prior term {field!r} must be finite.")
    if "amplitude" in item:
        item["amplitude"] = round(
            _clamp(float(item["amplitude"]), -AMPLITUDE_LIMIT, AMPLITUDE_LIMIT),
            4,
        )

    if "dim" in item:
        dim = int(_clamp(int(item["dim"]), 0, len(FEATURE_SPEC.columns) - 1))
        item["dim"] = dim
        lower, upper = FEATURE_SPEC.lower[dim], FEATURE_SPEC.upper[dim]
        span = upper - lower
        for field in ("center", "threshold", "ref"):
            if field in item:
                item[field] = round(_clamp(float(item[field]), lower, upper), 4)
        if "width" in item:
            item["width"] = round(
                max(float(item["width"]), MIN_WIDTH_FRACTION * span),
                4,
            )

    if "dims" in item:
        max_dim = len(FEATURE_SPEC.columns) - 1
        dims = [int(_clamp(int(dim), 0, max_dim)) for dim in item["dims"]]
        if len(dims) != 2:
            raise ValueError(f"Two-dimensional prior term requires exactly two dims: {term}")
        item["dims"] = dims
        for index, dim in enumerate(dims):
            lower, upper = FEATURE_SPEC.lower[dim], FEATURE_SPEC.upper[dim]
            center_field = f"center{index}"
            width_field = f"width{index}"
            if center_field in item:
                item[center_field] = round(
                    _clamp(float(item[center_field]), lower, upper),
                    4,
                )
            if width_field in item:
                item[width_field] = round(
                    max(
                        float(item[width_field]),
                        MIN_WIDTH_FRACTION * (upper - lower),
                    ),
                    4,
                )
        if "width" in item:
            item["width"] = round(max(float(item["width"]), MIN_WIDTH_FRACTION), 4)
    return item


def supported_expert_payload(payload: Any) -> dict[str, Any]:
    """Copy only fields supported by the active prior schema."""
    source = payload if isinstance(payload, dict) else {}
    return {
        "confidence": source.get("confidence", 0.0),
        **{field: source[field] for field in sorted(OPTIONAL_PRIOR_FIELDS) if field in source},
        **{
            field: list(source.get(field, [])) if isinstance(source.get(field), list) else []
            for field in TERM_FIELDS
        },
    }


def sanitize_prior_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded normalized priors, preserving optional target confidences."""
    supported = supported_expert_payload(payload)
    sanitized: dict[str, Any] = {
        "confidence": round(_clamp(float(supported["confidence"]), 0.0, 1.0), 4)
    }
    for field in sorted(OPTIONAL_PRIOR_FIELDS):
        if field in supported:
            try:
                value = float(supported[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Prior '{field}' must be finite and numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Prior '{field}' must be finite and numeric.")
            sanitized[field] = round(_clamp(value, 0.0, 1.0), 4)
    for field in TERM_FIELDS:
        terms = [
            sanitize_prior_term(term)
            for term in supported[field]
            if isinstance(term, dict)
        ]
        total_abs = sum(abs(float(term.get("amplitude", 0.0))) for term in terms)
        if total_abs > 0:
            terms = [
                {
                    **term,
                    "amplitude": round(float(term.get("amplitude", 0.0)) / total_abs, 4),
                }
                for term in terms
            ]
        sanitized[field] = terms
    return sanitized
