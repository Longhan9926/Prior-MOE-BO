"""Select frozen literature contributions separately for each expert and target.

No BO outcomes, coefficient scaling, or geometry sanitization enter selection.
An emptied nonempty target retains its highest-confidence source; this is prior
retention, not an acquisition or uncertainty fallback.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from src.prior.schema import PRIOR_KEYS, TERM_TYPES


FIELDS = ("eta_terms", "t80_terms")


def _read_entries(contributions: dict, priors: dict) -> dict:
    """Validate source metadata and group whole entries by expert and target."""
    if not isinstance(contributions, dict) or not isinstance(priors, dict):
        raise ValueError("Contributions and original priors must be dictionaries")
    if not priors:
        raise ValueError("Original priors cannot be empty")
    groups = {prior: {field: [] for field in FIELDS} for prior in priors}
    for source_index, (source_id, paper) in enumerate(contributions.items()):
        if not isinstance(paper, dict):
            raise ValueError(f"Invalid source payload: {source_id}")
        for prior in priors:
            payload = paper.get(prior, {})
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid expert payload: {source_id}/{prior}")
            entry_confidences = set()
            for field in FIELDS:
                records = payload.get(field, [])
                if not isinstance(records, list):
                    raise ValueError(f"Expected term list: {source_id}/{prior}/{field}")
                if not records:
                    continue
                try:
                    confidences = [float(record["paper_confidence"]) for record in records]
                    terms = [record["priors_json_contribution_term"] for record in records]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid source confidence or contribution: {source_id}") from exc
                if any(not math.isfinite(c) or not 0 <= c <= 1 for c in confidences):
                    raise ValueError(f"Invalid source confidence: {source_id}/{prior}")
                if any(not isinstance(term, dict) for term in terms):
                    raise ValueError(f"Invalid contribution term: {source_id}/{prior}")
                entry_confidences.update(confidences)
                groups[prior][field].append({
                    "source_id": source_id, "source_index": source_index,
                    "confidence": confidences[0], "terms": deepcopy(terms),
                })
            if len(entry_confidences) > 1:
                raise ValueError(f"Terms of one source/expert must share confidence: {source_id}/{prior}")
    return groups


def _select_by_threshold(entries: list, threshold: float) -> tuple[list, str | None]:
    selected = [entry for entry in entries if entry["confidence"] >= threshold]
    if entries and not selected:
        best = min(entries, key=lambda entry: (-entry["confidence"], entry["source_index"]))
        return [best], best["source_id"]
    return selected, None


def _number(value, name: str, *, confidence: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value, not a boolean")
    if confidence and not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)


def _validate_term(term, name: str) -> None:
    if not isinstance(term, dict) or term.get("type") not in TERM_TYPES:
        raise ValueError(f"Unsupported contribution term: {name}")
    _number(term.get("amplitude"), f"{name}.amplitude")
    product = term["type"] == "product_gauss"
    dims = term.get("dims") if product else [term.get("dim")]
    if not isinstance(dims, list) or len(dims) != (2 if product else 1):
        raise ValueError(f"Invalid dimensions: {name}")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or not 0 <= dim <= 3 for dim in dims):
        raise ValueError(f"Dimensions must be integers in [0, 3]: {name}")
    for suffix in (("0", "1") if product else ("",)):
        _number(term.get(f"center{suffix}"), f"{name}.center{suffix}")
        if _number(term.get(f"width{suffix}"), f"{name}.width{suffix}") <= 0:
            raise ValueError(f"Widths must be positive: {name}")


def _validated_groups(contributions: dict, original: dict) -> dict:
    if not isinstance(original, dict) or set(original) != set(PRIOR_KEYS):
        raise ValueError("Original priors must contain exactly the five experts P1-P5")
    for prior, payload in original.items():
        if not isinstance(payload, dict) or set(payload) != {"confidence", *FIELDS}:
            raise ValueError(f"Invalid original expert payload: {prior}")
        _number(payload["confidence"], f"{prior}.confidence", confidence=True)
        for field in FIELDS:
            if not isinstance(payload[field], list):
                raise ValueError(f"Expected original term list: {prior}/{field}")
            for term in payload[field]:
                _validate_term(term, f"{prior}/{field}")
    groups = _read_entries(contributions, original)
    # _read_entries converts source confidence to float, so inspect the original
    # values as well to reject booleans and numeric strings at this boundary.
    for source, paper in contributions.items():
        if not isinstance(source, str):
            raise ValueError("Literature source IDs must be strings")
        for prior in original:
            for field in FIELDS:
                for record in paper.get(prior, {}).get(field, []):
                    _number(record["paper_confidence"], f"{source}/{prior}.confidence", confidence=True)
                    _validate_term(record["priors_json_contribution_term"], f"{source}/{prior}/{field}")
    reconstructed = {
        prior: {"confidence": original[prior]["confidence"],
                **{field: [deepcopy(term) for entry in targets[field] for term in entry["terms"]]
                   for field in FIELDS}}
        for prior, targets in groups.items()
    }
    if reconstructed != original:
        raise ValueError("No-filter reconstruction does not exactly match original priors")
    return groups


def _selected_target(entries: list, threshold: float, original_confidence: float) -> tuple:
    selected, fallback_source = _select_by_threshold(entries, threshold)
    terms = [deepcopy(term) for entry in selected for term in entry["terms"]]
    confidences = [entry["confidence"] for entry in selected]
    confidence = sum(confidences) / len(confidences) if confidences else original_confidence
    amplitude_l1 = sum(abs(float(term["amplitude"])) for term in terms)
    audit = {
        "threshold": threshold, "nonempty_entry_count": len(entries),
        "source_term_count": sum(len(entry["terms"]) for entry in entries),
        "kept_entry_count": len(selected), "kept_term_count": len(terms),
        "kept_source_ids": [entry["source_id"] for entry in selected],
        "kept_source_confidences": confidences,
        "removed_source_ids": [entry["source_id"] for entry in entries if entry not in selected],
        "fallback_source_id": fallback_source,
        "retained_abs_amplitude": amplitude_l1, "output_abs_amplitude": amplitude_l1,
        "target_confidence": confidence,
        "confidence_default_reason": None if confidences else "no_nonempty_source_entries",
    }
    return terms, confidence, audit


def build_targetwise_priors(
    contributions: dict, original_priors: dict, *, threshold_eta: float = .90,
    threshold_t80: float = .70,
) -> tuple[dict, dict]:
    """Return new priors and an audit from frozen paper-entry evidence only.

    Adds ``confidence_eta`` and ``confidence_t80`` as means over retained source
    entries, giving each paper/expert entry one vote regardless of term count.
    """
    thresholds = {
        "eta_terms": _number(threshold_eta, "threshold_eta", confidence=True),
        "t80_terms": _number(threshold_t80, "threshold_t80", confidence=True),
    }
    groups = _validated_groups(contributions, original_priors)
    payload, experts = {}, {}
    for prior, targets in groups.items():
        original_confidence = original_priors[prior]["confidence"]
        selections = {field: _selected_target(entries, thresholds[field], original_confidence)
                      for field, entries in targets.items()}
        confidences = {"confidence_eta" if field == "eta_terms" else "confidence_t80": selection[1]
                       for field, selection in selections.items()}
        payload[prior] = {
            "confidence": original_confidence,
            **{field: selection[0] for field, selection in selections.items()},
            **confidences,
        }
        experts[prior] = {
            "fixed_aggregate_confidence": original_confidence,
            **{field: selection[2] for field, selection in selections.items()},
        }
    return payload, {
        "threshold_eta": thresholds["eta_terms"], "threshold_t80": thresholds["t80_terms"],
        "confidence_mode": "per_target", "no_filter_exact_match": True,
        "source_paper_count": len(contributions), "evidence": "frozen_literature_only",
        "selection_scope": "expert_target_nonempty_source_entries_before_concat",
        "threshold_inclusive": True, "amplitude_mode": "preserve_frozen",
        "confidence_aggregation": "mean_per_retained_source_entry_not_per_term",
        "tie_break": "confidence_descending_then_original_source_order",
        "concat_order": "original_source_order",
        "fallback": "retain_highest_confidence_nonempty_source_per_expert_target",
        "experts": experts,
    }
