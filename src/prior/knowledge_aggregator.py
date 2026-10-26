"""
Knowledge Extractor and Aggregator for RuO₂/Er OER Literature Analysis

This module contains classes for extracting knowledge from literature using expert-specific
prompts and aggregating the results into quantitative prior parameters.
"""

import json
import math
import traceback
import time
from typing import Dict, List, Optional
from src.prior.expert_prompts import ExpertPrompts, EXPERT_TO_PRIOR_MAPPING
from src.prior.schema import TERM_TYPES, sanitize_prior_payload


class KnowledgeExtractor:
    """Knowledge extractor using expert-specific prompts"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.expert_prompts = ExpertPrompts()

    VALID_TERM_TYPES = TERM_TYPES

    @staticmethod
    def _valid_dimension(value) -> Optional[int]:
        """Accept integral indices without truncating malformed model values."""
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                return None
            return int(number) if 0 <= number <= 3 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _format_prompt(self, prompt: str, content: str) -> str:
        """Safely inject content into prompt by escaping all braces except the {content} placeholder.
        This avoids KeyError from str.format when prompts contain JSON examples with braces.
        """
        escaped = prompt.replace('{', '{{').replace('}', '}}')
        # Only restore the *trailing* {content} placeholder — the prompt template
        # ends with "\nLITERATURE CONTENT:\n{content}\n\n..." so the first
        # occurrence after the format spec is the one we want.
        # Using count=1 avoids accidentally restoring braces that were originally
        # doubled in the prompt text.
        escaped = escaped.replace('{{content}}', '{content}', 1)
        return escaped.format(content=content)

    @classmethod
    def _validate_term(cls, term: Dict) -> Optional[Dict]:
        """Validate and sanitize a single prior term from LLM output.

        Returns the sanitized term dict, or None if the term is invalid.
        """
        if not isinstance(term, dict):
            return None
        t = term.get("type", "")
        if t not in cls.VALID_TERM_TYPES:
            return None
        try:
            amp = float(term["amplitude"])
            if not (abs(amp) <= 10.0 and amp == amp):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        # Two-dim terms require 'dims'
        if t == "product_gauss":
            dims = term.get("dims")
            if not isinstance(dims, list) or len(dims) != 2:
                return None
            d0, d1 = (cls._valid_dimension(value) for value in dims)
            if d0 is None or d1 is None:
                return None
            try:
                center0 = float(term["center0"])
                width0 = float(term["width0"])
                center1 = float(term["center1"])
                width1 = float(term["width1"])
                if not all(math.isfinite(value) for value in (center0, width0, center1, width1)):
                    return None
                if width0 <= 0 or width1 <= 0:
                    return None
            except (KeyError, TypeError, ValueError):
                return None
            return {
                "dims": [d0, d1],
                "type": t,
                "amplitude": amp,
                "center0": center0,
                "width0": width0,
                "center1": center1,
                "width1": width1,
            }
        # Single-dim terms require 'dim'
        if t in ("gauss", "sigmoid_rise", "sigmoid_fall"):
            dim = cls._valid_dimension(term.get("dim"))
            if dim is None:
                return None
            try:
                center = float(term["center"])
                width = float(term["width"])
                if not math.isfinite(center) or not math.isfinite(width) or width <= 0:
                    return None
            except (KeyError, TypeError, ValueError):
                return None
            return {
                "dim": dim,
                "type": t,
                "amplitude": amp,
                "center": center,
                "width": width,
            }
        return None

    @classmethod
    def _validate_knowledge(cls, knowledge: Dict) -> Dict:
        """Validate and sanitize a knowledge dict from LLM output.

        Ensures required fields exist and terms have valid structure.
        """
        if not isinstance(knowledge, dict):
            return {"confidence": 0.1, "eta_terms": [], "t80_terms": [], "parsing_error": True}

        result = {"confidence": 0.1, "eta_terms": [], "t80_terms": []}

        # Validate confidence
        try:
            conf = float(knowledge.get("confidence", 0.1))
            if math.isfinite(conf):
                result["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            pass

        # Validate term lists
        for field in ("eta_terms", "t80_terms"):
            raw_terms = knowledge.get(field, [])
            if not isinstance(raw_terms, list):
                continue
            validated = []
            for term in raw_terms[:6]:  # cap at 6 terms per field
                v = cls._validate_term(term)
                if v is not None:
                    validated.append(v)
            result[field] = validated

        if "parsing_error" in knowledge:
            result["parsing_error"] = knowledge["parsing_error"]

        return result

    def extract_knowledge_from_paper(self, paper: Dict) -> Dict:
        """Extract knowledge from one complete paper using all expert prompts."""

        # Prepare paper content
        content = f"""
            Title: {paper.get('title', '')}
            Abstract: {paper.get('abstract', '')}
            Content: {paper.get('content', '')}
            """

        extracted_knowledge = {}

        print(f"  🔍 Analyzing paper: {paper.get('title', 'Unknown')[:60]}...")

        # Extract knowledge using each expert
        for expert_type in self.expert_prompts.get_all_expert_types():
            try:
                prior_key = EXPERT_TO_PRIOR_MAPPING[expert_type]
                print(f"    - Consulting {expert_type.replace('_', ' ').title()} for {prior_key}...")

                prompt = self.expert_prompts.get_prompt(expert_type)
                formatted_prompt = self._format_prompt(prompt, content)

                response = self.llm_client.call_llm(formatted_prompt)

                # Try to parse JSON response
                try:
                    raw = json.loads(response.strip())
                    knowledge = self._validate_knowledge(raw)
                    extracted_knowledge[expert_type] = knowledge
                except json.JSONDecodeError:
                    # If JSON parsing fails, store raw response
                    extracted_knowledge[expert_type] = {
                        "raw_response": response,
                        "confidence_score": 0.1,
                        "parsing_error": True
                    }
                    print(f"      ⚠️ JSON parsing failed, storing raw response")

                # Add delay to avoid API rate limits
                time.sleep(1)

            except Exception as e:
                print(f"      ❌ Extraction failed: {traceback.format_exc()}")
                extracted_knowledge[expert_type] = {
                    "error": str(e),
                    "confidence_score": 0.0
                }

        extracted_knowledge['paper_id'] = paper.get('paper_id')
        extracted_knowledge['paper_title'] = paper.get('title')
        extracted_knowledge['source_path'] = paper.get('file_path')
        extracted_knowledge['doi'] = paper.get('doi', '')

        return extracted_knowledge


class KnowledgeAggregator:
    """Knowledge aggregator for converting expert insights to prior parameters"""

    def __init__(self, extracted_knowledge_list: List[Dict]):
        self.knowledge_list = extracted_knowledge_list
        self.aggregated_priors = {}

    def aggregate_knowledge(self) -> Dict[str, Dict[str, float]]:
        """Concatenate source terms, weighting amplitudes by paper confidence."""
        print("Aggregating extracted knowledge...")

        for expert_type in EXPERT_TO_PRIOR_MAPPING:
            prior_key = EXPERT_TO_PRIOR_MAPPING[expert_type]
            print(f"  Aggregating {expert_type.replace('_', ' ').title()} insights for {prior_key}...")

            # Collect valid knowledge from this expert
            expert_knowledge = []
            for paper_knowledge in self.knowledge_list:
                if (expert_type in paper_knowledge and
                    'error' not in paper_knowledge[expert_type] and
                    'parsing_error' not in paper_knowledge[expert_type]):
                    expert_knowledge.append(paper_knowledge[expert_type])

            print(f"    - Valid knowledge entries: {len(expert_knowledge)}")

            if expert_knowledge:
                self.aggregated_priors[prior_key] = self._aggregate_concat(expert_knowledge, prior_key)
            else:
                print(f"    No valid knowledge for {prior_key}, using defaults")
                self.aggregated_priors[prior_key] = self._get_default_params(prior_key)

        self.aggregated_priors = {
            key: sanitize_prior_payload(payload)
            for key, payload in self.aggregated_priors.items()
        }

        return self.aggregated_priors

    def _aggregate_concat(self, knowledge_list: List[Dict], prior_key: str) -> Dict:
        """Concatenation aggregator: collect all terms from all valid papers.

        Each term's amplitude is multiplied by the paper's confidence score so that
        high-confidence papers contribute more strongly. The overall confidence of
        the result is the mean of all paper confidences.
        """
        valid = []
        for k in knowledge_list:
            if (isinstance(k, dict) and
                    all(f in k for f in ('confidence', 'eta_terms', 't80_terms')) and
                    isinstance(k['eta_terms'], list) and
                    isinstance(k['t80_terms'], list)):
                try:
                    conf = float(k['confidence'])
                    if 0.0 <= conf <= 1.0:
                        valid.append(k)
                except (TypeError, ValueError):
                    pass

        if not valid:
            return self._get_default_params(prior_key)

        avg_conf = round(sum(float(k['confidence']) for k in valid) / len(valid), 4)
        result: Dict = {'confidence': avg_conf}

        for field in ('eta_terms', 't80_terms'):
            all_terms = []
            for k in valid:
                conf = float(k['confidence'])
                for term in k.get(field, []):
                    t = dict(term)
                    # Scale amplitude by this paper's confidence
                    if 'amplitude' in t:
                        t['amplitude'] = round(float(t['amplitude']) * conf, 4)
                    all_terms.append(t)
            result[field] = all_terms

        return result

    def _get_default_params(self, prior_key: str) -> Dict:
        """Represent missing literature evidence without inventing prior terms."""
        if prior_key not in EXPERT_TO_PRIOR_MAPPING.values():
            raise ValueError(f"Unknown prior key: {prior_key}")
        return {"confidence": 0.1, "eta_terms": [], "t80_terms": []}
