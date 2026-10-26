"""
Expert-specific prompts for RuO₂ (doped) OER knowledge extraction

This module contains specialized prompts designed for different types of experts
to extract specific knowledge about OER mechanisms in RuO₂ with cation doping from literature.

Each expert focuses on a specific aspect:
- Materials Science Expert: P1 (Oxygen Vacancy Promotion)
- Electrochemistry Expert: P2 (Conductivity-ECSA Tradeoffs)
- Crystal Engineering Expert: P3 (Phase Separation Thresholds)
- Interface Science Expert: P4 (Interface Bonding/Stress Relief)
- Transport Phenomena Expert: P5 (Precursor Volume vs Mass Transport)
"""

from typing import Dict, List

from src.prior.schema import EXPERT_TO_PRIOR


EXPERT_TO_PRIOR_MAPPING = EXPERT_TO_PRIOR


# Shared format specification injected into every prompt
_FORMAT_SPEC = """
== DESIGN SPACE (all numeric values MUST lie within these bounds) ==
  dim 0  Er_at_pct   at%      [1, 20]
  dim 1  V_Ru_mL     mL   [1.0, 2.0]
  dim 2  T_C         degC     [200, 600]
  dim 3  t_h         h        [1, 10]

== AVAILABLE TERM TYPES (use only these four) ==
  {"dim": i, "type": "gauss",        "amplitude": A, "center": c, "width": w}
    -> bell-curve; A~1 for strong benefit, A<0 for penalty

  {"dim": i, "type": "sigmoid_rise", "amplitude": A, "center": c, "width": w}
    -> S-curve rising above c; A>0 = reward above c, A<0 = penalty above c

  {"dim": i, "type": "sigmoid_fall", "amplitude": A, "center": c, "width": w}
    -> S-curve falling above c; use A=1 for a soft upper-bound gate

  {"dims": [i, j], "type": "product_gauss", "amplitude": A,
   "center0": c0, "width0": w0, "center1": c1, "width1": w1}
    -> 2-D Gaussian; use when two dimensions jointly determine performance

== FIELD SEMANTICS ==
  eta_terms:           prior on eta500_V (supplied potential response in V; reference and iR correction are unverified). Positive = lower response (better OER).
  t80_terms:           prior on T80 (durability hours). Positive = longer lifetime.

== EXTRACTION RULES ==
- Use only the supplied literature content. Do not rely on outside knowledge.
- Do not force a prior term. If the text gives no usable evidence for this expert objective, return empty eta_terms and t80_terms.
- Evidence from analogous oxide OER systems is allowed when the source text supports a transferable mechanism, trend, threshold, or representative value.
- Do not treat a different material identity as an automatic rejection. Transfer limits should mainly lower confidence or broaden width, not force a wrong term.
- A best-performing or representative single sample can justify a broad gauss or product_gauss when the text presents it as optimized, best, representative, or central to the claimed performance.
- Use broader width for indirect, weakly localized, or single-point evidence. Use narrower width only when the paper clearly supports a tight optimum or threshold.
- Prefer omission over hallucination. Do not invent a numeric optimum, threshold, loading, or time window from vague background statements.
- Use lower confidence for indirect or weakly transferable evidence, and high confidence only for explicit, source-grounded optimization or threshold evidence.

== CONFIDENCE GUIDE ==
- 0.80-1.00: explicit optimization curve, explicit threshold, or repeated strong source support.
- 0.55-0.75: source-grounded best-performing or representative condition with usable numeric localization.
- 0.25-0.50: indirect but still transferable evidence, or numeric support with substantial ambiguity.
- 0.10: no usable evidence for this expert objective.

== STRICT OUTPUT FORMAT (JSON only, no other text) ==
{
  "confidence": <float 0.0-1.0, based on evidence strength>,
  "eta_terms":           [at most 2 terms],
  "t80_terms":           [at most 2 terms]
}

If the text provides no relevant information, output:
{"confidence": 0.1, "eta_terms": [], "t80_terms": []}
"""


class ExpertPrompts:
    """Collection of expert-specific prompts for knowledge extraction"""

    def __init__(self):
        self.prompts = self._initialize_expert_prompts()

    def _initialize_expert_prompts(self) -> Dict[str, str]:
        """Initialize all expert-specific prompts"""

        return {
            "materials_science_expert": self._get_materials_science_prompt(),
            "electrochemistry_expert": self._get_electrochemistry_prompt(),
            "crystal_engineering_expert": self._get_crystal_engineering_prompt(),
            "interface_science_expert": self._get_interface_science_prompt(),
            "transport_phenomena_expert": self._get_transport_phenomena_prompt()
        }

    def _get_materials_science_prompt(self) -> str:
        """Materials Science Expert - P1: Oxygen Vacancy Promotion Mechanism"""
        return (
            "You are a Materials Science Expert on oxide defect chemistry and cation doping.\n\n"
            "OBJECTIVE: From the text below, extract the OPTIMAL DOPANT CONTENT (at%) that maximizes"
            " OER activity via oxygen vacancy promotion in doped RuO2. Do not assume a specific dopant.\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Reported optimal dopant at% that minimizes overpotential or maximizes current density.\n"
            "2. Width/range of the beneficial dopant window.\n"
            "3. Whether the same composition also improves durability (T80).\n\n"
            "HOW TO MAP TO PRIOR TERMS:\n"
            "- Main signal: one gauss on dim 0 (Er_at_pct), center = optimal at%, width = half-range.\n"
            "- eta_terms amplitude ~1.0; t80_terms amplitude ~0.5 (durability signal is weaker).\n"
            "- Use a broader width when the source reports only one representative composition.\n"
            "- If the paper only suggests a harmful high-doping threshold rather than an optimum, omit the term instead of forcing a gauss.\n\n"
            "EXAMPLE (replace values with what the text supports):\n"
            "{\"confidence\": 0.6,\n"
            " \"eta_terms\": [{\"dim\": 0, \"type\": \"gauss\", \"amplitude\": 1.0, \"center\": 10.0, \"width\": 5.0}],\n"
            " \"t80_terms\": [{\"dim\": 0, \"type\": \"gauss\", \"amplitude\": 0.5, \"center\": 10.0, \"width\": 6.0}]\n"
            "}\n\n"
            + _FORMAT_SPEC
            + "\nLITERATURE CONTENT:\n{content}\n\nRespond with only the JSON object."
        )

    def _get_electrochemistry_prompt(self) -> str:
        """Electrochemistry Expert - P2: Conductivity-ECSA Tradeoffs"""
        return (
            "You are an Electrochemistry Expert on electrode kinetics and surface area effects.\n\n"
            "OBJECTIVE: From the text below, identify the OPTIMAL ANNEALING TEMPERATURE (degC) for"
            " doped RuO2 OER electrodes, balancing electrical conductivity and ECSA."
            " Also assess whether a specific time window within [1, 10] h matters for durability (T80).\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Optimal temperature for conductivity improvement vs surface area loss.\n"
            "2. Whether higher temperature (within [200, 600] degC) monotonically improves or\n"
            "   peaks the OER activity.\n"
            "3. Joint T-t optimum for durability if evidence exists.\n\n"
            "HOW TO MAP TO PRIOR TERMS:\n"
            "- Use gauss when the paper supports a finite optimum temperature window for activity.\n"
            "- Use sigmoid_rise only when the text clearly supports a monotonic gain above a temperature threshold.\n"
            "- Use product_gauss on (dim2, dim3) only when temperature and time are jointly localized by the source.\n"
            "- t80_terms: product_gauss on (dim2, dim3) centered at optimal T and t.\n"
            "- NOTE: t_h range is [1, 10] hours. Express source times in hours before mapping.\n"
            "- Do not invent a time term when the paper discusses temperature only.\n\n"
            "EXAMPLE:\n"
            "{\"confidence\": 0.55,\n"
            " \"eta_terms\": [{\"dim\": 2, \"type\": \"sigmoid_rise\", \"amplitude\": 0.3, \"center\": 380.0, \"width\": 40.0}],\n"
            " \"t80_terms\": [{\"dims\": [2, 3], \"type\": \"product_gauss\", \"amplitude\": 0.5,\n"
            "                  \"center0\": 380.0, \"width0\": 80.0, \"center1\": 6.0, \"width1\": 2.0}]\n"
            "}\n\n"
            + _FORMAT_SPEC
            + "\nLITERATURE CONTENT:\n{content}\n\nRespond with only the JSON object."
        )

    def _get_crystal_engineering_prompt(self) -> str:
        """Crystal Engineering Expert - P3: Phase Separation Thresholds"""
        return (
            "You are a Crystal Engineering Expert on solid-solution limits and phase stability in oxide systems.\n\n"
            "OBJECTIVE: From the text below, identify THRESHOLDS beyond which phase separation or secondary"
            " oxide phases form in doped RuO2, degrading OER performance.\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Dopant solubility limit (at%) above which secondary phases (e.g., MxOy) precipitate.\n"
            "2. Temperature threshold (degC) above which segregation accelerates.\n"
            "3. Whether these thresholds fall within the design space.\n\n"
            "HOW TO MAP TO PRIOR TERMS:\n"
            "- eta_terms and t80_terms: sigmoid_rise with NEGATIVE amplitude = penalty above threshold.\n"
            "  Use dim 0 for dopant threshold, dim 2 for temperature threshold.\n"
            "- Use the sigmoid width to encode uncertainty in the reported threshold.\n"
            "- Do not encode a beneficial gauss when the mechanism is phase separation, segregation, or impurity formation.\n\n"
            "EXAMPLE:\n"
            "{\"confidence\": 0.45,\n"
            " \"eta_terms\": [\n"
            "   {\"dim\": 0, \"type\": \"sigmoid_rise\", \"amplitude\": -0.2, \"center\": 15.0, \"width\": 0.5},\n"
            "   {\"dim\": 2, \"type\": \"sigmoid_rise\", \"amplitude\": -0.1, \"center\": 450.0, \"width\": 25.0}\n"
            " ],\n"
            " \"t80_terms\": [\n"
            "   {\"dim\": 0, \"type\": \"sigmoid_rise\", \"amplitude\": -0.15, \"center\": 15.0, \"width\": 0.5},\n"
            "   {\"dim\": 2, \"type\": \"sigmoid_rise\", \"amplitude\": -0.1, \"center\": 450.0, \"width\": 25.0}\n"
            " ]\n"
            "}\n\n"
            + _FORMAT_SPEC
            + "\nLITERATURE CONTENT:\n{content}\n\nRespond with only the JSON object."
        )

    def _get_interface_science_prompt(self) -> str:
        """Interface Science Expert - P4: Interface Bonding/Stress Relief"""
        return (
            "You are an Interface Science Expert on interfacial bonding, stress relief, and adhesion in electrodes.\n\n"
            "OBJECTIVE: From the text below, identify the OPTIMAL annealing temperature (degC) and time (h)"
            " that jointly maximize interface bonding quality and electrode durability in doped RuO2.\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Optimal bonding temperature within [200, 600] degC.\n"
            "2. Optimal bonding time within [1, 10] h. Express source times in hours before mapping.\n"
            "3. Width of the beneficial T and t windows.\n\n"
            "HOW TO MAP TO PRIOR TERMS:\n"
            "- Both eta_terms and t80_terms: one product_gauss on (dim2=T_C, dim3=t_h).\n"
            "  center0=T_opt, width0=T_half-range, center1=t_opt, width1=t_half-range.\n"
            "- Broaden width0/width1 if the source does not tightly localize the joint optimum.\n"
            "- If the paper supports durability but not activity, it is acceptable to leave eta_terms empty and keep only t80_terms.\n\n"
            "EXAMPLE:\n"
            "{\"confidence\": 0.55,\n"
            " \"eta_terms\": [{\"dims\": [2, 3], \"type\": \"product_gauss\", \"amplitude\": 0.7,\n"
            "                  \"center0\": 300.0, \"width0\": 80.0, \"center1\": 6.0, \"width1\": 2.5}],\n"
            " \"t80_terms\": [{\"dims\": [2, 3], \"type\": \"product_gauss\", \"amplitude\": 1.0,\n"
            "                  \"center0\": 300.0, \"width0\": 80.0, \"center1\": 6.0, \"width1\": 2.5}]\n"
            "}\n\n"
            + _FORMAT_SPEC
            + "\nLITERATURE CONTENT:\n{content}\n\nRespond with only the JSON object."
        )

    def _get_transport_phenomena_prompt(self) -> str:
        """Transport Phenomena Expert - P5: Precursor Volume vs Mass Transport"""
        return (
            "You are a Transport Phenomena Expert on mass transfer and diffusion in electrochemical systems.\n\n"
            "OBJECTIVE: From the text below, identify the OPTIMAL RU PRECURSOR SOLUTION VOLUME (mL) for doped RuO2"
            " OER electrodes, balancing active site density and mass transport.\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Explicit Ru precursor solution volume (mL) in or near [1.0, 2.0].\n"
            "2. Width of the beneficial precursor volume window.\n"
            "3. Whether the same precursor volume optimum applies to both activity (eta) and durability (T80).\n\n"
            "HOW TO MAP TO PRIOR TERMS:\n"
            "- eta_terms and t80_terms: one gauss on dim 1 (V_Ru_mL), center = optimal precursor volume in mL.\n"
            "  amplitude ~0.8-1.0 for eta, ~0.7 for t80.\n"
            "- Use a broader width when the source reports one representative precursor volume.\n"
            "- Do not convert catalyst mass loading (mg/cm2) into precursor volume without an explicit source-supported conversion. If only mass loading is reported or no usable precursor-volume optimum or threshold is supplied, omit the term.\n\n"
            "EXAMPLE:\n"
            "{\"confidence\": 0.5,\n"
            " \"eta_terms\": [{\"dim\": 1, \"type\": \"gauss\", \"amplitude\": 0.8, \"center\": 1.3, \"width\": 0.3}],\n"
            " \"t80_terms\": [{\"dim\": 1, \"type\": \"gauss\", \"amplitude\": 0.7, \"center\": 1.3, \"width\": 0.3}]\n"
            "}\n\n"
            + _FORMAT_SPEC
            + "\nLITERATURE CONTENT:\n{content}\n\nRespond with only the JSON object."
        )

    def get_prompt(self, expert_type: str) -> str:
        """Get prompt for specific expert type"""
        if expert_type not in self.prompts:
            raise ValueError(f"Unknown expert type: {expert_type}. Available types: {list(self.prompts.keys())}")
        return self.prompts[expert_type]

    def get_all_expert_types(self) -> List[str]:
        """Get list of all available expert types"""
        return list(self.prompts.keys())
