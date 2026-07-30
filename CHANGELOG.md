# Changelog

## 2.0 — revised manuscript model

This is the public research-software version derived from the major-revision
process for the associated manuscript.

- Implemented the deterministic zero-order Sugeno engine with three
  input-specific Low/Medium/High membership partitions.
- Versioned and exported all 27 monotonic fuzzy rules, singleton consequents,
  rule confidences, design weights, epsilon, and complete inference traces.
- Unified output labels as Low `[0,1)`, Medium `[1,2)`, and High `[2,3]`.
- Separated structural applicability from numerical relevance.
- Evaluated the complete 30-KPI catalog across Product Transfer, Technology
  Transfer, and Facility Design while identifying each KPI's primary system.
- Added membership-threshold sensitivity, seeded Monte Carlo perturbation,
  tie-aware priority-set retention, component ablation, and 5S counterfactuals.
- Added internal convergent comparisons with TOPSIS, weighted sum, and
  PROMETHEE using the pre-inference antecedents.
- Added exact prompt, routed-model, response, grounding, rejection, and
  deterministic-fallback audit records for the optional LLM renderer.
- Frozen objective, industry, and role with each analyzed run while excluding
  them explicitly from fuzzy scoring and numerical validation.
- Clarified that KPI classes represent strategic priority rather than measured
  operational performance.
- Added reproducible command-line execution, frozen expected results, complete
  configuration export, and automated regression tests.

## 1.0 — submitted manuscript prototype

- Initial hybrid fuzzy–LLM decision-support prototype.
- Deterministic scoring separated from optional language generation.
- Initial lifecycle, Industry 5.0 5S, process, KPI, and resilience matrices.
