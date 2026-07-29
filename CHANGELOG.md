# Changelog

## 2.1-provisional — 2026-07-28

- Replaced the former 13-item, one-configuration-only KPI assignment with the
  complete 30-KPI catalog reported in the manuscript.
- Every current KPI is now evaluated across Product Transfer, Technology
  Transfer, and Facility Design using gradual relevance values `1–3`; the
  current KPI matrix contains no structural `N/A`.
- Reserved `N/A` only for future exclusions supported by an explicit domain
  rationale, while retaining numeric `0` for applicable but minimally relevant
  items.
- Added `KPI_PRIMARY_SYSTEM` to distinguish a KPI's manuscript family from its
  cross-configuration applicability.
- Added complete 5S and lifecycle associations for all 30 KPIs.
- Replaced the coarsely rounded five-level rule outputs with 27 explicit
  numeric singleton consequents (19 unique values). This preserves the
  declared baseline/5S/lifecycle weighting and makes lifecycle-stage changes
  observable in the deterministic score.
- Aligned core-process and resilience-driver baselines with the corresponding
  manuscript tables.
- Aligned scored benchmark names with the KPI catalog and marked benchmark-only
  contextual indicators explicitly.
- Added `KPI_MAPPING.md`, exported the mapping protocol, and regenerated all
  frozen replication artifacts.

## 2.0-provisional — 2026-07-28

- Replaced the shared membership partition with three input-specific
  Low/Medium/High partitions.
- Expanded Sugeno singleton outputs from three to five levels spanning
  `[0, 3]`.
- Constructed and exported all 27 monotonic rules using declared
  baseline/5S/lifecycle design weights of `0.50/0.30/0.20`.
- Added per-rule confidence coefficients, initialized to `1.0`.
- Separated structural `N/A` from a numeric baseline of zero and exposed the
  applicability gate in every trace.
- Added raw and confidence-adjusted firing strengths to rule traces.
- Added membership-threshold sensitivity at `-delta` and `+delta`.
- Expanded automated checks for partition coverage, full rule coverage,
  range, monotonicity, applicability, and repeatability.
- Exported the complete decision matrices and association mappings in
  `config/fuzzy_model.json`.
- Regenerated the frozen replication result; earlier scores and reported rank
  correlations are not valid for this version.
- Removed duplicate scoring and decision-matrix definitions from `app.py`;
  `fuzzy_engine.py` and `decision_model.py` are now the only scientific source
  of truth.
- Fixed What-If scenario recomputation so structural `N/A` KPI entries are
  gated instead of being converted with `float(None)`. What-If ablations now
  use the same authoritative `score_all()` path and exclude non-applicable
  KPIs from correlation and visualization.
- Added API-independent grounding validator `2.0`: payload-level checks for
  unsupported numbers/rules, item–score and item–rule mismatches, ranking
  inversions, reasoning leakage, and unsupported normative, causal, risk, or
  outcome claims.
- Tightened all LLM prompts to factual trace rendering and removed requests
  for prescriptive actions, maturity judgments, advantages, vulnerabilities,
  or claims that drivers create stability/flexibility.
- Added per-section/per-attempt routed-model audit, explicit fallback-section
  notices, and grounding rejection reasons to the downloadable run JSON.
- Deterministic explanations now include all items tied at the third-position
  cutoff and explicitly identify ties.
