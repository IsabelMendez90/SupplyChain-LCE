# Changelog

## 2.3 methodological consistency update — 2026-07-30

- Unified the decision-model and fuzzy-rule-base identifiers at version 2.3.
- Added the application release, decision-model version, and fuzzy-rule-base
  version directly below the Streamlit title.
- Unified every reader-facing score label with the manuscript's declared
  Low `[0,1)`, Medium `[1,2)`, and High `[2,3]` bands.
- Removed the contradictory five-band metadata that classified a score such
  as `1.950` differently from the Streamlit matrices.
- Frozen objective, resolved industry, and resolved role at `Analyze` time and
  exported them with a separate context identifier.
- Explicitly constrained objective, industry, and role to language rendering;
  they do not enter deterministic fuzzy scoring or numerical validation.
- Added a standard three-case ablation suite: remove 5S, remove lifecycle, and
  remove both, with tie-aware Kendall and Jaccard reporting.
- Exported epsilon, alpha-cut status, qualitative-band policy, LLM token limit,
  and the prospective calibration status of the `0.50/0.30/0.20` rule-design
  weights.
- Increased the optional renderer token allowance to reduce empty responses
  from free reasoning models; the grounding validator and deterministic
  fallback remain unchanged.

## 2.2.2 OpenRouter transport hotfix — 2026-07-30

- Fixed a response-parsing defect in which `content=None` raised
  `AttributeError` before the routed model identifier was recorded.
- Added safe parsing for plain text and text-part lists and retained model,
  finish-reason, and reasoning-presence metadata for empty responses.
- Separated genuine API failures, empty/non-text responses, and grounding
  rejections in the interface and downloadable audit.
- Added sanitized API diagnostics with error type, HTTP status, provider code,
  and short message while redacting the configured key.
- Aligned the optional attribution header with OpenRouter's current
  `X-OpenRouter-Title` specification and removed the incorrect localhost
  referer.
- Disabled hidden SDK retries so the two declared application attempts are the
  complete auditable retry policy.
- Pinned Streamlit `1.60.0` and OpenAI Python `2.50.0` to eliminate deployment
  drift.

## 2.2.1 grounding hotfix — 2026-07-28

- Bound every required fuzzy score and dominant rule to the item-specific text
  segment in which it appears.
- Prevented 5S priorities, baseline values, lifecycle values, or a neighboring
  item's score from satisfying the fuzzy-score requirement.
- Added regression tests for auxiliary-number leakage, cross-item leakage,
  parenthetical score/rule syntax, and missing item-specific rules.
- Added the grounding-validator version to Streamlit session compatibility so
  cached narratives from an older validator are cleared.

## 2.2 — 2026-07-28

- Retained `model="openrouter/free"` as a documented multi-model robustness
  condition instead of forcing a single provider model.
- Added a unique call identifier, actual routed model, prompt hash, raw draft,
  validation outcome, rejection reason, and deterministic fallback to every
  LLM audit record.
- Updated grounding validator `2.1` to permit harmless stylistic numbers such
  as list numbering and `Industry 5.0`, while continuing to reject incorrect
  scores, rule identifiers, item associations, ranking inversions, and
  unsupported scientific claims.
- Added true component ablation: setting the 5S or lifecycle rule-design weight
  to zero and renormalizing the remaining weights removes that component from
  the rule base.
- Replaced the former range-across-configurations display with one-at-a-time 5S
  endpoint counterfactuals for the selected supply-chain configuration.
- Rebuilt TOPSIS, weighted-sum, and PROMETHEE-style comparisons from the
  original pre-inference antecedents rather than fuzzy outputs.
- Added tie-aware priority sets, exact p-value formatting, persistent
  robustness results, and complete downloadable validation and What-If JSON.
- Replaced the temporary model qualifier in application labels,
  documentation, and version identifiers while preserving an explicit
  external-validation boundary.

## 2.1.1 interface hotfix — 2026-07-28

- Added strict catalog and version validation for frozen Streamlit results and
  reloaded run JSON files.
- Automatically clears stale session results after a model redeploy and asks
  the user to run `Analyze` again.
- Canonical evidence now iterates only over the authoritative current catalog,
  so obsolete item names cannot reach strict scientific key lookup.
- Added the fuzzy-rule version to command-line replication output.

## 2.1 — 2026-07-28

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

## 2.0 — 2026-07-28

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
