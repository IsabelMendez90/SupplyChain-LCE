# Changelog

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
