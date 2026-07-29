# KPI mapping and provisional baseline protocol

This document is the human-readable counterpart of `BASE_KPIS` in
`decision_model.py`. It aligns the implementation with the 30 representative
KPIs reported in the manuscript.

## Interpretation

- `0`: applicable, but minimally relevant;
- `1`: low or secondary relevance;
- `2`: important relevance;
- `3`: core or critical relevance; and
- `N/A`: structural exclusion supported by an explicit rationale.

No KPI in the current matrix is structurally excluded. All 30 are evaluated
under Product Transfer (PT), Technology Transfer (TT), and Facility Design
(FD). The primary configuration identifies the manuscript category in which
the KPI was introduced; it does not make the KPI inapplicable elsewhere.

| KPI | Primary configuration | PT | TT | FD |
|---|---|---:|---:|---:|
| Supplier on-time delivery | Product Transfer | 3 | 3 | 1 |
| Supplier quality defect rate | Product Transfer | 3 | 3 | 2 |
| Assembly cycle time | Product Transfer | 2 | 2 | 1 |
| Cost per assembled unit | Product Transfer | 2 | 1 | 2 |
| Logistics lead time | Product Transfer | 3 | 2 | 1 |
| Inventory turns | Product Transfer | 2 | 1 | 2 |
| OTIF | Product Transfer | 3 | 2 | 2 |
| Order cycle time | Product Transfer | 2 | 2 | 2 |
| Forecast accuracy | Product Transfer | 2 | 1 | 2 |
| Customer fill rate | Product Transfer | 3 | 1 | 2 |
| Technology ramp-up time | Technology Transfer | 1 | 3 | 2 |
| First-pass yield | Technology Transfer | 2 | 3 | 3 |
| Learning-curve productivity | Technology Transfer | 1 | 2 | 2 |
| Flexibility index | Technology Transfer | 2 | 2 | 3 |
| Revenue from new products (%) | Technology Transfer | 1 | 2 | 1 |
| Technology adoption cost | Technology Transfer | 1 | 2 | 2 |
| Plant utilization | Technology Transfer | 2 | 2 | 3 |
| Supplier on-time receipts | Technology Transfer | 3 | 3 | 2 |
| Supplier quality pass | Technology Transfer | 3 | 3 | 2 |
| Cycle time reduction | Technology Transfer | 2 | 2 | 3 |
| OEE | Facility Design | 2 | 2 | 3 |
| Production lead time | Facility Design | 2 | 2 | 2 |
| Customer fulfillment cycle time | Facility Design | 3 | 2 | 2 |
| Total lifecycle cost | Facility Design | 2 | 2 | 3 |
| ESG performance index | Facility Design | 2 | 2 | 3 |
| Workforce safety incident rate | Facility Design | 2 | 2 | 3 |
| Service uptime | Facility Design | 2 | 2 | 3 |
| Planned maintenance ratio | Facility Design | 1 | 2 | 3 |
| Labor content accuracy | Facility Design | 1 | 2 | 2 |
| Closed-loop recovery rate | Facility Design | 1 | 1 | 2 |

## Construction logic

The primary configuration and KPI families come directly from the manuscript:
supplier/assembly/logistics performance for PT; innovation/ramp-up/process
stabilisation for TT; and end-to-end efficiency/lifecycle/operational control
for FD. Cross-configuration values represent secondary relevance rather than
absence. The core-process and resilience-driver baselines were also aligned
with the manuscript tables.

These numerical values are an explicit, literature-informed design-science
starting point—not empirical calibration. Before strong validity claims, the
matrix must be reviewed through the protocol in
`CALIBRATION_PROTOCOL.md`. Report any expert-approved revision as a new model
version and preserve the previous mapping for reproducibility.
