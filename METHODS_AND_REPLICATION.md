# Methods and replication specification

## 1. Inputs

The DSS combines three bounded antecedents:

1. configuration-specific baseline relevance, originally coded on `[0, 3]`
   and normalized to `[0, 1]`;
2. compatibility between the user's five Industry 5.0 priorities and the
   item's 5S associations;
3. lifecycle relevance for the selected LCE stage, multiplied by the declared
   `stage_gain` parameter.

The five priorities are Social, Sustainable, Sensing, Smart, and Safe. They
are independent priority intensities in `[0, 1]`; they are not compositional
weights and therefore are not normalized to sum to one.

Objective, industry, and role are interpretive metadata. The application
freezes and exports them with the run so the optional language rendering can
be audited, but they are excluded from the deterministic scoring function and
from the numerical run identifier.

Applicability is encoded separately from relevance. The engine supports
`None` as a structural exclusion, which invokes a gate and bypasses inference.
A numeric baseline of `0`, by contrast, means an applicable item with minimal
initial relevance and remains eligible for fuzzy inference. In the current
30-KPI matrix, every KPI is applicable across all three configurations, so no
KPI uses `None` or displays `N/A`. Structural exclusion is retained only as an
auditable capability for a future expert-justified case.

## 2. Fuzzy inference

The implementation uses a zero-order type-1 Sugeno system. Each antecedent has
its own declared partition because baseline relevance, 5S alignment, and
lifecycle relevance are different constructs. Low and High use trapezoidal
shoulder functions; Medium uses a triangular function. Exact shapes and
breakpoints are exported in `config/fuzzy_model.json`.

The complete rule table was constructed monotonically from an explicit
design-priority vector: baseline relevance `0.50`, 5S alignment `0.30`, and
lifecycle relevance `0.20`. Linguistic antecedents are encoded Low `0`, Medium
`1`, and High `2`. Each of the 27 antecedent combinations receives the
zero-order Sugeno singleton:

```text
z_r = 1.5 × (0.50 q_baseline + 0.30 q_5S + 0.20 q_lifecycle)
```

where each `q` is the corresponding ordinal code. The weights construct the
rule table once and are not re-applied during inference.

For rule `r`, the firing strength is:

```text
w_r = mu_baseline × mu_5S × mu_lifecycle
```

The AND operator is the product t-norm. The 27 rules contain 19 unique numeric
singletons spanning `0.00–3.00`; retaining the unrounded numeric consequences
prevents lifecycle relevance from being erased by coarse output categories.
Reader-facing labels use the manuscript's three declared bands: Low
`[0,1)`, Medium `[1,2)`, and High `[2,3]`. These labels do not replace the
numeric consequences used in defuzzification.
Defuzzification is:

```text
score = sum(w_r × z_r) / (sum(w_r) + 1e-12)
```

Alpha-cuts are not used because the implementation is a zero-order Sugeno
system. The complete 27-rule base is machine readable in
`config/fuzzy_model.json`.

Each rule also exposes a confidence coefficient in `[0, 1]`:

```text
w'_r = w_r × confidence_r
```

All coefficients are initially `1.0`. They must remain unchanged unless a
documented expert-elicitation or calibration procedure supports an adjustment.

## 3. Traceability

Every scored item records:

- normalized antecedent inputs;
- Low/Medium/High membership degrees;
- activated rule identifiers and antecedents;
- raw firing strengths, rule confidences, effective firing strengths, and
  consequent values;
- firing-strength sum;
- epsilon, t-norm, and defuzzification method;
- applicability-gate status, final score, and rule-base version.

The run JSON also separates the deterministic decision authority from the
optional language-rendering layer.

## 4. Robustness protocol

The application implements independent multiplicative perturbation of each 5S
priority:

```text
x'_j = clip(x_j × (1 + u_j), 0, 1)
u_j ~ Uniform(-delta, +delta)
```

The interface exposes the perturbation magnitude and repetition count and uses
seed 42 by default. It reports mean Kendall tau-b, an empirical 95% interval,
and mean tie-aware priority-set retention. The priority set contains every item
tied at the third-rank score threshold, avoiding alphabetical or row-order
selection among exact ties. Recommended publication experiments are 5,000 or
10,000 repetitions at `delta = 0.05`, `0.10`, and `0.20`.

The application also implements membership-threshold sensitivity. Every
interior breakpoint is shifted together by `-delta` and `+delta`, while the
domain endpoints remain fixed at `0` and `1`. The supported range is
`delta <= 0.10`. For applicable KPIs only, the app reports Kendall tau-b,
its p-value, and tie-aware priority-set retention against the unshifted
ranking. This is a transparent global threshold-location test; independent
breakpoint perturbations should additionally be reported in a full calibration
study.

Counterfactual 5S influence is measured directly. One Industry 5.0 priority at
a time is set to `0` and `1`, with the other four priorities held at the
selected values. The mean and maximum KPI score ranges and the affected-KPI
count are reported for the selected supply-chain configuration.

Component-weight ablation is reported in three separate cases: 5S contribution
removed, lifecycle contribution removed, and both removed. In each case the
remaining positive rule-design weights are renormalized before all fuzzy rules
are reconstructed and recomputed. The app reports Kendall tau-b, its p-value,
and tie-aware priority-set Jaccard as primary ranking diagnostics; Pearson
correlation is retained only as a secondary score-vector diagnostic.

## 5. Benchmark comparison

For the selected supply-chain configuration, KPI rows are alternatives and the
three original pre-inference antecedents are benefit criteria: normalized
baseline relevance, 5S alignment, and lifecycle relevance. TOPSIS, a weighted
sum, and a PROMETHEE-style net flow use the declared design weights
`0.50/0.30/0.20`. They never use fuzzy output scores as their input criteria.
Kendall tau-b and its p-value compare each crisp ranking with the fuzzy
ranking.

This is an internal convergent-method comparison, not independent industrial
validation. The weighted-sum method is not presented as AHP because no expert
pairwise comparison matrix has been elicited. The PROMETHEE-style diagnostic
uses a simple positive-difference preference function and must not be claimed
as a validated benchmark without elicited preference and indifference
thresholds.

## 6. LLM boundary and failure mode

The LLM receives a frozen canonical evidence object. Its system instruction
forbids recalculation, re-ranking, invented numbers, unsupported causal claims,
and industrial-validation claims. Temperature is zero. If the API key is
missing or the call fails, the app renders the deterministic score-and-rule
explanation.

Before display, validator revision `2.0.3` checks the draft against the exact
payload. It rejects: reasoning or prompt leakage; numbers or rule identifiers
absent from the payload; incorrect item–score or item–rule associations;
descending-order violations (while permitting exact ties); and unsupported
normative, causal, risk, performance, advantage, or outcome claims. The first
call generates a draft. If a call returns no text, the next call performs a
fresh generation. If validation fails and a textual draft exists, the next
available call receives that draft, the machine-readable rejection reasons,
and the same canonical evidence as a directed repair task. The repaired text
is validated again. A maximum of three calls is allowed per section,
permitting the sequence empty generation → textual generation → repair.
Sections that still fail after the declared limit are replaced by
deterministic explanations. Every generation and repair records the routed
model, parent call, prompt hash, and rejection reasons.

The OpenRouter completion allowance is 2,400 tokens. This is a transport
allowance for routed reasoning models, not a scientific model parameter and
not the intended reader-facing length. Before validation, explicit
`Final answer:`, `Final response:`, or quoted `Paragraph:` blocks are extracted
from longer model output. The raw draft, extracted candidate, and displayed
text are archived separately. Only the extracted candidate that passes the
grounding validator is displayed.

When exact scores or rule identifiers are required, validation is normally
local to each mentioned item. Baseline, 5S, lifecycle, list-numbering, and
neighboring-item values cannot satisfy this requirement. A tie-aware exception
allows one shared score and rule statement only when every supplied item is
mentioned, every item has exactly the same canonical score and rule, and the
text explicitly signals the shared/tied status.
For generic chat tasks, the renderer may omit items and vary its prose
structure. For the three principal interpretation sections, the payload is first reduced
to the tie-aware priority set at or above the third-position score cutoff. The
renderer must cover every item in this smaller supplied set, but remains free
to vary sentence structure, vocabulary, and presentation order within exact
ties.

The OpenRouter free router may realize different models across calls. This is
handled as a multi-model robustness condition rather than hidden variation:
the actual model, call identifier, prompt hash, complete draft, validator
outcome, and fallback status are archived for each attempt. The validator does
not reject harmless stylistic variation, list numbering, word-count references,
or the term `Industry 5.0`; it remains strict about scientific content.

Transport and grounding failures are reported separately. A response containing
no reader-facing text retains the actual routed-model identifier and is marked
`empty_response`; it is not mislabeled as an API connection failure. Genuine
API exceptions record a sanitized exception type, HTTP status, provider code,
and short message without exposing the configured key.

Exact prose is not a replication target. Fidelity is evaluated by numerical
agreement, ordering agreement, rule-reference agreement, unsupported-claim
rate, and contradiction rate.

## 7. Validation still required before strong empirical claims

Software reproducibility does not validate the model's assumptions. A revised
study should separately report:

- expert-selection criteria and conflicts of interest;
- elicitation and revision of baseline matrices, 5S associations, lifecycle
  associations, membership breakpoints, and rule consequents;
- inter-expert consensus and treatment of disagreement;
- holdout cases or case-based external assessment;
- sensitivity to membership breakpoints and stage gain;
- comparison against a crisp ablation and a fully specified MCDA baseline.

Until those steps are completed, the app should be described as a transparent
research prototype rather than a fully industrially validated expert system.

## 8. Automated verification

`validate_engine()` and the unit tests verify:

- all 27 antecedent combinations and all 19 unique numeric consequents;
- membership coverage for each of the three input-specific partitions;
- output range `[0, 3]`;
- monotonicity when each antecedent increases while the other two are held
  fixed;
- deterministic repeatability;
- structural handling of `N/A` items; and
- machine-readable trace completeness.

These are software and internal-validity checks. They do not replace expert,
case-based, or industrial external validation.
