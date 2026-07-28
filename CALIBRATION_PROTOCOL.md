# Fuzzy-model calibration protocol

This file specifies a prospective validation procedure. Its presence does not
mean that expert or industrial validation has already been completed.

## 1. Expert panel

Recruit independent participants with documented experience in at least one of
the three manufacturing-system configurations and in supply-chain
decision-making. Report inclusion criteria, sector, role, years of experience,
configuration expertise, conflicts of interest, and the number invited,
participating, and completing every round. Authors may facilitate the process
but should not be the only raters.

## 2. Round 1: construct and applicability review

For every process, KPI, and resilience driver, ask experts to assess:

- whether the item is applicable to each system configuration;
- baseline relevance on the declared `0–3` scale;
- relevant 5S dimensions and their association strengths;
- relevant lifecycle stages and their membership strengths; and
- whether any construct, item, or stage is missing.

Keep `N/A` separate from the numeric relevance scale. Collect a short rationale
for every `N/A`, `0`, or extreme rating.

## 3. Round 2: fuzzy-parameter and rule review

Show the anonymized Round-1 distribution and the proposed membership plots.
Ask experts to assess the Low/Medium/High breakpoints, the three rule-design
weights, the five Sugeno singleton values, and each rule's direction and
confidence. Do not change a confidence coefficient from `1.0` without
documenting the panel evidence and decision rule.

## 4. Consensus and revision

Predefine the consensus statistic and threshold before inspecting responses.
Report medians and dispersion for ordinal ratings and an appropriate
agreement statistic for the panel design. Flag items below the threshold for a
second review rather than silently averaging disagreement. Archive the
de-identified item-level ratings and a change log from proposed to final
parameters.

## 5. Calibration and holdout validation

Use one set of expert cases to calibrate parameters and a separate set to
evaluate ranking agreement, top-priority agreement, and decision usefulness.
Compare the calibrated model with:

- the uncalibrated fuzzy model;
- a crisp baseline using the same constructs; and
- at least one fully specified MCDA method.

Do not tune parameters on the holdout cases.

## 6. Sensitivity analysis

Vary membership breakpoints, rule-design weights, singleton consequents,
stage-gain values, and 5S inputs over prespecified ranges. Report Kendall
tau-b with p-values and empirical intervals, top-k retention, and cases where
the leading recommendation changes. Use fixed random seeds and archive every
scenario configuration.

## 7. Reporting boundary

Until this protocol has been executed, describe the numerical mappings and
rule base as an author-designed, literature-informed, provisional
design-science artifact. Software reproducibility demonstrates that results
can be regenerated; it does not demonstrate empirical validity.
