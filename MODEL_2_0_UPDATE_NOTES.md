# Model 2.0 methodological update

The Streamlit header displays only the decision-model version and fuzzy-rule-
base version. Both scientific components are identified as **2.0**, so a mixed
deployment can be detected immediately.

This release is based directly on the v11 `app.py` used for the reported
Product Transfer, Technology Transfer, and Facility Design runs. It does not
include the later experimental contextual-weight transformation.

## Implemented changes

0. **Generate–validate–repair pipeline**
   - The first OpenRouter call generates a candidate explanation.
   - A rejected textual draft is passed to a second call together with the
     exact validator issues and frozen canonical evidence.
   - The repaired response is validated again before display.
   - Exact ties may share one score/rule statement only when all items,
     scores, and dominant rules match the canonical evidence.
   - Empty responses, reasoning leakage, and failed repairs still trigger the
     deterministic fallback.
   - Up to three calls are allowed per section so an empty response can be
     followed by a textual generation and then a directed repair.
   - Internal validator revision `2.0.5` invalidates cached pre-repair
     narratives while the public decision model and fuzzy rule base remain
     version `2.0`.
   - Unicode hyphens and Markdown emphasis are normalized only for evidence
     matching, so valid grouped ties are not rejected because of typography.
   - The completion allowance is 2,400 tokens. Explicit final-answer and
     quoted-paragraph blocks are extracted before validation; raw and
     displayed text remain separately auditable.
   - Reader-facing interpretations now translate the fuzzy trace into natural
     managerial language. Scores, numerical inputs, and rule identifiers are
     excluded from the narrative and retained in a collapsed supporting-
     evidence table.

1. **One qualitative-label policy**
   - Low: `0 <= score < 1`
   - Medium: `1 <= score < 2`
   - High: `2 <= score <= 3`
   - The interface, canonical evidence, rule traces, and exported model
     configuration now call the same function.

2. **Frozen interpretive context**
   - Objective, resolved industry, and resolved role are frozen when
     `Analyze` is clicked.
   - Custom `Other` values are resolved before freezing.
   - Context is exported with a separate identifier.
   - Context is explicitly excluded from fuzzy scoring and numerical
     validation.

3. **Reviewer-ready ablation suite**
   - 5S contribution removed.
   - Lifecycle contribution removed.
   - Both contributions removed.
   - Each case reconstructs and recomputes the Sugeno rule consequents after
     renormalizing the remaining positive design weights.
   - Kendall tau-b, p-value, and tie-aware priority-set Jaccard are reported.

4. **Expanded reproducibility metadata**
   - Epsilon and alpha-cut status.
   - Exact output-band policy.
   - Rule-design-weight calibration status.
   - LLM token limit, routed model, prompt hash, rejection reason, and
     deterministic fallback.

5. **LLM reliability**
   - `model="openrouter/free"` remains unchanged.
   - The response allowance was increased for free reasoning models.
   - The strict grounding validator and deterministic fallback remain the
     authority boundary.

## Methodological boundary

The rule-design weights `0.50/0.30/0.20` and internal `stage_gain=0.80` remain
versioned author-designed parameters. Robustness testing can assess their
effects, but expert elicitation or case-based calibration is still required
before claiming empirical validity.

## Verification

- Python syntax/AST checks passed.
- Command-line replication completed.
- Fuzzy-engine output-band consistency passed.
- Three-case ablation smoke tests passed.
- All 52 automated tests passed.
