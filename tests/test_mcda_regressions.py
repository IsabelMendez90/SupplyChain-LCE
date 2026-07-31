import unittest
from pathlib import Path

import pandas as pd
from llm_grounding import (
    GROUNDING_VALIDATOR_VERSION,
    extract_chat_completion,
    grounding_issues,
    validate_grounded_output,
)
from types import SimpleNamespace
from validation_engine import promethee_rank


class McdaRegressionTests(unittest.TestCase):
    def test_promethee_accepts_decimal_preferences(self):
        matrix = pd.DataFrame(
            {
                "baseline": [1.0, 0.5, 0.25],
                "5s_alignment": [0.4, 0.8, 0.3],
                "lifecycle_relevance": [0.7, 0.6, 0.2],
            },
            index=["KPI A", "KPI B", "KPI C"],
        )
        ranks = promethee_rank(matrix)
        self.assertEqual(set(ranks.index), set(matrix.index))
        self.assertTrue(ranks.notna().all())

    def test_llm_reasoning_leakage_is_rejected(self):
        leaked = (
            "We need to produce a response using only supplied evidence. "
            "The user wants an explanation and we must preserve ordering and cite rules. "
            "Let's craft a concise paragraph."
        )
        valid = (
            "Order Fulfillment is the leading core-process priority because its reported "
            "fuzzy score and dominant rule place it above the remaining evaluated processes."
        )
        self.assertEqual(validate_grounded_output(leaked), "")
        self.assertEqual(validate_grounded_output(valid), valid)

    @staticmethod
    def canonical_payload():
        return {
            "canonical_evidence": [
                {
                    "item": "Order Fulfillment",
                    "score": 2.25,
                    "label": "High",
                    "dominant_rule": {"rule_id": "R24"},
                },
                {
                    "item": "SRM",
                    "score": 2.06,
                    "label": "High",
                    "dominant_rule": {"rule_id": "R23"},
                },
            ],
            "stage": "Operation",
        }

    def test_unsupported_normative_claim_is_rejected(self):
        text = (
            "Order Fulfillment (score 2.25, rule R24) is High. "
            "SRM (score 2.06, rule R23) is High. These results indicate "
            "vulnerabilities and SRM requires rethinking under Operation."
        )
        cleaned, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
        )
        self.assertTrue(cleaned)
        self.assertIn("unsupported_normative_or_outcome_claim", issues)
        self.assertEqual(
            validate_grounded_output(text, self.canonical_payload()), ""
        )

    def test_unsupported_number_and_rule_are_rejected(self):
        text = (
            "Order Fulfillment has score 2.75 under rule R99, while SRM has "
            "score 2.06 under rule R23 in the reported Operation evidence."
        )
        _, issues = grounding_issues(text, self.canonical_payload())
        self.assertTrue(
            any(issue.startswith("unsupported_numbers:") for issue in issues)
        )
        self.assertTrue(
            any(issue.startswith("unsupported_rule_ids:") for issue in issues)
        )

    def test_item_score_and_rule_mismatch_are_rejected(self):
        text = (
            "Order Fulfillment has score 2.06 and dominant R23 in the evidence. "
            "SRM has score 2.25 and dominant R24 in the same Operation run."
        )
        _, issues = grounding_issues(text, self.canonical_payload())
        self.assertIn("item_score_mismatch:Order Fulfillment", issues)
        self.assertIn("item_rule_mismatch:Order Fulfillment", issues)

    def test_ordering_violation_is_rejected(self):
        text = (
            "SRM (score 2.06, rule R23) is reported before Order Fulfillment "
            "(score 2.25, rule R24) in Operation."
        )
        _, issues = grounding_issues(text, self.canonical_payload())
        self.assertTrue(
            any(issue.startswith("ordering_violation:") for issue in issues)
        )

    def test_exact_grounded_trace_is_accepted(self):
        text = (
            "Order Fulfillment (score 2.25, rule R24) is reported first. "
            "SRM (score 2.06, rule R23) follows in the canonical Operation "
            "evidence; both labels are High."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                self.canonical_payload(),
                require_rule_ids=True,
                require_scores=True,
            ),
            text,
        )

    def test_auxiliary_5s_values_do_not_satisfy_fuzzy_scores(self):
        text = (
            "Order Fulfillment (High) aligns with Social (0.50) and Smart "
            "(0.50) via Rule R24, with baseline High and 5S alignment Medium. "
            "SRM (High) aligns with Social (0.50) via Rule R23."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
        )
        self.assertIn("missing_item_score:Order Fulfillment", issues)
        self.assertIn("missing_item_score:SRM", issues)

    def test_score_from_next_item_cannot_satisfy_previous_item(self):
        text = (
            "Order Fulfillment is reported as High under rule R24. "
            "SRM (score 2.06, rule R23) follows in the Operation evidence."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
        )
        self.assertIn("missing_item_score:Order Fulfillment", issues)
        self.assertNotIn("missing_item_score:SRM", issues)

    def test_parenthetical_score_and_rule_style_is_accepted(self):
        text = (
            "Order Fulfillment (2.25, R24) is reported first in the canonical "
            "evidence. SRM (2.06, R23) follows at the Operation stage, with "
            "both entries retaining their High qualitative labels."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                self.canonical_payload(),
                require_rule_ids=True,
                require_scores=True,
            ),
            text,
        )

    def test_exact_tie_can_share_one_score_and_rule_statement(self):
        payload = {
            "canonical_evidence": [
                {
                    "item": item,
                    "score": 1.95,
                    "label": "Medium",
                    "dominant_rule": {"rule_id": "R22"},
                }
                for item in (
                    "Network Diversification",
                    "Multisourcing",
                    "Ecosystem Partnerships",
                )
            ]
        }
        text = (
            "All three tied drivers—Network Diversification, Multisourcing, "
            "and Ecosystem Partnerships—have score 1.95 and dominant rule "
            "R22 in the canonical evidence. Each retains the reported Medium "
            "label for Product Transfer at the Operation stage."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                payload,
                require_rule_ids=True,
                require_scores=True,
                require_all_items=True,
            ),
            text,
        )

    def test_exact_tie_accepts_typographic_hyphens_in_item_names(self):
        items = (
            "Customer fill rate",
            "Customer fulfillment cycle time",
            "Logistics lead time",
            "OTIF",
            "Supplier on-time delivery",
            "Supplier on-time receipts",
            "Supplier quality defect rate",
            "Supplier quality pass",
        )
        payload = {
            "canonical_evidence": [
                {
                    "item": item,
                    "score": 2.55,
                    "label": "High",
                    "dominant_rule": {"rule_id": "R24"},
                }
                for item in items
            ]
        }
        text = (
            "All priority KPIs—Customer fill rate, Customer fulfillment cycle "
            "time, Logistics lead time, OTIF, Supplier on‑time delivery, "
            "Supplier on‑time receipts, Supplier quality defect rate, and "
            "Supplier quality pass—receive a High label, a score of 2.55, and "
            "are governed by dominant rule R24; each remains tied in the "
            "canonical Product Transfer evidence."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                payload,
                require_rule_ids=True,
                require_scores=True,
                require_all_items=True,
            ),
            text,
        )

    def test_exact_tie_accepts_markdown_formatted_score_and_rule(self):
        payload = {
            "canonical_evidence": [
                {
                    "item": item,
                    "score": 1.95,
                    "label": "Medium",
                    "dominant_rule": {"rule_id": "R22"},
                }
                for item in (
                    "Network Diversification",
                    "Multisourcing",
                    "Ecosystem Partnerships",
                )
            ]
        }
        text = (
            "**Network Diversification**, **Multisourcing**, and **Ecosystem "
            "Partnerships** are all labeled **Medium** with a score of "
            "**1.95**, tied under dominant rule **R22** in the canonical "
            "Product Transfer evidence."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                payload,
                require_rule_ids=True,
                require_scores=True,
                require_all_items=True,
            ),
            text,
        )

    def test_quoted_final_paragraph_is_extracted_before_validation(self):
        payload = {
            "canonical_evidence": [
                {
                    "item": item,
                    "score": 2.55,
                    "label": "High",
                    "dominant_rule": {"rule_id": "R24"},
                }
                for item in (
                    "Customer fill rate",
                    "Logistics lead time",
                    "OTIF",
                )
            ]
        }
        candidate = (
            "All three tied KPIs—Customer fill rate, Logistics lead time, "
            "and OTIF—have score 2.55 and dominant rule R24 in the canonical "
            "Product Transfer evidence. Each retains the reported High label "
            "at the Operation stage."
        )
        raw = (
            "We need to produce a factual paragraph and count the words. "
            f'Paragraph: "{candidate}"\n\nCheck word count: one, two, three.'
        )
        self.assertEqual(
            validate_grounded_output(
                raw,
                payload,
                require_rule_ids=True,
                require_scores=True,
                require_all_items=True,
            ),
            candidate,
        )

    def test_non_tied_items_cannot_share_one_score_and_rule(self):
        text = (
            "Both Order Fulfillment and SRM have score 2.25 and dominant "
            "rule R24 in the canonical evidence."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
            require_all_items=True,
        )
        self.assertTrue(
            any(
                issue.startswith(
                    ("missing_item_score:", "item_score_mismatch:")
                )
                for issue in issues
            )
        )

    def test_each_mentioned_item_requires_its_own_rule(self):
        text = (
            "Order Fulfillment (score 2.25) is reported first. "
            "SRM (score 2.06, rule R23) follows in the Operation evidence."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
        )
        self.assertIn("missing_item_rule:Order Fulfillment", issues)

    def test_required_priority_set_rejects_omitted_item(self):
        text = (
            "Order Fulfillment (score 2.25, rule R24) is reported as High "
            "within the canonical evidence for the Operation lifecycle stage."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
            require_all_items=True,
        )
        self.assertIn("missing_item:SRM", issues)

    def test_structural_numbers_do_not_trigger_false_rejection(self):
        text = (
            "1. Order Fulfillment (score 2.25, rule R24) is reported first. "
            "2. SRM (score 2.06, rule R23) follows. Both are interpreted "
            "within the Industry 5.0 context."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_rule_ids=True,
            require_scores=True,
        )
        self.assertFalse(
            any(issue.startswith("unsupported_numbers:") for issue in issues)
        )

    def test_natural_language_mode_rejects_visible_trace_notation(self):
        text = (
            "Order Fulfillment has score 2.25 under dominant rule R24. "
            "SRM follows with score 2.06 under dominant rule R23 in the "
            "reported Product Transfer ordering."
        )
        _, issues = grounding_issues(
            text,
            self.canonical_payload(),
            require_all_items=True,
            natural_language_only=True,
        )
        self.assertIn(
            "technical_numbers_not_allowed_in_narrative",
            issues,
        )
        self.assertIn(
            "technical_rule_ids_not_allowed_in_narrative",
            issues,
        )
        self.assertIn(
            "technical_trace_jargon_not_allowed_in_narrative",
            issues,
        )

    def test_grounded_natural_language_interpretation_is_accepted(self):
        text = (
            "Order Fulfillment is the leading priority, reflecting strong "
            "starting strategic importance, a moderate fit with the selected "
            "5S priorities, and strong support from the Operation stage. SRM "
            "follows in the reported ordering with moderate support from the "
            "same lifecycle stage."
        )
        self.assertEqual(
            validate_grounded_output(
                text,
                self.canonical_payload(),
                require_all_items=True,
                natural_language_only=True,
            ),
            text,
        )

    def test_llm_mode_has_no_manual_selector(self):
        source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
        self.assertNotIn('st.radio(\n        "Explanation mode"', source)

    def test_header_exposes_only_scientific_versions(self):
        source = (
            Path(__file__).resolve().parents[1]
            .joinpath("app.py")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn("Application release", source)
        self.assertNotIn("APP_RELEASE", source)
        self.assertIn("Decision model v{DECISION_MODEL_VERSION}", source)
        self.assertIn("Fuzzy rule base v{FUZZY_RULE_BASE_VERSION}", source)
        self.assertIn("REPAIR MODE:", source)
        self.assertIn('"validator_issues": repair_issues', source)
        self.assertIn("Inspect supporting fuzzy evidence", source)
        self.assertIn("natural_language_only=True", source)
        self.assertIn("issues.extend(natural_narrative_issues(out))", source)
        self.assertNotIn(
            "strict_claims=True,\n"
            "                natural_language_only=natural_language_only",
            source,
        )
        self.assertNotIn("cite its exact score", source)
        self.assertNotIn("cite its exact score and dominant", source)
        self.assertIn("Download complete analysis PDF", source)

    def test_streamlit_results_are_invalidated_by_grounding_version(self):
        source = (
            Path(__file__).resolve().parents[1]
            .joinpath("app.py")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(GROUNDING_VALIDATOR_VERSION, "2.0.5")
        self.assertIn(
            'existing_results.get("grounding_validator_version")',
            source,
        )
        self.assertIn(
            "LLM rendering pipeline: Generate → Validate → Repair →",
            source,
        )
        self.assertIn("LLM_MAX_ATTEMPTS = 3", source)
        self.assertIn("retries=LLM_MAX_ATTEMPTS", source)
        self.assertIn("LLM_MAX_TOKENS = 2400", source)
        self.assertIn('"candidate_text": out', source)
        self.assertIn('"displayed_text": out if accepted else None', source)

    def test_empty_model_content_preserves_actual_model(self):
        response = SimpleNamespace(
            model="example/free-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning="internal reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        parsed = extract_chat_completion(response, "openrouter/free")
        self.assertEqual(parsed["actual_model"], "example/free-model")
        self.assertEqual(parsed["issue"], "empty_model_content")
        self.assertTrue(parsed["has_reasoning"])

    def test_text_model_content_is_extracted(self):
        response = SimpleNamespace(
            model="example/free-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="  Grounded explanation.  ",
                        reasoning=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        parsed = extract_chat_completion(response, "openrouter/free")
        self.assertEqual(parsed["text"], "Grounded explanation.")
        self.assertIsNone(parsed["issue"])

    def test_content_part_list_is_extracted(self):
        response = SimpleNamespace(
            model="example/free-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=[
                            {"type": "text", "text": "First"},
                            {"type": "text", "text": "second"},
                        ],
                        reasoning=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        parsed = extract_chat_completion(response, "openrouter/free")
        self.assertEqual(parsed["text"], "First\nsecond")
        self.assertIsNone(parsed["issue"])


if __name__ == "__main__":
    unittest.main()
