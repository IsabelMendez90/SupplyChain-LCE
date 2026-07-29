import ast
import unittest
from pathlib import Path

import pandas as pd
from llm_grounding import grounding_issues, validate_grounded_output


def load_app_function(name):
    """Load one pure function from app.py without executing Streamlit."""
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"pd": pd, "max": max}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace[name]


class McdaRegressionTests(unittest.TestCase):
    def test_promethee_accepts_decimal_preferences(self):
        promethee_compare = load_app_function("promethee_compare")
        matrix = {
            "KPI A": {"Product Transfer": 2.5, "Technology Transfer": 1.2},
            "KPI B": {"Product Transfer": 1.1, "Technology Transfer": 2.0},
            "KPI C": {"Product Transfer": 0.7, "Technology Transfer": 1.4},
        }
        ranks = promethee_compare(matrix)
        self.assertEqual(set(ranks.index), set(matrix))
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

    def test_llm_mode_has_no_manual_selector(self):
        source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
        self.assertNotIn('st.radio(\n        "Explanation mode"', source)


if __name__ == "__main__":
    unittest.main()
