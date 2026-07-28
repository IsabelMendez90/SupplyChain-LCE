import unittest

from decision_model import FIVE_S, SYSTEMS, is_applicable, score_all
from fuzzy_engine import (
    FUZZY_MEMBERSHIP_PARAMETERS,
    SUGENO_CONSEQUENTS,
    SUGENO_RULES,
    fuzzify_input,
    shifted_membership_parameters,
    sugeno_fuzzy_score,
    validate_engine,
)


class FuzzyEngineTests(unittest.TestCase):
    def test_rule_base_has_all_27_combinations(self):
        self.assertEqual(len(SUGENO_RULES), 27)
        self.assertEqual(len(SUGENO_CONSEQUENTS), 5)

    def test_membership_domain_has_no_gaps(self):
        for input_name in FUZZY_MEMBERSHIP_PARAMETERS:
            for step in range(101):
                value = step / 100
                self.assertGreater(
                    sum(fuzzify_input(input_name, value).values()), 0.0
                )

    def test_scores_remain_in_declared_range(self):
        for baseline in (0, 1, 2, 3):
            for s_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                for lifecycle in (0.0, 0.25, 0.5, 0.75, 1.0):
                    score, trace = sugeno_fuzzy_score(baseline, s_value, lifecycle)
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 3.0)
                    self.assertAlmostEqual(score, trace["score"], places=6)

    def test_not_applicable_is_distinct_from_numeric_zero(self):
        gated_score, gated_trace = sugeno_fuzzy_score(
            None, 1.0, 1.0, applicable=False
        )
        low_score, low_trace = sugeno_fuzzy_score(
            0.0, 1.0, 1.0, applicable=True
        )
        self.assertEqual(gated_score, 0.0)
        self.assertFalse(gated_trace["applicable"])
        self.assertTrue(low_trace["applicable"])
        self.assertGreater(low_score, gated_score)

    def test_rule_outputs_are_monotonic_by_each_antecedent(self):
        ordinal = {"Low": 0, "Medium": 1, "High": 2}
        output_order = {
            label: index for index, label in enumerate(SUGENO_CONSEQUENTS)
        }
        for antecedents, output in SUGENO_RULES.items():
            for axis in range(3):
                if ordinal[antecedents[axis]] == 2:
                    continue
                raised = list(antecedents)
                raised[axis] = ("Medium", "High")[ordinal[antecedents[axis]]]
                self.assertLessEqual(
                    output_order[output],
                    output_order[SUGENO_RULES[tuple(raised)]],
                )

    def test_declared_membership_threshold_shifts_preserve_coverage(self):
        for delta in (-0.10, -0.05, 0.05, 0.10):
            parameters = shifted_membership_parameters(delta)
            for input_name in parameters:
                for step in range(101):
                    memberships = fuzzify_input(
                        input_name, step / 100, parameters
                    )
                    self.assertGreater(sum(memberships.values()), 0.0)

    def test_repeated_runs_are_identical(self):
        weights = {name: 0.5 for name in FIVE_S}
        first = score_all(weights, "Operation", stage_gain=0.8, return_trace=True)
        second = score_all(weights, "Operation", stage_gain=0.8, return_trace=True)
        self.assertEqual(first, second)

    def test_all_systems_are_returned(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores = score_all(weights, "Operation")
        for matrix in scores.values():
            for item in matrix.values():
                self.assertEqual(set(item), set(SYSTEMS))

    def test_kpi_applicability_gate_is_traced(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores, traces = score_all(
            weights, "Operation", return_trace=True
        )
        self.assertFalse(
            is_applicable("kpis", "OEE", "Product Transfer")
        )
        self.assertEqual(scores["kpis"]["OEE"]["Product Transfer"], 0.0)
        self.assertFalse(
            traces["kpis"]["OEE"]["Product Transfer"]["applicable"]
        )

    def test_engine_validation_passes(self):
        self.assertTrue(validate_engine()["passed"])


if __name__ == "__main__":
    unittest.main()
