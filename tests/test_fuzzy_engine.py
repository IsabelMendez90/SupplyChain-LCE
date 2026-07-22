import unittest

from decision_model import FIVE_S, SYSTEMS, score_all
from fuzzy_engine import SUGENO_RULES, fuzzify_unit, sugeno_fuzzy_score, validate_engine


class FuzzyEngineTests(unittest.TestCase):
    def test_rule_base_has_all_27_combinations(self):
        self.assertEqual(len(SUGENO_RULES), 27)

    def test_membership_domain_has_no_gaps(self):
        for step in range(101):
            value = step / 100
            self.assertGreater(sum(fuzzify_unit(value).values()), 0.0)

    def test_scores_remain_in_declared_range(self):
        for baseline in (0, 1, 2, 3):
            for s_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                for lifecycle in (0.0, 0.25, 0.5, 0.75, 1.0):
                    score, trace = sugeno_fuzzy_score(baseline, s_value, lifecycle)
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 3.0)
                    self.assertAlmostEqual(score, trace["score"], places=6)

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

    def test_engine_validation_passes(self):
        self.assertTrue(validate_engine()["passed"])


if __name__ == "__main__":
    unittest.main()
