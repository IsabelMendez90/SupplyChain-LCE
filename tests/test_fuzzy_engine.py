import json
import unittest
from pathlib import Path

from decision_model import (
    BASE_CORE,
    BASE_DRIVERS,
    BASE_KPIS,
    FIVE_S,
    KPI_PRIMARY_SYSTEM,
    LCE,
    SYSTEMS,
    is_applicable,
    score_all,
    scored_catalog_issues,
)
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
        self.assertEqual(len(SUGENO_CONSEQUENTS), 19)
        self.assertEqual(min(SUGENO_CONSEQUENTS), 0.0)
        self.assertEqual(max(SUGENO_CONSEQUENTS), 3.0)

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
        for antecedents, output in SUGENO_RULES.items():
            for axis in range(3):
                if ordinal[antecedents[axis]] == 2:
                    continue
                raised = list(antecedents)
                raised[axis] = ("Medium", "High")[ordinal[antecedents[axis]]]
                self.assertLessEqual(
                    output,
                    SUGENO_RULES[tuple(raised)],
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

    def test_current_score_catalog_is_compatible(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores = score_all(weights, "Operation")
        self.assertEqual(scored_catalog_issues(scores), [])

    def test_obsolete_score_item_is_rejected_without_key_lookup(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores = score_all(weights, "Operation")
        scores["kpis"]["Incoming defect rate"] = scores["kpis"].pop(
            "Supplier quality defect rate"
        )
        issues = scored_catalog_issues(scores)
        self.assertTrue(
            any("obsolete or unknown item: Incoming defect rate" in x for x in issues)
        )
        self.assertTrue(
            any("missing item: Supplier quality defect rate" in x for x in issues)
        )

    def test_all_30_kpis_are_applicable_and_traced(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores, traces = score_all(
            weights, "Operation", return_trace=True
        )
        self.assertEqual(len(BASE_KPIS), 30)
        self.assertEqual(len(KPI_PRIMARY_SYSTEM), 30)
        for item, systems in BASE_KPIS.items():
            self.assertEqual(set(systems), set(SYSTEMS))
            for system, baseline in systems.items():
                self.assertIn(baseline, (0, 1, 2, 3))
                self.assertTrue(is_applicable("kpis", item, system))
                self.assertTrue(
                    traces["kpis"][item][system]["applicable"]
                )
                self.assertGreaterEqual(scores["kpis"][item][system], 0.0)

    def test_kpi_primary_groups_have_ten_items_each(self):
        for system in SYSTEMS:
            self.assertEqual(
                sum(
                    primary == system
                    for primary in KPI_PRIMARY_SYSTEM.values()
                ),
                10,
            )

    def test_otif_and_oee_have_cross_configuration_relevance(self):
        self.assertEqual(BASE_KPIS["OTIF"]["Product Transfer"], 3)
        self.assertEqual(BASE_KPIS["OTIF"]["Technology Transfer"], 2)
        self.assertEqual(BASE_KPIS["OTIF"]["Facility Design"], 2)
        self.assertEqual(BASE_KPIS["OEE"]["Product Transfer"], 2)
        self.assertEqual(BASE_KPIS["OEE"]["Technology Transfer"], 2)
        self.assertEqual(BASE_KPIS["OEE"]["Facility Design"], 3)

    def test_lifecycle_stage_changes_kpi_scores(self):
        weights = {name: 0.5 for name in FIVE_S}
        for system in SYSTEMS:
            stage_signatures = {
                tuple(
                    score_all(weights, stage)["kpis"][item][system]
                    for item in BASE_KPIS
                )
                for stage in LCE
            }
            self.assertGreater(
                len(stage_signatures),
                1,
                f"Lifecycle stage has no observable KPI effect for {system}.",
            )

    def test_rule_consequent_retains_lifecycle_ordering(self):
        self.assertLess(
            SUGENO_RULES[("High", "Medium", "Low")],
            SUGENO_RULES[("High", "Medium", "Medium")],
        )
        self.assertLess(
            SUGENO_RULES[("High", "Medium", "Medium")],
            SUGENO_RULES[("High", "Medium", "High")],
        )

    def test_core_and_driver_baselines_match_manuscript_tables(self):
        self.assertEqual(
            BASE_CORE["NPD"],
            {
                "Product Transfer": 1,
                "Technology Transfer": 3,
                "Facility Design": 3,
            },
        )
        self.assertEqual(
            BASE_DRIVERS["Multisourcing"],
            {
                "Product Transfer": 3,
                "Technology Transfer": 3,
                "Facility Design": 1,
            },
        )

    def test_benchmark_kpi_names_match_scored_catalog(self):
        benchmark_path = (
            Path(__file__).resolve().parents[1] / "benchmarks.json"
        )
        benchmarks = json.loads(benchmark_path.read_text(encoding="utf-8"))
        scored_benchmark_names = {
            name
            for system_values in benchmarks.values()
            for name, specification in system_values.items()
            if specification.get("DSS KPI") is True
        }
        self.assertTrue(scored_benchmark_names.issubset(BASE_KPIS))

    def test_what_if_overrides_preserve_applicability(self):
        weights = {name: 0.5 for name in FIVE_S}
        scores, traces = score_all(
            weights,
            "Operation",
            return_trace=True,
            s_alignment_override=0.5,
            lifecycle_relevance_override=0.5,
        )
        self.assertGreater(scores["kpis"]["OEE"]["Product Transfer"], 0.0)
        self.assertTrue(
            traces["kpis"]["OEE"]["Product Transfer"]["applicable"]
        )
        self.assertTrue(
            traces["kpis"]["Supplier on-time delivery"]["Product Transfer"][
                "applicable"
            ]
        )

    def test_empty_what_if_is_identical_to_full_model(self):
        weights = {name: 0.5 for name in FIVE_S}
        full = score_all(weights, "Operation", stage_gain=0.8)
        empty_ablation = score_all(
            weights,
            "Operation",
            stage_gain=0.8,
            s_alignment_override=None,
            lifecycle_relevance_override=None,
        )
        self.assertEqual(full, empty_ablation)

    def test_engine_validation_passes(self):
        self.assertTrue(validate_engine()["passed"])


if __name__ == "__main__":
    unittest.main()
