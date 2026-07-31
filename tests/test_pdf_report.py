import unittest

from pdf_report import build_analysis_pdf


class PdfReportTests(unittest.TestCase):
    @staticmethod
    def payload():
        systems = (
            "Product Transfer",
            "Technology Transfer",
            "Facility Design",
        )

        def matrix(items):
            return {
                item: {
                    system: {
                        "label": "High" if index == 0 else "Medium",
                        "score": 2.55 if index == 0 else 1.95,
                    }
                    for system in systems
                }
                for index, item in enumerate(items)
            }

        return {
            "run_id": "test123456",
            "system": "Product Transfer",
            "lce_stage": "Operation",
            "weights_5s": {
                "Social": 0.5,
                "Sustainable": 0.5,
                "Sensing": 0.5,
                "Smart": 0.5,
                "Safe": 0.5,
            },
            "context": {
                "industry": "Precision components",
                "role": "Supply Chain Analyst",
                "objective": "Improve fulfillment reliability.",
            },
            "decision_model_version": "2.0",
            "rule_base_version": "2.0",
            "matrices": {
                "core_processes": matrix(
                    ["Order Fulfillment", "Customer Service"]
                ),
                "kpis": matrix(["OTIF", "Customer fill rate"]),
                "drivers": matrix(
                    ["Network Diversification", "Multisourcing"]
                ),
            },
            "interpretations": {
                "core": "Order Fulfillment is the leading priority.",
                "kpi": "OTIF and Customer fill rate form the leading group.",
                "drivers": "Network Diversification and Multisourcing are tied.",
            },
            "comparative": None,
            "technical_evidence": {
                "core_processes": [
                    {
                        "item": "Order Fulfillment",
                        "label": "High",
                        "score": 2.55,
                        "rule": "R24",
                        "baseline": 1.0,
                        "5s_alignment": 0.5,
                        "lifecycle_relevance": 0.8,
                    }
                ],
                "kpis": [],
                "drivers": [],
            },
            "validation": {
                "internal_consistency": "Pass",
                "engine_validation": "Pass",
                "fuzzy_method": "zero-order Sugeno",
                "grounding_validator_version": "2.0.5",
                "pearson": 1.0,
                "minimum_kendall": 1.0,
                "membership_threshold_sensitivity": [],
                "monte_carlo": None,
                "mcda_metrics": {
                    "topsis": {"kendall_tau_b": 1.0, "p_value": 0.001}
                },
                "mcda_ranks": [
                    {
                        "item": "OTIF",
                        "fuzzy": 1.0,
                        "topsis": 1.0,
                        "weighted_sum": 1.0,
                        "promethee": 1.0,
                    }
                ],
                "counterfactual_5s": {
                    "mean_kpi_score_range": 0.5,
                    "maximum_kpi_score_range": 0.8,
                    "affected_kpi_count": 2,
                    "kpi_count": 2,
                    "design": "One-at-a-time endpoints.",
                },
                "llm_audit": {
                    "router": "openrouter/free",
                    "models": ["example/free"],
                    "accepted": 3,
                    "rejected": 0,
                    "api_errors": 0,
                    "empty_responses": 0,
                    "fallback_sections": [],
                },
            },
            "whatif_suite": [],
            "whatif_selected": None,
            "benchmark_meta": {
                "objective": "Contextual benchmark objective.",
                "source": "Illustrative source",
                "mapping_framework": "5S-LCE",
                "note": "Context only.",
            },
            "benchmarks": {
                "OTIF": {
                    "High": ">=95%",
                    "Medium": "85-95%",
                    "Low": "<85%",
                    "Source": "Illustrative source",
                    "DSS KPI": True,
                }
            },
        }

    def test_complete_pdf_is_generated(self):
        pdf = build_analysis_pdf(self.payload())
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 8000)
        self.assertIn(b"/Type /Page", pdf)


if __name__ == "__main__":
    unittest.main()

