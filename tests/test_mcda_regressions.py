import ast
import re
import unittest
from pathlib import Path

import pandas as pd


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
        source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "META_OUTPUT_PATTERNS"
                for target in node.targets
            ):
                nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "validate_llm_output":
                nodes.append(node)
        namespace = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
        validate = namespace["validate_llm_output"]
        leaked = (
            "We need to produce a response using only supplied evidence. "
            "The user wants an explanation and we must preserve ordering and cite rules. "
            "Let's craft a concise paragraph."
        )
        valid = (
            "Order Fulfillment is the leading core-process priority because its reported "
            "fuzzy score and dominant rule place it above the remaining evaluated processes."
        )
        self.assertEqual(validate(leaked), "")
        self.assertEqual(validate(valid), valid)

    def test_llm_mode_has_no_manual_selector(self):
        source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
        self.assertNotIn('st.radio(\n        "Explanation mode"', source)


if __name__ == "__main__":
    unittest.main()
