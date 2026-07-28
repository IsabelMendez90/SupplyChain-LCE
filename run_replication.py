"""Run the deterministic DSS without Streamlit, OpenRouter, or an API key."""

import argparse
import hashlib
import json
from pathlib import Path

from decision_model import (
    BASE_CORE,
    BASE_DRIVERS,
    BASE_KPIS,
    S_TAGS_CORE,
    S_TAGS_DRIVERS,
    S_TAGS_KPI,
    STAGE_TAGS_CORE,
    STAGE_TAGS_DRIVERS,
    STAGE_TAGS_KPI,
    score_all,
)
from fuzzy_engine import (
    EPSILON,
    FUZZY_MEMBERSHIP_PARAMETERS,
    FUZZY_RULE_BASE_VERSION,
    FUZZY_RULE_PROVENANCE,
    RULE_DESIGN_WEIGHTS,
    SUGENO_CONSEQUENTS,
    SUGENO_RULE_CONFIDENCES,
    SUGENO_RULES,
    validate_engine,
)


def load_scenario(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serializable_rules():
    return [
        {
            "rule_id": f"R{number:02d}",
            "baseline": antecedents[0],
            "5s_alignment": antecedents[1],
            "lifecycle_relevance": antecedents[2],
            "output": output,
            "confidence": SUGENO_RULE_CONFIDENCES[f"R{number:02d}"],
        }
        for number, (antecedents, output) in enumerate(SUGENO_RULES.items(), start=1)
    ]


def model_configuration():
    return {
        "method": "zero-order Sugeno",
        "rule_base_version": FUZZY_RULE_BASE_VERSION,
        "membership_functions": FUZZY_MEMBERSHIP_PARAMETERS,
        "consequents": SUGENO_CONSEQUENTS,
        "rule_design_weights": RULE_DESIGN_WEIGHTS,
        "rule_confidences": SUGENO_RULE_CONFIDENCES,
        "epsilon": EPSILON,
        "alpha_cuts_used": False,
        "applicability_encoding": {
            "not_applicable": None,
            "numeric_zero": "applicable with very-low baseline relevance",
        },
        "baseline_scale": {
            "0": "very low relevance (applicable)",
            "1": "low relevance",
            "2": "important",
            "3": "core relevance",
        },
        "decision_matrices": {
            "baseline": {
                "core_processes": BASE_CORE,
                "kpis": BASE_KPIS,
                "drivers": BASE_DRIVERS,
            },
            "5s_associations": {
                "core_processes": S_TAGS_CORE,
                "kpis": S_TAGS_KPI,
                "drivers": S_TAGS_DRIVERS,
            },
            "lifecycle_associations": {
                "core_processes": STAGE_TAGS_CORE,
                "kpis": STAGE_TAGS_KPI,
                "drivers": STAGE_TAGS_DRIVERS,
            },
        },
        "antecedent_operator": "product t-norm",
        "defuzzification": "weighted average",
        "rules": serializable_rules(),
        "provenance": FUZZY_RULE_PROVENANCE,
    }


def run(scenario):
    scores, traces = score_all(
        scenario["weights_5s"],
        scenario["lce_stage"],
        stage_gain=scenario.get("stage_gain", 0.8),
        return_trace=True,
    )
    hash_payload = {
        "scenario": scenario,
        "model_configuration": model_configuration(),
    }
    run_id = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "run_id": run_id,
        "scenario": scenario,
        "engine_validation": validate_engine(),
        "scores": scores,
        "fuzzy_trace": traces,
        "decision_authority": "deterministic fuzzy engine",
        "llm_required": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="config/example_scenario.json")
    parser.add_argument("--output", default="replication_output.json")
    parser.add_argument("--export-config", default="config/fuzzy_model.json")
    args = parser.parse_args()

    Path(args.export_config).write_text(
        json.dumps(model_configuration(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = run(load_scenario(args.scenario))
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Run {result['run_id']} written to {args.output}")


if __name__ == "__main__":
    main()
