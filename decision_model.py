"""Configuration-sensitive supply-chain decision model.

All scientific scoring is implemented here and in ``fuzzy_engine.py``. The
module is independent of Streamlit and external APIs so that experiments can
be reproduced from the command line.
"""

from fuzzy_engine import sugeno_fuzzy_score


DECISION_MODEL_VERSION = "2.0"
SYSTEMS = ["Product Transfer", "Technology Transfer", "Facility Design"]
LCE = ["Ideation", "Basic Development", "Advanced Development", "Launch", "Operation", "End-of-Life"]
FIVE_S = ["Social", "Sustainable", "Sensing", "Smart", "Safe"]

COMPETITIVE = {
    "Product Transfer": "Operational Excellence",
    "Technology Transfer": "Product Innovation",
    "Facility Design": "Customer Focus/Intimacy",
}
VALUE_CHAIN = {
    "Product Transfer": "Collaboration Networks",
    "Technology Transfer": "Strategic Business Units",
    "Facility Design": "Vertical Integration",
}
PROD_SERVICE = {
    "Product Transfer": "Assemble to Order (ATO) + Catalogue of Services",
    "Technology Transfer": "Make/Build to Order (MTO/BTO) + Configuration of Services",
    "Facility Design": "Hybrid MTS + BTO + Design of Services",
}

BASE_CORE = {
    "Customer Driven Design": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 3},
    "CRM": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Co-Engineering": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 1},
    "SRM": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 1},
    "NPD": {"Product Transfer": 1, "Technology Transfer": 3, "Facility Design": 3},
    "Obtain Customer Commit.": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 3},
    "Order Fulfillment": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 3},
    "Customer Service": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
}
BASE_KPIS = {
    # All 30 manuscript KPIs remain applicable across the three supply-chain
    # configurations. Values encode gradual contextual relevance:
    # 1 = low/secondary, 2 = important, 3 = core/critical.
    "Supplier on-time delivery": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 1},
    "Supplier quality defect rate": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 2},
    "Assembly cycle time": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 1},
    "Cost per assembled unit": {"Product Transfer": 2, "Technology Transfer": 1, "Facility Design": 2},
    "Logistics lead time": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 1},
    "Inventory turns": {"Product Transfer": 2, "Technology Transfer": 1, "Facility Design": 2},
    "OTIF": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 2},
    "Order cycle time": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 2},
    "Forecast accuracy": {"Product Transfer": 2, "Technology Transfer": 1, "Facility Design": 2},
    "Customer fill rate": {"Product Transfer": 3, "Technology Transfer": 1, "Facility Design": 2},
    "Technology ramp-up time": {"Product Transfer": 1, "Technology Transfer": 3, "Facility Design": 2},
    "First-pass yield": {"Product Transfer": 2, "Technology Transfer": 3, "Facility Design": 3},
    "Learning-curve productivity": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 2},
    "Flexibility index": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Revenue from new products (%)": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 1},
    "Technology adoption cost": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 2},
    "Plant utilization": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Supplier on-time receipts": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 2},
    "Supplier quality pass": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 2},
    "Cycle time reduction": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "OEE": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Production lead time": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 2},
    "Customer fulfillment cycle time": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 2},
    "Total lifecycle cost": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "ESG performance index": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Workforce safety incident rate": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Service uptime": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Planned maintenance ratio": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 3},
    "Labor content accuracy": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 2},
    "Closed-loop recovery rate": {"Product Transfer": 1, "Technology Transfer": 1, "Facility Design": 2},
}

KPI_PRIMARY_SYSTEM = {
    item: system
    for system, items in {
        "Product Transfer": (
            "Supplier on-time delivery",
            "Supplier quality defect rate",
            "Assembly cycle time",
            "Cost per assembled unit",
            "Logistics lead time",
            "Inventory turns",
            "OTIF",
            "Order cycle time",
            "Forecast accuracy",
            "Customer fill rate",
        ),
        "Technology Transfer": (
            "Technology ramp-up time",
            "First-pass yield",
            "Learning-curve productivity",
            "Flexibility index",
            "Revenue from new products (%)",
            "Technology adoption cost",
            "Plant utilization",
            "Supplier on-time receipts",
            "Supplier quality pass",
            "Cycle time reduction",
        ),
        "Facility Design": (
            "OEE",
            "Production lead time",
            "Customer fulfillment cycle time",
            "Total lifecycle cost",
            "ESG performance index",
            "Workforce safety incident rate",
            "Service uptime",
            "Planned maintenance ratio",
            "Labor content accuracy",
            "Closed-loop recovery rate",
        ),
    }.items()
    for item in items
}

KPI_BASELINE_PROTOCOL = {
    "scale": {
        0: "minimal relevance but still applicable",
        1: "low or secondary relevance",
        2: "important relevance",
        3: "core or critical relevance",
    },
    "applicability_policy": (
        "All current KPI items are evaluated in all three configurations. "
        "N/A is reserved for a separately justified structural exclusion."
    ),
    "calibration_status": (
        "Versioned, literature-informed design-science mapping; numerical "
        "values require structured expert elicitation and case-based validation."
    ),
}
BASE_DRIVERS = {
    "Inventory/Capacity Buffers": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 3},
    "Network Diversification": {"Product Transfer": 3, "Technology Transfer": 2, "Facility Design": 1},
    "Multisourcing": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 1},
    "Nearshoring": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Platform/Plant Harmonization": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 3},
    "Ecosystem Partnerships": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 2},
}

S_TAGS_KPI = {
    "Supplier on-time delivery": {"Social": 0.5, "Sensing": 0.3, "Smart": 0.2, "Safe": 0.2},
    "Supplier quality defect rate": {"Safe": 0.5, "Sensing": 0.4, "Smart": 0.3},
    "Assembly cycle time": {"Smart": 0.5, "Sensing": 0.4, "Social": 0.2},
    "Cost per assembled unit": {"Sustainable": 0.4, "Smart": 0.4, "Social": 0.2},
    "Logistics lead time": {"Social": 0.4, "Smart": 0.3, "Sustainable": 0.3, "Safe": 0.2},
    "Inventory turns": {"Sustainable": 0.4, "Smart": 0.3, "Sensing": 0.3},
    "OTIF": {"Social": 0.5, "Safe": 0.3, "Smart": 0.2},
    "Order cycle time": {"Social": 0.4, "Smart": 0.4, "Sensing": 0.2},
    "Forecast accuracy": {"Sensing": 0.6, "Smart": 0.4},
    "Customer fill rate": {"Social": 0.6, "Smart": 0.2, "Safe": 0.2},
    "Technology ramp-up time": {"Smart": 0.5, "Sensing": 0.3, "Safe": 0.3, "Social": 0.2},
    "First-pass yield": {"Safe": 0.4, "Sensing": 0.4, "Smart": 0.4},
    "Learning-curve productivity": {"Social": 0.5, "Smart": 0.4, "Safe": 0.2},
    "Flexibility index": {"Smart": 0.4, "Social": 0.3, "Sustainable": 0.2, "Safe": 0.2},
    "Revenue from new products (%)": {"Sustainable": 0.4, "Smart": 0.4, "Social": 0.2},
    "Technology adoption cost": {"Smart": 0.5, "Sustainable": 0.3},
    "Plant utilization": {"Smart": 0.4, "Sensing": 0.4, "Sustainable": 0.3, "Safe": 0.2},
    "Supplier on-time receipts": {"Social": 0.4, "Sensing": 0.3, "Smart": 0.2, "Safe": 0.2},
    "Supplier quality pass": {"Safe": 0.5, "Sensing": 0.4, "Smart": 0.3},
    "Cycle time reduction": {"Smart": 0.5, "Sensing": 0.3, "Sustainable": 0.2, "Social": 0.2},
    "OEE": {"Smart": 0.4, "Sensing": 0.4, "Safe": 0.3, "Sustainable": 0.2},
    "Production lead time": {"Smart": 0.4, "Sensing": 0.3, "Social": 0.2},
    "Customer fulfillment cycle time": {"Social": 0.5, "Smart": 0.3, "Safe": 0.2},
    "Total lifecycle cost": {"Sustainable": 0.6, "Smart": 0.2, "Safe": 0.2},
    "ESG performance index": {"Sustainable": 0.8, "Social": 0.2},
    "Workforce safety incident rate": {"Safe": 0.8, "Social": 0.3, "Sensing": 0.2},
    "Service uptime": {"Safe": 0.5, "Smart": 0.3, "Sensing": 0.3},
    "Planned maintenance ratio": {"Safe": 0.5, "Sensing": 0.4, "Smart": 0.3},
    "Labor content accuracy": {"Social": 0.4, "Smart": 0.3, "Sensing": 0.3, "Safe": 0.2},
    "Closed-loop recovery rate": {"Sustainable": 0.7, "Smart": 0.2, "Social": 0.2},
}
S_TAGS_CORE = {
    "Customer Driven Design": {"Smart": 0.4, "Social": 0.4, "Sustainable": 0.2},
    "CRM": {"Social": 0.8, "Smart": 0.2},
    "Co-Engineering": {"Social": 0.6, "Smart": 0.3, "Sensing": 0.2},
    "SRM": {"Social": 0.8, "Sustainable": 0.3},
    "NPD": {"Smart": 0.6, "Sustainable": 0.4},
    "Obtain Customer Commit.": {"Social": 0.7, "Smart": 0.3},
    "Order Fulfillment": {"Social": 0.8, "Smart": 0.2, "Safe": 0.2},
    "Customer Service": {"Social": 0.8, "Safe": 0.3, "Sustainable": 0.2},
}
S_TAGS_DRIVERS = {
    "Inventory/Capacity Buffers": {"Safe": 0.6, "Sustainable": 0.3},
    "Network Diversification": {"Social": 0.5, "Safe": 0.3, "Sustainable": 0.3},
    "Multisourcing": {"Sustainable": 0.4, "Social": 0.4, "Safe": 0.2},
    "Nearshoring": {"Sustainable": 0.6, "Safe": 0.2, "Social": 0.2},
    "Platform/Plant Harmonization": {"Smart": 0.5, "Sensing": 0.3, "Safe": 0.2},
    "Ecosystem Partnerships": {"Social": 0.6, "Sustainable": 0.4, "Smart": 0.2},
}

STAGE_TAGS_KPI = {
    "Supplier on-time delivery": {"Launch": 0.5, "Operation": 1.0},
    "Supplier quality defect rate": {"Advanced Development": 0.4, "Launch": 0.7, "Operation": 1.0},
    "Assembly cycle time": {"Advanced Development": 0.5, "Launch": 0.8, "Operation": 1.0},
    "Cost per assembled unit": {"Launch": 0.6, "Operation": 1.0},
    "Logistics lead time": {"Launch": 0.6, "Operation": 1.0},
    "Inventory turns": {"Operation": 1.0, "End-of-Life": 0.3},
    "OTIF": {"Launch": 0.6, "Operation": 1.0},
    "Order cycle time": {"Launch": 0.5, "Operation": 1.0},
    "Forecast accuracy": {"Ideation": 0.3, "Launch": 0.6, "Operation": 1.0},
    "Customer fill rate": {"Launch": 0.6, "Operation": 1.0},
    "Technology ramp-up time": {"Basic Development": 0.4, "Advanced Development": 0.8, "Launch": 1.0, "Operation": 0.4},
    "First-pass yield": {"Advanced Development": 0.6, "Launch": 1.0, "Operation": 0.8},
    "Learning-curve productivity": {"Launch": 1.0, "Operation": 0.8},
    "Flexibility index": {"Ideation": 0.4, "Basic Development": 0.6, "Advanced Development": 0.8, "Launch": 1.0, "Operation": 0.8},
    "Revenue from new products (%)": {"Ideation": 0.5, "Advanced Development": 0.7, "Launch": 1.0, "Operation": 0.8},
    "Technology adoption cost": {"Ideation": 0.4, "Basic Development": 0.7, "Advanced Development": 1.0, "Launch": 0.8},
    "Plant utilization": {"Launch": 0.7, "Operation": 1.0},
    "Supplier on-time receipts": {"Advanced Development": 0.5, "Launch": 0.8, "Operation": 1.0},
    "Supplier quality pass": {"Advanced Development": 0.6, "Launch": 0.9, "Operation": 1.0},
    "Cycle time reduction": {"Launch": 0.8, "Operation": 1.0},
    "OEE": {"Launch": 0.7, "Operation": 1.0},
    "Production lead time": {"Advanced Development": 0.4, "Launch": 0.7, "Operation": 1.0},
    "Customer fulfillment cycle time": {"Launch": 0.6, "Operation": 1.0},
    "Total lifecycle cost": {"Ideation": 0.5, "Basic Development": 0.7, "Advanced Development": 0.8, "Launch": 0.8, "Operation": 0.9, "End-of-Life": 1.0},
    "ESG performance index": {"Ideation": 0.5, "Basic Development": 0.7, "Advanced Development": 0.8, "Launch": 0.8, "Operation": 0.9, "End-of-Life": 1.0},
    "Workforce safety incident rate": {"Basic Development": 0.5, "Advanced Development": 0.8, "Launch": 1.0, "Operation": 1.0, "End-of-Life": 0.7},
    "Service uptime": {"Launch": 0.8, "Operation": 1.0},
    "Planned maintenance ratio": {"Launch": 0.7, "Operation": 1.0},
    "Labor content accuracy": {"Advanced Development": 0.5, "Launch": 0.8, "Operation": 1.0},
    "Closed-loop recovery rate": {"Ideation": 0.3, "Advanced Development": 0.6, "Launch": 0.6, "Operation": 0.8, "End-of-Life": 1.0},
}
STAGE_TAGS_CORE = {
    "Co-Engineering": {"Ideation": 0.8, "Basic Development": 0.6},
    "NPD": {"Advanced Development": 0.8, "Launch": 0.5},
    "Order Fulfillment": {"Operation": 0.9},
    "Customer Service": {"Operation": 0.7},
    "SRM": {"Operation": 0.6},
}
STAGE_TAGS_DRIVERS = {
    "Inventory/Capacity Buffers": {"Operation": 0.7},
    "Platform/Plant Harmonization": {"Operation": 0.7, "Launch": 0.4},
    "Nearshoring": {"Operation": 0.4, "End-of-Life": 0.4},
}

MATRIX_CONFIGURATION = {
    "core_processes": (BASE_CORE, S_TAGS_CORE, STAGE_TAGS_CORE),
    "kpis": (BASE_KPIS, S_TAGS_KPI, STAGE_TAGS_KPI),
    "drivers": (BASE_DRIVERS, S_TAGS_DRIVERS, STAGE_TAGS_DRIVERS),
}


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def s_boost(weights, tags, name):
    """Weighted compatibility between 5S priorities and an item's 5S tags."""
    item_tags = tags.get(name, {})
    total = sum(item_tags.values())
    if total == 0:
        return 0.0
    return sum(weights.get(key, 0.0) * value for key, value in item_tags.items()) / total


def stage_boost(stage, tags, name, max_gain=0.8):
    """Lifecycle relevance scaled by the declared stage-gain parameter."""
    membership = clamp01(tags.get(name, {}).get(stage, 0.0))
    return clamp01(membership * max_gain)


def is_applicable(matrix, item, system):
    """Return whether an item belongs to the selected system configuration."""
    if matrix not in MATRIX_CONFIGURATION:
        raise KeyError(f"Unknown decision matrix: {matrix}")
    base_map, _, _ = MATRIX_CONFIGURATION[matrix]
    return base_map[item][system] is not None


def scored_catalog_issues(scored):
    """Return structural incompatibilities in an externally supplied score set.

    This protects the interface from stale Streamlit state and run JSON files
    created with a different decision-model catalog. Scientific scoring keeps
    strict key access; compatibility is checked explicitly at the boundary.
    """
    if not isinstance(scored, dict):
        return ["scores must be a dictionary"]

    issues = []
    expected_matrices = set(MATRIX_CONFIGURATION)
    actual_matrices = set(scored)
    for matrix in sorted(expected_matrices - actual_matrices):
        issues.append(f"missing matrix: {matrix}")
    for matrix in sorted(actual_matrices - expected_matrices):
        issues.append(f"unexpected matrix: {matrix}")

    for matrix, (base_map, _, _) in MATRIX_CONFIGURATION.items():
        actual_items = scored.get(matrix)
        if not isinstance(actual_items, dict):
            continue
        expected_items = set(base_map)
        supplied_items = set(actual_items)
        for item in sorted(expected_items - supplied_items):
            issues.append(f"{matrix}: missing item: {item}")
        for item in sorted(supplied_items - expected_items):
            issues.append(f"{matrix}: obsolete or unknown item: {item}")

        for item in sorted(expected_items & supplied_items):
            system_values = actual_items[item]
            if not isinstance(system_values, dict):
                issues.append(f"{matrix}/{item}: system scores must be a dictionary")
                continue
            supplied_systems = set(system_values)
            expected_systems = set(SYSTEMS)
            for system in sorted(expected_systems - supplied_systems):
                issues.append(f"{matrix}/{item}: missing system: {system}")
            for system in sorted(supplied_systems - expected_systems):
                issues.append(f"{matrix}/{item}: unknown system: {system}")
            for system in sorted(expected_systems & supplied_systems):
                try:
                    float(system_values[system])
                except (TypeError, ValueError):
                    issues.append(
                        f"{matrix}/{item}/{system}: score must be numeric"
                    )
    return issues


def score_matrix(
    base_map,
    matrix,
    weights_5s,
    stage,
    stage_gain=0.8,
    trace_out=None,
    membership_parameters=None,
    s_alignment_override=None,
    lifecycle_relevance_override=None,
    rule_design_weights=None,
):
    _, s_tags, stage_tags = MATRIX_CONFIGURATION[matrix]
    output = {}
    for item, systems in base_map.items():
        output[item] = {}
        for system, base in systems.items():
            applicable = base is not None
            s_alignment = (
                clamp01(s_alignment_override)
                if s_alignment_override is not None
                else s_boost(weights_5s, s_tags, item)
            )
            lifecycle_relevance = (
                clamp01(lifecycle_relevance_override)
                if lifecycle_relevance_override is not None
                else stage_boost(stage, stage_tags, item, stage_gain)
            )
            score, trace = sugeno_fuzzy_score(
                base=float(base) if applicable else None,
                s_alignment=s_alignment,
                lifecycle_relevance=lifecycle_relevance,
                applicable=applicable,
                membership_parameters=membership_parameters,
                rule_design_weights=rule_design_weights,
            )
            output[item][system] = round(score, 3)
            if trace_out is not None:
                trace_out.setdefault(matrix, {}).setdefault(item, {})[system] = trace
    return output


def score_all(
    weights_5s,
    stage,
    stage_gain=0.8,
    return_trace=False,
    membership_parameters=None,
    s_alignment_override=None,
    lifecycle_relevance_override=None,
    rule_design_weights=None,
):
    trace = {} if return_trace else None
    scored = {
        matrix: score_matrix(
            base_map,
            matrix,
            weights_5s,
            stage,
            stage_gain,
            trace,
            membership_parameters,
            s_alignment_override,
            lifecycle_relevance_override,
            rule_design_weights,
        )
        for matrix, (base_map, _, _) in MATRIX_CONFIGURATION.items()
    }
    return (scored, trace) if return_trace else scored
