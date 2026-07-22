"""Configuration-sensitive supply-chain decision model.

All scientific scoring is implemented here and in ``fuzzy_engine.py``. The
module is independent of Streamlit and external APIs so that experiments can
be reproduced from the command line.
"""

from fuzzy_engine import sugeno_fuzzy_score


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
    "Customer Driven Design": {"Product Transfer": 1, "Technology Transfer": 3, "Facility Design": 3},
    "CRM": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 3},
    "Co-Engineering": {"Product Transfer": 3, "Technology Transfer": 1, "Facility Design": 1},
    "SRM": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 1},
    "NPD": {"Product Transfer": 1, "Technology Transfer": 2, "Facility Design": 2},
    "Obtain Customer Commit.": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 2},
    "Order Fulfillment": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 3},
    "Customer Service": {"Product Transfer": 1, "Technology Transfer": 1, "Facility Design": 3},
}
BASE_KPIS = {
    "Supplier on-time delivery": {"Product Transfer": 3, "Technology Transfer": 0, "Facility Design": 0},
    "Incoming defect rate": {"Product Transfer": 3, "Technology Transfer": 0, "Facility Design": 0},
    "Assembly cost per unit": {"Product Transfer": 3, "Technology Transfer": 0, "Facility Design": 0},
    "Logistics lead time": {"Product Transfer": 3, "Technology Transfer": 0, "Facility Design": 0},
    "Ramp-up time": {"Product Transfer": 0, "Technology Transfer": 3, "Facility Design": 0},
    "First-pass yield": {"Product Transfer": 0, "Technology Transfer": 3, "Facility Design": 0},
    "Learning-curve productivity": {"Product Transfer": 0, "Technology Transfer": 3, "Facility Design": 0},
    "% revenue from new products": {"Product Transfer": 0, "Technology Transfer": 3, "Facility Design": 0},
    "OEE": {"Product Transfer": 0, "Technology Transfer": 0, "Facility Design": 3},
    "OTIF": {"Product Transfer": 0, "Technology Transfer": 0, "Facility Design": 3},
    "Lifecycle cost": {"Product Transfer": 0, "Technology Transfer": 0, "Facility Design": 3},
    "ESG index": {"Product Transfer": 0, "Technology Transfer": 0, "Facility Design": 3},
    "Safety incidents": {"Product Transfer": 0, "Technology Transfer": 0, "Facility Design": 3},
}
BASE_DRIVERS = {
    "Inventory/Capacity Buffers": {"Product Transfer": 1, "Technology Transfer": 3, "Facility Design": 3},
    "Network Diversification": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 1},
    "Multisourcing": {"Product Transfer": 3, "Technology Transfer": 1, "Facility Design": 1},
    "Nearshoring": {"Product Transfer": 2, "Technology Transfer": 2, "Facility Design": 2},
    "Platform/Plant Harmonization": {"Product Transfer": 1, "Technology Transfer": 1, "Facility Design": 3},
    "Ecosystem Partnerships": {"Product Transfer": 3, "Technology Transfer": 3, "Facility Design": 1},
}

S_TAGS_KPI = {
    "Supplier on-time delivery": {"Social": 0.6, "Sustainable": 0.2},
    "Incoming defect rate": {"Smart": 0.4, "Sensing": 0.6},
    "Assembly cost per unit": {"Smart": 0.4, "Sustainable": 0.4},
    "Logistics lead time": {"Social": 0.6, "Smart": 0.2},
    "Ramp-up time": {"Smart": 0.6, "Safe": 0.3},
    "First-pass yield": {"Smart": 0.4, "Sensing": 0.4},
    "Learning-curve productivity": {"Smart": 0.5, "Social": 0.3},
    "% revenue from new products": {"Sustainable": 0.6, "Smart": 0.4},
    "OEE": {"Smart": 0.4, "Sensing": 0.4, "Safe": 0.2},
    "OTIF": {"Social": 0.5, "Safe": 0.3},
    "Lifecycle cost": {"Sustainable": 0.8, "Smart": 0.2},
    "ESG index": {"Sustainable": 1.0},
    "Safety incidents": {"Safe": 1.0, "Social": 0.2},
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
    "Ramp-up time": {"Launch": 1.0, "Advanced Development": 0.6},
    "First-pass yield": {"Launch": 0.7, "Operation": 0.5},
    "Learning-curve productivity": {"Launch": 0.6, "Operation": 0.4},
    "OEE": {"Operation": 1.0},
    "OTIF": {"Operation": 0.7},
    "Lifecycle cost": {"End-of-Life": 0.6, "Operation": 0.4},
    "ESG index": {"End-of-Life": 0.9, "Operation": 0.5},
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


def score_matrix(base_map, matrix, weights_5s, stage, stage_gain=0.8, trace_out=None):
    _, s_tags, stage_tags = MATRIX_CONFIGURATION[matrix]
    output = {}
    for item, systems in base_map.items():
        output[item] = {}
        for system, base in systems.items():
            score, trace = sugeno_fuzzy_score(
                base=float(base),
                s_alignment=s_boost(weights_5s, s_tags, item),
                lifecycle_relevance=stage_boost(stage, stage_tags, item, stage_gain),
            )
            output[item][system] = round(score, 3)
            if trace_out is not None:
                trace_out.setdefault(matrix, {}).setdefault(item, {})[system] = trace
    return output


def score_all(weights_5s, stage, stage_gain=0.8, return_trace=False):
    trace = {} if return_trace else None
    scored = {
        matrix: score_matrix(base_map, matrix, weights_5s, stage, stage_gain, trace)
        for matrix, (base_map, _, _) in MATRIX_CONFIGURATION.items()
    }
    return (scored, trace) if return_trace else scored
