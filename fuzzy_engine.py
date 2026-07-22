"""Deterministic zero-order Sugeno inference engine.

This module has no Streamlit, network, or LLM dependency. It is the
computationally reproducible core of the DSS.
"""

from itertools import product


EPSILON = 1e-12
FUZZY_RULE_BASE_VERSION = "1.0-provisional"

FUZZY_MEMBERSHIP_PARAMETERS = {
    "Low": (0.0, 0.0, 0.20, 0.50),
    "Medium": (0.20, 0.50, 0.80),
    "High": (0.50, 0.80, 1.0, 1.0),
}

SUGENO_CONSEQUENTS = {"Low": 0.5, "Medium": 1.5, "High": 2.5}

FUZZY_RULE_PROVENANCE = {
    "baseline_relevance": {
        "source_keys": ["lambert2008supply", "gunasekaran2004framework"],
        "role": "configuration-sensitive process and KPI relevance",
    },
    "5s_associations": {
        "source_keys": ["molina2021sensing", "molina2024comprehensive", "supplychaingdev"],
        "role": "Social, Sustainable, Sensing, Smart, and Safe alignment",
    },
    "lifecycle_associations": {
        "source_keys": ["garetti2012sustainable", "supplychaingdev"],
        "role": "lifecycle-dependent relevance",
    },
    "resilience_logic": {
        "source_keys": ["ivanov2020viability", "wong2024empirical"],
        "role": "buffers, diversification, multisourcing, and ecosystem response",
    },
    "combination_rules": {
        "type": "author-designed design-science synthesis",
        "status": "provisional pending structured expert elicitation and calibration",
    },
}

# Antecedent order: baseline relevance, 5S alignment, lifecycle relevance.
SUGENO_RULES = {
    ("Low", "Low", "Low"): "Low",
    ("Low", "Low", "Medium"): "Low",
    ("Low", "Low", "High"): "Low",
    ("Low", "Medium", "Low"): "Low",
    ("Low", "Medium", "Medium"): "Low",
    ("Low", "Medium", "High"): "Medium",
    ("Low", "High", "Low"): "Low",
    ("Low", "High", "Medium"): "Medium",
    ("Low", "High", "High"): "Medium",
    ("Medium", "Low", "Low"): "Low",
    ("Medium", "Low", "Medium"): "Medium",
    ("Medium", "Low", "High"): "Medium",
    ("Medium", "Medium", "Low"): "Medium",
    ("Medium", "Medium", "Medium"): "Medium",
    ("Medium", "Medium", "High"): "Medium",
    ("Medium", "High", "Low"): "Medium",
    ("Medium", "High", "Medium"): "High",
    ("Medium", "High", "High"): "High",
    ("High", "Low", "Low"): "Medium",
    ("High", "Low", "Medium"): "Medium",
    ("High", "Low", "High"): "High",
    ("High", "Medium", "Low"): "Medium",
    ("High", "Medium", "Medium"): "High",
    ("High", "Medium", "High"): "High",
    ("High", "High", "Low"): "High",
    ("High", "High", "Medium"): "High",
    ("High", "High", "High"): "High",
}


def clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def trimf(x, a, b, c):
    """Triangular membership with safe boundary handling."""
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership, including left and right shoulders."""
    x = float(x)
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def fuzzify_unit(value):
    """Map an input in [0, 1] to Low/Medium/High memberships."""
    x = clamp(value, 0.0, 1.0)
    return {
        "Low": trapmf(x, *FUZZY_MEMBERSHIP_PARAMETERS["Low"]),
        "Medium": trimf(x, *FUZZY_MEMBERSHIP_PARAMETERS["Medium"]),
        "High": trapmf(x, *FUZZY_MEMBERSHIP_PARAMETERS["High"]),
    }


def sugeno_fuzzy_score(base, s_alignment, lifecycle_relevance, epsilon=EPSILON):
    """Return a priority score in [0, 3] and a complete inference trace."""
    inputs = {
        "baseline": clamp(float(base) / 3.0, 0.0, 1.0),
        "5s_alignment": clamp(s_alignment, 0.0, 1.0),
        "lifecycle_relevance": clamp(lifecycle_relevance, 0.0, 1.0),
    }
    memberships = {key: fuzzify_unit(value) for key, value in inputs.items()}

    activated_rules = []
    weighted_sum = 0.0
    firing_sum = 0.0
    for rule_number, (antecedents, output_label) in enumerate(SUGENO_RULES.items(), start=1):
        baseline_label, s_label, lifecycle_label = antecedents
        firing = (
            memberships["baseline"][baseline_label]
            * memberships["5s_alignment"][s_label]
            * memberships["lifecycle_relevance"][lifecycle_label]
        )
        if firing <= 0.0:
            continue
        consequent = SUGENO_CONSEQUENTS[output_label]
        weighted_sum += firing * consequent
        firing_sum += firing
        activated_rules.append(
            {
                "rule_id": f"R{rule_number:02d}",
                "if": {
                    "baseline": baseline_label,
                    "5s_alignment": s_label,
                    "lifecycle_relevance": lifecycle_label,
                },
                "then": output_label,
                "firing_strength": round(float(firing), 6),
                "consequent": consequent,
            }
        )

    score = float(base) if firing_sum <= epsilon else weighted_sum / (firing_sum + epsilon)
    score = clamp(score, 0.0, 3.0)
    trace = {
        "rule_base_version": FUZZY_RULE_BASE_VERSION,
        "inputs": inputs,
        "memberships": memberships,
        "activated_rules": activated_rules,
        "firing_sum": float(firing_sum),
        "epsilon": epsilon,
        "defuzzification": "zero-order Sugeno weighted average",
        "antecedent_operator": "product t-norm",
        "score": round(score, 6),
    }
    return score, trace


def validate_engine(grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Run deterministic coverage, range, and diagonal-monotonicity checks."""
    coverage_failures = [x for x in grid if sum(fuzzify_unit(x).values()) <= 0.0]
    range_failures = []
    for baseline, s_value, lifecycle in product((0.0, 1.5, 3.0), grid, grid):
        score, _ = sugeno_fuzzy_score(baseline, s_value, lifecycle)
        if not 0.0 <= score <= 3.0:
            range_failures.append((baseline, s_value, lifecycle, score))

    diagonal_scores = [sugeno_fuzzy_score(3.0 * x, x, x)[0] for x in grid]
    monotonicity_failures = [
        (grid[i - 1], grid[i], diagonal_scores[i - 1], diagonal_scores[i])
        for i in range(1, len(grid))
        if diagonal_scores[i] + 1e-12 < diagonal_scores[i - 1]
    ]
    return {
        "rule_count": len(SUGENO_RULES),
        "coverage_failures": coverage_failures,
        "range_failures": range_failures,
        "monotonicity_failures": monotonicity_failures,
        "passed": (
            len(SUGENO_RULES) == 27
            and not coverage_failures
            and not range_failures
            and not monotonicity_failures
        ),
    }
