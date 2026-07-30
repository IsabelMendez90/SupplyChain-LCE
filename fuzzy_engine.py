"""Deterministic zero-order Sugeno inference engine.

This module has no Streamlit, network, or LLM dependency. It is the
computationally reproducible core of the DSS.
"""

from itertools import product


EPSILON = 1e-12
FUZZY_RULE_BASE_VERSION = "2.3"
LINGUISTIC_LEVELS = ("Low", "Medium", "High")

# The three antecedents represent different constructs and therefore use
# separately declared partitions. All coordinates are on a normalized [0, 1]
# universe. Low/High are shoulder trapezoids and Medium is triangular.
FUZZY_MEMBERSHIP_PARAMETERS = {
    "baseline": {
        "Low": {"type": "trapezoidal", "points": (0.0, 0.0, 0.33, 0.50)},
        "Medium": {"type": "triangular", "points": (0.25, 0.50, 0.75)},
        "High": {"type": "trapezoidal", "points": (0.50, 0.67, 1.0, 1.0)},
    },
    "5s_alignment": {
        "Low": {"type": "trapezoidal", "points": (0.0, 0.0, 0.20, 0.45)},
        "Medium": {"type": "triangular", "points": (0.20, 0.50, 0.80)},
        "High": {"type": "trapezoidal", "points": (0.55, 0.80, 1.0, 1.0)},
    },
    "lifecycle_relevance": {
        "Low": {"type": "trapezoidal", "points": (0.0, 0.0, 0.15, 0.40)},
        "Medium": {"type": "triangular", "points": (0.20, 0.50, 0.80)},
        "High": {"type": "trapezoidal", "points": (0.60, 0.85, 1.0, 1.0)},
    },
}

# These weights define the transparent construction of the 27-rule singleton
# consequents; they are not silently re-applied during inference.
RULE_DESIGN_WEIGHTS = {
    "baseline": 0.50,
    "5s_alignment": 0.30,
    "lifecycle_relevance": 0.20,
}

# Reader-facing output labels follow the three-level scale described in the
# manuscript and used by the Streamlit matrices. Numeric singleton
# consequents remain unrounded and continue to span [0, 3].
SUGENO_OUTPUT_BANDS = {
    "Low": (0.0, 1.0),
    "Medium": (1.0, 2.0),
    "High": (2.0, 3.000001),
}

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
        "type": "author-designed monotonic design-science synthesis",
        "rule_design_weights": RULE_DESIGN_WEIGHTS,
        "singleton_formula": (
            "1.5 * (0.50*q_baseline + 0.30*q_5s + "
            "0.20*q_lifecycle), q in {0,1,2}"
        ),
        "validation_status": (
            "versioned research model; numerical mappings require structured "
            "expert calibration and external case validation"
        ),
    },
}


def rule_consequents_from_weights(rule_design_weights=None):
    """Build all 27 singleton consequents from declared component weights."""
    weights = dict(RULE_DESIGN_WEIGHTS)
    if rule_design_weights is not None:
        weights.update(rule_design_weights)
    unknown = set(weights) - set(RULE_DESIGN_WEIGHTS)
    if unknown:
        raise KeyError(f"Unknown rule-design weights: {sorted(unknown)}")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("Rule-design weights must be non-negative.")
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("At least one rule-design weight must remain active.")

    return {
        antecedents: _rule_consequent(*antecedents, weights=weights)
        for antecedents in product(LINGUISTIC_LEVELS, repeat=3)
    }


def _rule_consequent(
    baseline_label, s_label, lifecycle_label, weights=None
):
    """Map an antecedent combination to an explicit singleton in [0, 3]."""
    ordinal = {"Low": 0, "Medium": 1, "High": 2}
    weights = weights or RULE_DESIGN_WEIGHTS
    total = sum(float(value) for value in weights.values())
    combined = (
        float(weights["baseline"]) * ordinal[baseline_label]
        + float(weights["5s_alignment"]) * ordinal[s_label]
        + float(weights["lifecycle_relevance"]) * ordinal[lifecycle_label]
    ) / total
    return round(1.5 * combined, 6)


def qualitative_consequent_label(consequent):
    for label, (lower, upper) in SUGENO_OUTPUT_BANDS.items():
        if lower <= float(consequent) < upper:
            return label
    raise ValueError(f"Consequent outside declared output bands: {consequent}")


# Antecedent order: baseline relevance, 5S alignment, lifecycle relevance.
# The insertion order fixes stable rule identifiers R01-R27.
SUGENO_RULES = rule_consequents_from_weights()
SUGENO_CONSEQUENTS = tuple(sorted(set(SUGENO_RULES.values())))
SUGENO_RULE_CONFIDENCES = {
    f"R{number:02d}": 1.0 for number in range(1, len(SUGENO_RULES) + 1)
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


def fuzzify_input(input_name, value, parameters=None):
    """Map one normalized antecedent to Low/Medium/High memberships."""
    parameters = parameters or FUZZY_MEMBERSHIP_PARAMETERS
    if input_name not in parameters:
        raise KeyError(f"Unknown fuzzy input: {input_name}")
    x = clamp(value, 0.0, 1.0)
    result = {}
    for label, specification in parameters[input_name].items():
        points = specification["points"]
        if specification["type"] == "triangular":
            result[label] = trimf(x, *points)
        elif specification["type"] == "trapezoidal":
            result[label] = trapmf(x, *points)
        else:
            raise ValueError(
                f"Unsupported membership type for {input_name}/{label}: "
                f"{specification['type']}"
            )
    return result


def fuzzify_unit(value, input_name="5s_alignment"):
    """Backward-compatible helper for fuzzifying a normalized input."""
    return fuzzify_input(input_name, value)


def shifted_membership_parameters(delta):
    """Return a threshold-location sensitivity case.

    Every interior breakpoint is shifted by the same declared amount while
    fixed domain endpoints remain at 0 and 1. The supported ±0.10 interval
    preserves the ordering of the declared partitions.
    """
    delta = float(delta)
    if not -0.10 <= delta <= 0.10:
        raise ValueError("Membership-threshold shift must be in [-0.10, 0.10].")
    shifted = {}
    for input_name, labels in FUZZY_MEMBERSHIP_PARAMETERS.items():
        shifted[input_name] = {}
        for label, specification in labels.items():
            points = tuple(
                point if point in (0.0, 1.0) else clamp(point + delta, 0.0, 1.0)
                for point in specification["points"]
            )
            if tuple(sorted(points)) != points:
                raise ValueError(
                    f"Shift {delta} invalidates {input_name}/{label}: {points}"
                )
            shifted[input_name][label] = {
                "type": specification["type"],
                "points": points,
            }
    return shifted


def _not_applicable_trace(base, s_alignment, lifecycle_relevance, epsilon):
    return {
        "rule_base_version": FUZZY_RULE_BASE_VERSION,
        "applicable": False,
        "structural_gate": "not applicable",
        "raw_inputs": {
            "baseline": base,
            "5s_alignment": float(s_alignment),
            "lifecycle_relevance": float(lifecycle_relevance),
        },
        "inputs": None,
        "memberships": {},
        "activated_rules": [],
        "firing_sum": 0.0,
        "epsilon": epsilon,
        "alpha_cuts_used": False,
        "defuzzification": "not executed",
        "antecedent_operator": "not executed",
        "score": 0.0,
    }


def sugeno_fuzzy_score(
    base,
    s_alignment,
    lifecycle_relevance,
    epsilon=EPSILON,
    applicable=True,
    rule_confidences=None,
    membership_parameters=None,
    rule_design_weights=None,
):
    """Return a priority score in [0, 3] and a complete inference trace.

    ``applicable=False`` is a structural gate and is distinct from a baseline
    value of zero. Rule confidences default to one and are exposed so that a
    documented expert elicitation can later calibrate individual rules.
    """
    if not applicable:
        return 0.0, _not_applicable_trace(
            base, s_alignment, lifecycle_relevance, epsilon
        )
    if base is None:
        raise ValueError("An applicable item requires a numeric baseline in [0, 3].")

    parameters = membership_parameters or FUZZY_MEMBERSHIP_PARAMETERS
    active_rules = (
        SUGENO_RULES
        if rule_design_weights is None
        else rule_consequents_from_weights(rule_design_weights)
    )
    active_design_weights = dict(RULE_DESIGN_WEIGHTS)
    if rule_design_weights is not None:
        active_design_weights.update(rule_design_weights)
    confidences = dict(SUGENO_RULE_CONFIDENCES)
    if rule_confidences:
        confidences.update(rule_confidences)
    invalid_confidences = {
        key: value
        for key, value in confidences.items()
        if not 0.0 <= float(value) <= 1.0
    }
    if invalid_confidences:
        raise ValueError(f"Rule confidences must be in [0, 1]: {invalid_confidences}")

    inputs = {
        "baseline": clamp(float(base) / 3.0, 0.0, 1.0),
        "5s_alignment": clamp(s_alignment, 0.0, 1.0),
        "lifecycle_relevance": clamp(lifecycle_relevance, 0.0, 1.0),
    }
    memberships = {
        key: fuzzify_input(key, value, parameters) for key, value in inputs.items()
    }

    activated_rules = []
    weighted_sum = 0.0
    firing_sum = 0.0
    for rule_number, (antecedents, consequent) in enumerate(
        active_rules.items(), start=1
    ):
        baseline_label, s_label, lifecycle_label = antecedents
        raw_firing = (
            memberships["baseline"][baseline_label]
            * memberships["5s_alignment"][s_label]
            * memberships["lifecycle_relevance"][lifecycle_label]
        )
        rule_id = f"R{rule_number:02d}"
        confidence = float(confidences[rule_id])
        firing = raw_firing * confidence
        if firing <= 0.0:
            continue
        output_label = qualitative_consequent_label(consequent)
        weighted_sum += firing * consequent
        firing_sum += firing
        activated_rules.append(
            {
                "rule_id": rule_id,
                "if": {
                    "baseline": baseline_label,
                    "5s_alignment": s_label,
                    "lifecycle_relevance": lifecycle_label,
                },
                "then": output_label,
                "raw_firing_strength": round(float(raw_firing), 6),
                "rule_confidence": confidence,
                "firing_strength": round(float(firing), 6),
                "consequent": consequent,
            }
        )

    if firing_sum <= epsilon:
        raise RuntimeError(
            "No fuzzy rule fired for an applicable item; check membership coverage."
        )
    score = clamp(weighted_sum / (firing_sum + epsilon), 0.0, 3.0)
    trace = {
        "rule_base_version": FUZZY_RULE_BASE_VERSION,
        "applicable": True,
        "structural_gate": "passed",
        "raw_inputs": {
            "baseline": float(base),
            "5s_alignment": float(s_alignment),
            "lifecycle_relevance": float(lifecycle_relevance),
        },
        "inputs": inputs,
        "memberships": memberships,
        "activated_rules": activated_rules,
        "firing_sum": float(firing_sum),
        "epsilon": epsilon,
        "alpha_cuts_used": False,
        "defuzzification": "zero-order Sugeno weighted average",
        "antecedent_operator": "product t-norm",
        "rule_design_weights": active_design_weights,
        "score": round(score, 6),
    }
    return score, trace


def validate_engine(grid=None):
    """Run rule, coverage, range, applicability, and monotonicity checks."""
    if grid is None:
        grid = tuple(step / 20.0 for step in range(21))

    coverage_failures = []
    for input_name in FUZZY_MEMBERSHIP_PARAMETERS:
        for value in grid:
            if sum(fuzzify_input(input_name, value).values()) <= 0.0:
                coverage_failures.append((input_name, value))

    range_failures = []
    for baseline, s_value, lifecycle in product(
        (0.0, 1.5, 3.0), grid, grid
    ):
        score, _ = sugeno_fuzzy_score(baseline, s_value, lifecycle)
        if not 0.0 <= score <= 3.0:
            range_failures.append((baseline, s_value, lifecycle, score))

    monotonicity_failures = []
    dense_grid = tuple(step / 10.0 for step in range(11))
    axes = ("baseline", "5s_alignment", "lifecycle_relevance")
    for axis in axes:
        for fixed_a, fixed_b in product(dense_grid, repeat=2):
            axis_scores = []
            for varying in dense_grid:
                values = {
                    "baseline": fixed_a,
                    "5s_alignment": fixed_b,
                    "lifecycle_relevance": fixed_b,
                }
                values[axis] = varying
                remaining = [name for name in axes if name != axis]
                values[remaining[0]] = fixed_a
                values[remaining[1]] = fixed_b
                axis_scores.append(
                    sugeno_fuzzy_score(
                        3.0 * values["baseline"],
                        values["5s_alignment"],
                        values["lifecycle_relevance"],
                    )[0]
                )
            for index in range(1, len(axis_scores)):
                if axis_scores[index] + 1e-10 < axis_scores[index - 1]:
                    monotonicity_failures.append(
                        {
                            "axis": axis,
                            "fixed": (fixed_a, fixed_b),
                            "interval": (dense_grid[index - 1], dense_grid[index]),
                            "scores": (axis_scores[index - 1], axis_scores[index]),
                        }
                    )

    not_applicable_score, not_applicable_trace = sugeno_fuzzy_score(
        None, 1.0, 1.0, applicable=False
    )
    applicability_failures = []
    if not_applicable_score != 0.0 or not not_applicable_trace.get(
        "structural_gate"
    ):
        applicability_failures.append("structural gate failed")

    return {
        "rule_count": len(SUGENO_RULES),
        "membership_partition_count": len(FUZZY_MEMBERSHIP_PARAMETERS),
        "coverage_failures": coverage_failures,
        "range_failures": range_failures,
        "monotonicity_failures": monotonicity_failures,
        "applicability_failures": applicability_failures,
        "passed": (
            len(SUGENO_RULES) == 27
            and len(SUGENO_CONSEQUENTS) == 19
            and not coverage_failures
            and not range_failures
            and not monotonicity_failures
            and not applicability_failures
        ),
    }
