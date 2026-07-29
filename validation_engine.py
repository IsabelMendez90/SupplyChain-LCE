"""API-independent validation and comparison utilities for the DSS."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from decision_model import (
    BASE_KPIS,
    FIVE_S,
    S_TAGS_KPI,
    STAGE_TAGS_KPI,
    is_applicable,
    s_boost,
    score_all,
    stage_boost,
)
from fuzzy_engine import RULE_DESIGN_WEIGHTS


def format_p_value(value):
    """Format a p-value without reporting a rounded zero."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def tie_aware_top_items(series, cutoff=3):
    """Return every item tied at or above the requested rank cutoff."""
    numeric = pd.Series(series, dtype=float).sort_values(
        ascending=False, kind="mergesort"
    )
    if numeric.empty:
        return []
    cutoff_index = min(int(cutoff) - 1, len(numeric) - 1)
    cutoff_score = float(numeric.iloc[cutoff_index])
    return sorted(
        numeric[numeric >= cutoff_score - 5e-4].index.tolist()
    )


def kpi_antecedent_matrix(weights_5s, stage, system, stage_gain=0.8):
    """Return independent crisp MCDA criteria used before fuzzy inference."""
    if system not in next(iter(BASE_KPIS.values())):
        raise KeyError(f"Unknown system: {system}")
    rows = {}
    for item, systems in BASE_KPIS.items():
        if not is_applicable("kpis", item, system):
            continue
        rows[item] = {
            "baseline": float(systems[system]) / 3.0,
            "5s_alignment": s_boost(weights_5s, S_TAGS_KPI, item),
            "lifecycle_relevance": stage_boost(
                stage, STAGE_TAGS_KPI, item, stage_gain
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _criterion_weights(columns):
    values = pd.Series(
        {column: RULE_DESIGN_WEIGHTS[column] for column in columns},
        dtype=float,
    )
    return values / values.sum()


def topsis_rank(criteria):
    """Benefit-criterion TOPSIS on pre-inference antecedents."""
    df = pd.DataFrame(criteria, dtype=float)
    denominators = np.sqrt((df**2).sum(axis=0)).replace(
        0, np.finfo(float).eps
    )
    normalized = df / denominators
    weighted = normalized * _criterion_weights(df.columns)
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = distance_worst / (
        distance_best + distance_worst + np.finfo(float).eps
    )
    return closeness.rank(ascending=False, method="average")


def weighted_sum_rank(criteria):
    """Weighted-sum ranking on normalized pre-inference antecedents."""
    df = pd.DataFrame(criteria, dtype=float)
    weights = _criterion_weights(df.columns)
    priority = df.mul(weights, axis=1).sum(axis=1)
    return priority.rank(ascending=False, method="average")


def promethee_rank(criteria):
    """PROMETHEE-style net flow on pre-inference antecedents."""
    df = pd.DataFrame(criteria, dtype=float)
    weights = _criterion_weights(df.columns)
    preference = pd.DataFrame(
        0.0, index=df.index, columns=df.index, dtype=float
    )
    for item_i in df.index:
        for item_j in df.index:
            if item_i == item_j:
                continue
            positive_difference = (df.loc[item_i] - df.loc[item_j]).clip(
                lower=0.0
            )
            preference.loc[item_i, item_j] = float(
                positive_difference.dot(weights)
            )
    flow = preference.mean(axis=1) - preference.mean(axis=0)
    return flow.rank(ascending=False, method="average")


def convergent_mcda_comparison(
    fuzzy_scores, weights_5s, stage, system, stage_gain=0.8
):
    """Compare fuzzy ranking with crisp methods using original antecedents."""
    criteria = kpi_antecedent_matrix(
        weights_5s, stage, system, stage_gain=stage_gain
    )
    fuzzy_series = pd.Series(
        {
            item: systems[system]
            for item, systems in fuzzy_scores.items()
            if item in criteria.index
        },
        dtype=float,
    )
    ranks = {
        "fuzzy": fuzzy_series.rank(ascending=False, method="average"),
        "topsis": topsis_rank(criteria),
        "weighted_sum": weighted_sum_rank(criteria),
        "promethee": promethee_rank(criteria),
    }
    rank_frame = pd.DataFrame(ranks).dropna()
    metrics = {}
    for method in ("topsis", "weighted_sum", "promethee"):
        result = kendalltau(rank_frame["fuzzy"], rank_frame[method])
        metrics[method] = {
            "kendall_tau_b": (
                None if pd.isna(result.statistic) else float(result.statistic)
            ),
            "p_value": None if pd.isna(result.pvalue) else float(result.pvalue),
        }
    return criteria, rank_frame, metrics


def counterfactual_5s_amplitude(
    weights_5s, stage, system, stage_gain=0.8
):
    """Measure score variation across one-at-a-time 5S low/high endpoints."""
    scenario_scores = {}
    for dimension in FIVE_S:
        for endpoint in (0.0, 1.0):
            scenario_weights = dict(weights_5s)
            scenario_weights[dimension] = endpoint
            scored = score_all(
                scenario_weights, stage, stage_gain=stage_gain
            )["kpis"]
            scenario_scores[f"{dimension}={endpoint:.1f}"] = {
                item: systems[system]
                for item, systems in scored.items()
                if is_applicable("kpis", item, system)
            }
    frame = pd.DataFrame(scenario_scores, dtype=float)
    item_ranges = frame.max(axis=1) - frame.min(axis=1)
    return {
        "design": (
            "one-at-a-time endpoints: each 5S priority is set to 0 and 1 "
            "while the other four remain at the selected values"
        ),
        "mean_kpi_score_range": float(item_ranges.mean()),
        "maximum_kpi_score_range": float(item_ranges.max()),
        "affected_kpi_count": int((item_ranges > 5e-4).sum()),
        "kpi_count": int(len(item_ranges)),
        "item_ranges": {
            item: float(value) for item, value in item_ranges.items()
        },
    }


def score_comparison_metrics(base_series, alternative_series):
    """Return Pearson, Kendall tau-b, and tie-aware priority-set retention."""
    base = pd.Series(base_series, dtype=float)
    alternative = pd.Series(alternative_series, dtype=float).reindex(base.index)
    pearson = base.corr(alternative, method="pearson")
    tau = kendalltau(
        base.rank(ascending=False, method="average"),
        alternative.rank(ascending=False, method="average"),
    )
    base_top = set(tie_aware_top_items(base))
    alternative_top = set(tie_aware_top_items(alternative))
    union = base_top | alternative_top
    return {
        "pearson_score_correlation": (
            None if pd.isna(pearson) else float(pearson)
        ),
        "kendall_tau_b": (
            None if pd.isna(tau.statistic) else float(tau.statistic)
        ),
        "kendall_p_value": (
            None if pd.isna(tau.pvalue) else float(tau.pvalue)
        ),
        "base_priority_set": sorted(base_top),
        "alternative_priority_set": sorted(alternative_top),
        "priority_set_jaccard": (
            float(len(base_top & alternative_top) / len(union))
            if union
            else 1.0
        ),
    }
