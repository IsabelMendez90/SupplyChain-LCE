"""Complete PDF export for one frozen DSS analysis run."""

from io import BytesIO
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN = 12 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F5597")
LIGHT_BLUE = colors.HexColor("#D9EAF7")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#D7DCE2")
GREEN = colors.HexColor("#E2F0D9")
AMBER = colors.HexColor("#FFF2CC")
RED = colors.HexColor("#FCE4D6")


CHARACTER_REPLACEMENTS = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00b1": "+/-",
    "\u00d7": "x",
    "\u03c4": "tau",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
}


def _plain(value):
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    text = str(value)
    for source, target in CHARACTER_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def _paragraph(value, style):
    return Paragraph(escape(_plain(value)).replace("\n", "<br/>"), style)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4F5B66"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.2,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#333333"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.2,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.2,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
        ),
    }


def _table(headers, rows, styles, widths=None, priority_column=None):
    data = [
        [_paragraph(header, styles["table_header"]) for header in headers]
    ]
    for row in rows:
        data.append(
            [_paragraph(cell, styles["table_cell"]) for cell in row]
        )
    if widths is None:
        widths = [CONTENT_WIDTH / len(headers)] * len(headers)
    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for index in range(1, len(data)):
        commands.append(
            (
                "BACKGROUND",
                (0, index),
                (-1, index),
                colors.white if index % 2 else LIGHT_GRAY,
            )
        )
        if priority_column is not None:
            priority = _plain(rows[index - 1][priority_column]).lower()
            fill = {
                "high": GREEN,
                "medium": AMBER,
                "low": RED,
                "n/a": LIGHT_GRAY,
            }.get(priority)
            if fill:
                commands.append(
                    (
                        "BACKGROUND",
                        (priority_column, index),
                        (priority_column, index),
                        fill,
                    )
                )
    table.setStyle(TableStyle(commands))
    return table


def _key_value_table(rows, styles):
    return _table(
        ["Field", "Value"],
        rows,
        styles,
        widths=[55 * mm, CONTENT_WIDTH - 55 * mm],
    )


def _matrix_rows(matrix):
    systems = (
        "Product Transfer",
        "Technology Transfer",
        "Facility Design",
    )
    rows = []
    for item, values in matrix.items():
        row = [item]
        for system in systems:
            value = values.get(system, {})
            if isinstance(value, dict):
                label = value.get("label", "N/A")
                score = value.get("score")
                row.append(
                    label
                    if score is None
                    else f"{label} ({float(score):.3f})"
                )
            else:
                row.append(value)
        rows.append(row)
    return systems, rows


def _header_footer(canvas, doc, payload):
    canvas.saveState()
    canvas.setStrokeColor(MID_GRAY)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, PAGE_HEIGHT - 9 * mm, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 9 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#5B6570"))
    canvas.drawString(
        MARGIN,
        PAGE_HEIGHT - 7 * mm,
        "Supply-Chain Strategy Agent - Complete Analysis",
    )
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN,
        PAGE_HEIGHT - 7 * mm,
        f"Run {_plain(payload.get('run_id', 'N/A'))}",
    )
    canvas.line(MARGIN, 8 * mm, PAGE_WIDTH - MARGIN, 8 * mm)
    canvas.drawString(
        MARGIN,
        5 * mm,
        "Deterministic fuzzy decision authority; optional LLM language rendering.",
    )
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN,
        5 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def build_analysis_pdf(payload):
    """Return a polished, self-contained PDF report as bytes."""
    styles = _styles()
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        title="Supply-Chain Strategy Agent - Complete Analysis",
        author="Dr. J. Isabel Mendez and Dr. Arturo Molina",
        subject="LCE and 5S fuzzy-LLM decision-support analysis",
    )
    story = []

    story.append(Spacer(1, 15 * mm))
    story.append(
        _paragraph(
            "Supply-Chain Strategy Agent (LCE + 5S)",
            styles["title"],
        )
    )
    story.append(
        _paragraph(
            "Complete Analysis Report",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    context = payload.get("context", {})
    cover_rows = [
        ["Run ID", payload.get("run_id")],
        ["Manufacturing system", payload.get("system")],
        ["Lifecycle stage", payload.get("lce_stage")],
        ["Industry", context.get("industry")],
        ["User role", context.get("role")],
        ["Objective", context.get("objective")],
        ["Decision model", payload.get("decision_model_version")],
        ["Fuzzy rule base", payload.get("rule_base_version")],
        ["Decision authority", "Deterministic zero-order Sugeno engine"],
    ]
    story.append(_key_value_table(cover_rows, styles))
    story.append(Spacer(1, 6 * mm))
    story.append(
        _paragraph(
            "The objective, industry, and role tailor language only. "
            "They do not alter fuzzy scores, rankings, or validation metrics.",
            styles["subtitle"],
        )
    )

    story.append(PageBreak())
    story.append(_paragraph("1. Configuration", styles["h1"]))
    weight_rows = [
        [name, value] for name, value in payload.get("weights_5s", {}).items()
    ]
    story.append(_paragraph("Selected 5S priorities", styles["h2"]))
    story.append(
        _table(
            ["5S dimension", "Selected priority"],
            weight_rows,
            styles,
            widths=[70 * mm, 55 * mm],
        )
    )

    story.append(PageBreak())
    story.append(_paragraph("2. Decision Matrices", styles["h1"]))
    matrix_titles = (
        ("core_processes", "Core Processes x System"),
        ("kpis", "KPIs x System"),
        ("drivers", "Resilience Drivers x System"),
    )
    for index, (matrix_key, title) in enumerate(matrix_titles):
        if index:
            story.append(PageBreak())
        story.append(_paragraph(title, styles["h2"]))
        story.append(Spacer(1, 3 * mm))
        systems, rows = _matrix_rows(
            payload.get("matrices", {}).get(matrix_key, {})
        )
        story.append(
            _table(
                ["Decision item", *systems],
                rows,
                styles,
                widths=[
                    80 * mm,
                    (CONTENT_WIDTH - 80 * mm) / 3,
                    (CONTENT_WIDTH - 80 * mm) / 3,
                    (CONTENT_WIDTH - 80 * mm) / 3,
                ],
            )
        )
        story.append(
            _paragraph(
                "Labels denote strategic priority rather than measured "
                "operational performance. Numerical values are fuzzy scores.",
                styles["small"],
            )
        )

    story.append(PageBreak())
    story.append(_paragraph("3. Strategic Insights", styles["h1"]))
    insights = payload.get("interpretations", {})
    for key, title in (
        ("core", "Core Processes Interpretation"),
        ("kpi", "KPI Interpretation"),
        ("drivers", "Resilience Drivers Interpretation"),
    ):
        story.append(_paragraph(title, styles["h2"]))
        story.append(
            _paragraph(
                insights.get(key, "Interpretation not generated."),
                styles["body"],
            )
        )
    if payload.get("comparative"):
        story.append(_paragraph("Comparative Interpretation", styles["h2"]))
        story.append(
            _paragraph(payload["comparative"], styles["body"])
        )

    technical = payload.get("technical_evidence", {})
    if technical:
        story.append(_paragraph("Supporting fuzzy evidence", styles["h2"]))
        for key, title in (
            ("core_processes", "Core Processes"),
            ("kpis", "KPIs"),
            ("drivers", "Resilience Drivers"),
        ):
            rows = technical.get(key, [])
            if not rows:
                continue
            story.append(_paragraph(title, styles["h2"]))
            story.append(
                _table(
                    [
                        "Item",
                        "Priority",
                        "Score",
                        "Rule",
                        "Starting importance",
                        "5S fit",
                        "Lifecycle relevance",
                    ],
                    [
                        [
                            row.get("item"),
                            row.get("label"),
                            row.get("score"),
                            row.get("rule"),
                            row.get("baseline"),
                            row.get("5s_alignment"),
                            row.get("lifecycle_relevance"),
                        ]
                        for row in rows
                    ],
                    styles,
                    widths=[
                        52 * mm,
                        18 * mm,
                        17 * mm,
                        15 * mm,
                        34 * mm,
                        28 * mm,
                        34 * mm,
                    ],
                    priority_column=1,
                )
            )

    story.append(PageBreak())
    story.append(
        _paragraph("4. Validation, Robustness and Reproducibility", styles["h1"])
    )
    validation = payload.get("validation", {})
    summary_rows = [
        ["Internal consistency", validation.get("internal_consistency")],
        ["Fuzzy-engine validation", validation.get("engine_validation")],
        ["Fuzzy method", validation.get("fuzzy_method")],
        ["Grounding validator", validation.get("grounding_validator_version")],
        ["Validation stage scaling", validation.get("validation_stage_gain")],
        ["Single-perturbation Pearson correlation", validation.get("pearson")],
        ["Minimum convergent Kendall tau-b", validation.get("minimum_kendall")],
    ]
    story.append(_key_value_table(summary_rows, styles))

    llm_audit = validation.get("llm_audit", {})
    story.append(_paragraph("Language-rendering audit", styles["h2"]))
    story.append(
        _key_value_table(
            [
                ["Router", llm_audit.get("router")],
                ["Returned models", ", ".join(llm_audit.get("models", [])) or "None"],
                ["Accepted calls", llm_audit.get("accepted")],
                ["Rejected calls", llm_audit.get("rejected")],
                ["API errors", llm_audit.get("api_errors")],
                ["Empty responses", llm_audit.get("empty_responses")],
                [
                    "Fallback sections",
                    ", ".join(llm_audit.get("fallback_sections", [])) or "None",
                ],
            ],
            styles,
        )
    )

    threshold_rows = validation.get("membership_threshold_sensitivity") or []
    story.append(_paragraph("Membership-threshold sensitivity", styles["h2"]))
    if threshold_rows:
        story.append(
            _table(
                ["Threshold shift", "Kendall tau-b", "p-value", "Priority retention"],
                [
                    [
                        row.get("threshold_shift"),
                        row.get("kendall_tau_b"),
                        row.get("p_value"),
                        row.get("priority_set_retention"),
                    ]
                    for row in threshold_rows
                ],
                styles,
            )
        )
    else:
        story.append(_paragraph("Not run for this frozen analysis.", styles["body"]))

    monte_carlo = validation.get("monte_carlo")
    story.append(_paragraph("Monte Carlo robustness", styles["h2"]))
    if monte_carlo:
        story.append(
            _key_value_table(
                [
                    ["Distribution", monte_carlo.get("distribution")],
                    ["Repetitions", monte_carlo.get("repetitions")],
                    ["Seed", monte_carlo.get("seed")],
                    ["Mean Kendall tau-b", monte_carlo.get("mean_tau_b")],
                    ["95% interval", monte_carlo.get("tau_b_95pct_interval")],
                    [
                        "Mean priority-set retention",
                        monte_carlo.get("mean_priority_set_retention"),
                    ],
                ],
                styles,
            )
        )
    else:
        story.append(_paragraph("Not run for this frozen analysis.", styles["body"]))

    story.append(_paragraph("Convergent MCDA comparison", styles["h2"]))
    mcda_rows = []
    for method, metric in validation.get("mcda_metrics", {}).items():
        mcda_rows.append(
            [
                method,
                metric.get("kendall_tau_b"),
                metric.get("p_value"),
            ]
        )
    if mcda_rows:
        story.append(
            _table(
                ["Method", "Kendall tau-b", "p-value"],
                mcda_rows,
                styles,
                widths=[70 * mm, 45 * mm, 45 * mm],
            )
        )
        story.append(
            _paragraph(
                "This is an internal convergent-method comparison using the "
                "same constructs; it is not external industrial validation.",
                styles["small"],
            )
        )

    mcda_ranks = validation.get("mcda_ranks") or []
    if mcda_ranks:
        story.append(_paragraph("Aligned KPI ranks", styles["h2"]))
        story.append(
            _table(
                ["KPI", "Fuzzy", "TOPSIS", "Weighted sum", "PROMETHEE"],
                [
                    [
                        row.get("item"),
                        row.get("fuzzy"),
                        row.get("topsis"),
                        row.get("weighted_sum"),
                        row.get("promethee"),
                    ]
                    for row in mcda_ranks
                ],
                styles,
                widths=[
                    80 * mm,
                    (CONTENT_WIDTH - 80 * mm) / 4,
                    (CONTENT_WIDTH - 80 * mm) / 4,
                    (CONTENT_WIDTH - 80 * mm) / 4,
                    (CONTENT_WIDTH - 80 * mm) / 4,
                ],
            )
        )

    counterfactual = validation.get("counterfactual_5s") or {}
    story.append(CondPageBreak(35 * mm))
    story.append(_paragraph("Counterfactual 5S influence", styles["h2"]))
    if counterfactual:
        story.append(
            _key_value_table(
                [
                    [
                        "Mean KPI score range",
                        counterfactual.get("mean_kpi_score_range"),
                    ],
                    [
                        "Maximum KPI score range",
                        counterfactual.get("maximum_kpi_score_range"),
                    ],
                    [
                        "Affected KPIs",
                        f"{counterfactual.get('affected_kpi_count')}/"
                        f"{counterfactual.get('kpi_count')}",
                    ],
                    ["Design", counterfactual.get("design")],
                ],
                styles,
            )
        )

    story.append(CondPageBreak(55 * mm))
    story.append(_paragraph("5. What-If Scenarios", styles["h1"]))
    suite = payload.get("whatif_suite", [])
    if suite:
        story.append(_paragraph("Standard ablation suite", styles["h2"]))
        story.append(
            _table(
                [
                    "Deactivated components",
                    "Kendall tau-b",
                    "p-value",
                    "Priority-set Jaccard",
                    "Base priority count",
                    "Ablated priority count",
                ],
                [
                    [
                        " + ".join(case.get("deactivated_components", [])),
                        case.get("comparison", {}).get("kendall_tau_b"),
                        case.get("comparison", {}).get("kendall_p_value"),
                        case.get("comparison", {}).get("priority_set_jaccard"),
                        len(case.get("comparison", {}).get("base_priority_set", [])),
                        len(
                            case.get("comparison", {}).get(
                                "alternative_priority_set", []
                            )
                        ),
                    ]
                    for case in suite
                ],
                styles,
            )
        )
    else:
        story.append(
            _paragraph(
                "The standard ablation suite was not run for this analysis.",
                styles["body"],
            )
        )

    selected_whatif = payload.get("whatif_selected")
    if selected_whatif:
        story.append(_paragraph("Selected ablation scenario", styles["h2"]))
        comparison = selected_whatif.get("comparison", {})
        story.append(
            _key_value_table(
                [
                    [
                        "Deactivated components",
                        " + ".join(
                            selected_whatif.get("deactivated_components", [])
                        ),
                    ],
                    [
                        "Pearson score correlation",
                        comparison.get("pearson_score_correlation"),
                    ],
                    ["Kendall tau-b", comparison.get("kendall_tau_b")],
                    ["Kendall p-value", comparison.get("kendall_p_value")],
                    [
                        "Priority-set Jaccard",
                        comparison.get("priority_set_jaccard"),
                    ],
                    [
                        "Priority items",
                        ", ".join(selected_whatif.get("priority_items", [])),
                    ],
                ],
                styles,
            )
        )
        if selected_whatif.get("explanation"):
            story.append(_paragraph("Scenario explanation", styles["h2"]))
            story.append(
                _paragraph(selected_whatif["explanation"], styles["body"])
            )
        score_rows = selected_whatif.get("selected_system_scores", {})
        if score_rows:
            story.append(_paragraph("Ablated KPI priorities", styles["h2"]))
            story.append(
                _table(
                    ["KPI", "Score"],
                    sorted(
                        score_rows.items(),
                        key=lambda item: (-float(item[1]), item[0]),
                    ),
                    styles,
                    widths=[120 * mm, 40 * mm],
                )
            )

    story.append(PageBreak())
    story.append(_paragraph("6. Industry Benchmarks", styles["h1"]))
    benchmark_meta = payload.get("benchmark_meta", {})
    story.append(
        _key_value_table(
            [
                ["Objective", benchmark_meta.get("objective")],
                ["Source", benchmark_meta.get("source")],
                ["Framework", benchmark_meta.get("mapping_framework")],
                ["Note", benchmark_meta.get("note")],
            ],
            styles,
        )
    )
    benchmark_rows = payload.get("benchmarks", {})
    if benchmark_rows:
        story.append(Spacer(1, 4 * mm))
        story.append(
            _table(
                ["KPI", "High", "Medium", "Low", "Source", "In DSS"],
                [
                    [
                        item,
                        values.get("High"),
                        values.get("Medium"),
                        values.get("Low"),
                        values.get("Source"),
                        values.get("DSS KPI"),
                    ]
                    for item, values in benchmark_rows.items()
                ],
                styles,
                widths=[
                    52 * mm,
                    30 * mm,
                    35 * mm,
                    30 * mm,
                    48 * mm,
                    20 * mm,
                ],
            )
        )
    else:
        story.append(
            _paragraph("No benchmark ranges are loaded for this system.", styles["body"])
        )
    story.append(
        _paragraph(
            "Benchmark ranges are contextual references, not plant-level "
            "validation data.",
            styles["small"],
        )
    )

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _header_footer(canvas, doc, payload),
        onLaterPages=lambda canvas, doc: _header_footer(canvas, doc, payload),
    )
    return output.getvalue()
