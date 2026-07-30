"""Deterministic validation for optional LLM narrative outputs.

The validator is intentionally independent of Streamlit and OpenRouter so it
can be tested and reproduced without an API key.
"""

import re


GROUNDING_VALIDATOR_VERSION = "2.0.1"

META_OUTPUT_PATTERNS = (
    r"\bwe need to\b",
    r"\bthe user wants\b",
    r"\bi (?:need|must|should)\b",
    r"\blet(?:'|’)s craft\b",
    r"\bmust (?:preserve|use|cite|limit|explain|not)\b",
    r"\bword count\b",
    r"\bprompt requirements?\b",
    r"\bwe have evidence\b",
    r"\bneed to produce\b",
)

# These expressions flag prescriptive, comparative, causal, or risk claims
# that are not encoded by a score-and-rule trace. A deterministic fallback is
# preferable to presenting polished but unsupported managerial advice.
UNSUPPORTED_CLAIM_PATTERNS = (
    r"\bvulnerabilit(?:y|ies)\b",
    r"\brequires? rethinking\b",
    r"\bneeds? rethinking\b",
    r"\brecommend(?:s|ed|ation|ations)?\b",
    r"\bimplement(?:s|ed|ation)?\b",
    r"\badopt(?:s|ed|ion)?\b",
    r"\binvest(?:s|ed|ment)?\b",
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bcompetitive advantage(?:s)?\b",
    r"\bguarantee(?:s|d)?\b",
    r"\bwill (?:improve|reduce|increase|ensure|strengthen|enhance)\b",
    r"\breinforce(?:s|d|ing)? stability\b",
    r"\benhance(?:s|d|ing)? flexibility\b",
    r"\b(?:must|need to) (?:improve|change|increase|decrease|prioriti[sz]e)\b",
)

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![A-Za-z0-9])"
)
RULE_PATTERN = re.compile(r"\bR\d{2,}\b", flags=re.IGNORECASE)
SCIENTIFIC_NUMBER_PATTERN = re.compile(
    r"\b(?:scores?|baseline|alignment|lifecycle relevance|correlation|"
    r"kendall(?: tau(?:-b)?)?|p[- ]?value|retention|weight)"
    r"\s*(?:is|of|=|:)?\s*"
    r"([-+]?(?:\d+(?:\.\d+)?|\.\d+)%?)",
    flags=re.IGNORECASE,
)


def _clean_output(text):
    if not text:
        return ""
    cleaned = re.sub(
        r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL
    ).strip()
    if re.search(r"(?i)final (?:answer|response)\s*:", cleaned):
        cleaned = re.split(
            r"(?i)final (?:answer|response)\s*:", cleaned
        )[-1].strip()
    return cleaned


def extract_chat_completion(response, default_model="unknown"):
    """Safely extract text and model metadata from an OpenAI-style response."""
    actual_model = getattr(response, "model", None) or default_model
    choices = getattr(response, "choices", None) or []
    if not choices:
        return {
            "actual_model": actual_model,
            "text": "",
            "finish_reason": None,
            "has_reasoning": False,
            "issue": "empty_choices",
        }

    choice = choices[0]
    message = getattr(choice, "message", None)
    finish_reason = getattr(choice, "finish_reason", None)
    if message is None:
        return {
            "actual_model": actual_model,
            "text": "",
            "finish_reason": finish_reason,
            "has_reasoning": False,
            "issue": "missing_message",
        }

    content = getattr(message, "content", None)
    text_parts = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, dict):
                part_text = part.get("text")
            else:
                part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                text_parts.append(part_text)

    text = "\n".join(text_parts).strip()
    reasoning = getattr(message, "reasoning", None)
    has_reasoning = bool(reasoning)
    return {
        "actual_model": actual_model,
        "text": text,
        "finish_reason": finish_reason,
        "has_reasoning": has_reasoning,
        "issue": None if text else "empty_model_content",
    }


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk(nested)


def _numeric_tokens(value):
    tokens = []
    for nested in _walk(value):
        if isinstance(nested, bool) or nested is None:
            continue
        if isinstance(nested, (int, float)):
            tokens.append(float(nested))
        elif isinstance(nested, str):
            for match in NUMBER_PATTERN.findall(nested):
                tokens.append(float(match.rstrip("%")))
    return tokens


def _rule_tokens(value):
    rules = set()
    for nested in _walk(value):
        if isinstance(nested, str):
            rules.update(rule.upper() for rule in RULE_PATTERN.findall(nested))
    return rules


def _canonical_rows(value):
    """Collect unique canonical evidence rows containing item and score."""
    rows = []
    seen = set()
    for nested in _walk(value):
        if (
            isinstance(nested, dict)
            and "item" in nested
            and "score" in nested
        ):
            item = str(nested["item"])
            if item in seen:
                continue
            dominant = nested.get("dominant_rule") or {}
            rows.append(
                {
                    "item": item,
                    "score": float(nested["score"]),
                    "label": nested.get("label"),
                    "rule_id": str(dominant.get("rule_id", "")).upper(),
                }
            )
            seen.add(item)
    return rows


def _number_is_allowed(number, allowed, tolerance=5e-4):
    return any(abs(float(number) - candidate) <= tolerance for candidate in allowed)


def _mentioned_row_segments(rows, text, max_length=360):
    """Return non-overlapping text segments anchored to mentioned items."""
    mentioned = []
    for row in rows:
        match = re.search(re.escape(row["item"]), text, flags=re.IGNORECASE)
        if match:
            mentioned.append(
                {
                    "position": match.start(),
                    "row": row,
                }
            )
    mentioned.sort(key=lambda entry: entry["position"])
    for index, entry in enumerate(mentioned):
        start = entry["position"]
        next_start = (
            mentioned[index + 1]["position"]
            if index + 1 < len(mentioned)
            else len(text)
        )
        end = min(next_start, start + max_length)
        entry["segment"] = text[start:end]
    return mentioned


def _associated_score(segment, item):
    """Extract only a fuzzy score explicitly associated with one item."""
    number = r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))"
    keyword_match = re.search(
        r"\b(?:reported\s+|final\s+)?(?:fuzzy\s+)?scores?"
        r"\s*(?:is|of|=|:)?\s*" + number,
        segment,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        return float(keyword_match.group(1))
    parenthetical_match = re.search(
        r"^"
        + re.escape(item)
        + r"\s*\(\s*(?:score\s*)?"
        + number
        + r"(?=\s*[,;)])",
        segment,
        flags=re.IGNORECASE,
    )
    if parenthetical_match:
        return float(parenthetical_match.group(1))
    return None


def _associated_rules(segment):
    """Return rule identifiers contained in one item-bounded segment."""
    return {
        rule.upper() for rule in RULE_PATTERN.findall(segment)
    }


def grounding_issues(
    text,
    payload=None,
    *,
    require_rule_ids=False,
    require_scores=False,
    require_all_items=False,
    strict_claims=True,
):
    """Return a cleaned draft and machine-readable rejection reasons."""
    cleaned = _clean_output(text)
    issues = []
    if not cleaned:
        return "", ["empty_output"]
    if len(cleaned.split()) < 20:
        issues.append("truncated_or_too_short")

    if any(
        re.search(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in META_OUTPUT_PATTERNS
    ):
        issues.append("reasoning_or_prompt_leakage")

    if strict_claims:
        matched_claims = [
            pattern
            for pattern in UNSUPPORTED_CLAIM_PATTERNS
            if re.search(pattern, cleaned, flags=re.IGNORECASE)
        ]
        if matched_claims:
            issues.append("unsupported_normative_or_outcome_claim")

    if payload is None:
        return cleaned, issues

    allowed_numbers = _numeric_tokens(payload)
    # Validate numbers presented as scientific evidence. Structural numbering,
    # word limits, and stable domain names such as Industry 5.0 are permitted.
    # Item-associated parenthetical scores are added below.
    output_numbers = [
        float(token.rstrip("%"))
        for token in SCIENTIFIC_NUMBER_PATTERN.findall(cleaned)
    ]
    unsupported_numbers = sorted(
        {
            number
            for number in output_numbers
            if not _number_is_allowed(number, allowed_numbers)
        }
    )
    if unsupported_numbers:
        issues.append(
            "unsupported_numbers:" + ",".join(f"{number:g}" for number in unsupported_numbers)
        )

    allowed_rules = _rule_tokens(payload)
    output_rules = {
        rule.upper() for rule in RULE_PATTERN.findall(cleaned)
    }
    unsupported_rules = sorted(output_rules - allowed_rules)
    if unsupported_rules:
        issues.append("unsupported_rule_ids:" + ",".join(unsupported_rules))
    if require_rule_ids and allowed_rules and not output_rules:
        issues.append("missing_rule_ids")

    rows = _canonical_rows(payload)
    # Bind evidence locally: an auxiliary 5S value, baseline value, or another
    # item's score cannot satisfy the score requirement for this item.
    mentioned = _mentioned_row_segments(rows, cleaned)
    mentioned_items = {
        entry["row"]["item"] for entry in mentioned
    }
    all_items_mentioned = bool(rows) and {
        row["item"] for row in rows
    }.issubset(mentioned_items)
    tie_cue_present = bool(
        re.search(
            r"\b(?:all|each|identical|same|tied|equally)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    unique_scores = {
        round(float(row["score"]), 6) for row in rows
    }
    unique_rules = {
        row["rule_id"] for row in rows if row["rule_id"]
    }
    shared_score_is_valid = (
        all_items_mentioned
        and tie_cue_present
        and len(unique_scores) == 1
        and any(
            _number_is_allowed(number, unique_scores)
            for number in output_numbers
        )
    )
    shared_rule_is_valid = (
        all_items_mentioned
        and tie_cue_present
        and len(unique_rules) == 1
        and output_rules == unique_rules
    )
    for entry in mentioned:
        row = entry["row"]
        segment = entry["segment"]
        associated_score = _associated_score(segment, row["item"])
        if associated_score is None:
            if require_scores and not shared_score_is_valid:
                issues.append("missing_item_score:" + row["item"])
        elif not _number_is_allowed(associated_score, [row["score"]]):
            issues.append("item_score_mismatch:" + row["item"])

        associated_rules = _associated_rules(segment)
        if not associated_rules:
            if (
                require_rule_ids
                and row["rule_id"]
                and not shared_rule_is_valid
            ):
                issues.append("missing_item_rule:" + row["item"])
        elif row["rule_id"] and associated_rules != {row["rule_id"]}:
            issues.append("item_rule_mismatch:" + row["item"])

    if require_scores and rows and not mentioned:
        issues.append("missing_scores")
    if require_rule_ids and allowed_rules and not mentioned:
        issues.append("missing_rule_ids")
    if require_all_items and rows:
        for row in rows:
            if row["item"] not in mentioned_items:
                issues.append("missing_item:" + row["item"])

    # Preserve descending score order for the first occurrence of each
    # mentioned canonical item. Different ordering within an exact tie is valid.
    ordered_mentions = [
        (entry["position"], entry["row"]["score"], entry["row"]["item"])
        for entry in mentioned
    ]
    for previous, current in zip(ordered_mentions, ordered_mentions[1:]):
        if current[1] > previous[1] + 5e-4:
            issues.append(
                "ordering_violation:" + previous[2] + ">" + current[2]
            )
            break

    return cleaned, list(dict.fromkeys(issues))


def validate_grounded_output(text, payload=None, **kwargs):
    """Return the cleaned output only when every grounding check passes."""
    cleaned, issues = grounding_issues(text, payload, **kwargs)
    return cleaned if not issues else ""
