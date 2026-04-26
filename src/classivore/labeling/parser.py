#!/usr/bin/env python3
"""Response parsing and category validation for labeling.

Handles JSON extraction from Anthropic API responses, code fence stripping,
category name validation, and confidence/label count filtering.
"""

import json
import re

from classivore.logging_config import get_logger

logger = get_logger(__name__)


def _extract_text(message):
    """Extract and concatenate text from Anthropic message content blocks."""
    parts = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)


def _strip_code_fences(text):
    """Strip markdown code fences from response text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json(text):
    """Parse JSON from response text, stripping code fences."""
    text = _strip_code_fences(text)
    return json.loads(text)


def parse_stage1_response(message):
    """Parse stage 1 (tier-1 triage) response.

    Expected format: {"categories": [{"name": "...", "confidence": 0.85}]}
    Fallback: if top-level is a list, treat as categories directly.

    Returns:
        List of {"name", "confidence"} dicts. Empty list on parse failure.
    """
    try:
        text = _extract_text(message)
        data = _parse_json(text)

        if isinstance(data, list):
            return data

        return data.get("categories", [])
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning("stage1_parse_error", error=str(e))
        # Attempt to salvage truncated JSON
        try:
            text = _extract_text(message)
            return _salvage_truncated_categories(text)
        except Exception as salvage_err:
            logger.debug("stage1_salvage_failed", error=str(salvage_err))
            return []


def _salvage_truncated_categories(text):
    """Extract valid category entries from truncated JSON.

    When max_tokens cuts off the response mid-JSON, extract any complete
    {"name": "...", "confidence": ...} objects that appear before the truncation.
    """
    results = []
    for match in re.finditer(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([\d.]+)\s*\}',
        text,
    ):
        name = match.group(1)
        try:
            confidence = float(match.group(2))
        except ValueError as e:
            logger.debug("salvage_skip_bad_confidence", name=name, raw=match.group(2), error=str(e))
            continue
        results.append({"name": name, "confidence": confidence})

    if results:
        logger.info("salvaged_truncated_json", category_count=len(results))
    return results


def parse_stage2_response(message, valid_names, min_confidence=0.5, max_labels=3):
    """Parse stage 2 (subtree classification) response.

    Expected: {"reasoning": "...", "categories": [{"name": "...", "confidence": 0.92}]}

    Validates category names, filters by confidence, enforces max_labels.

    Returns:
        Dict with "reasoning", "categories" (validated), and optionally "error".
    """
    try:
        text = _extract_text(message)
        data = _parse_json(text)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning("stage2_parse_error", error=str(e))
        return {"reasoning": "", "categories": [], "error": str(e)}

    reasoning = data.get("reasoning", "")
    raw_categories = data.get("categories", [])

    # Validate and filter
    validated = []
    for cat in raw_categories:
        name = cat.get("name", "")
        confidence = cat.get("confidence", 0.0)

        canonical = validate_category_name(name, valid_names)
        if canonical is None:
            logger.info("dropping_invalid_category", name=name)
            continue

        if confidence < min_confidence:
            continue

        validated.append({"name": canonical, "confidence": confidence})

    # Sort by confidence descending, enforce max_labels
    validated.sort(key=lambda c: c["confidence"], reverse=True)
    validated = validated[:max_labels]

    return {"reasoning": reasoning, "categories": validated}


def validate_category_name(name, valid_names):
    """Validate a category name against the set of valid names.

    Tries exact match, case-insensitive match, and display_name extraction.

    Args:
        name: Category name from LLM response.
        valid_names: Set of valid canonical category names.

    Returns:
        Canonical name if matched, or None.
    """
    if not name:
        return None

    # Exact match
    if name in valid_names:
        return name

    # Case-insensitive match
    name_lower = name.lower()
    for valid in valid_names:
        if valid.lower() == name_lower:
            return valid

    # Display name format: "Tier1: ActualName" → try extracting last part
    if ": " in name:
        last_part = name.split(": ")[-1].strip()
        if last_part in valid_names:
            return last_part
        # Case-insensitive on extracted part
        last_lower = last_part.lower()
        for valid in valid_names:
            if valid.lower() == last_lower:
                return valid

    return None
