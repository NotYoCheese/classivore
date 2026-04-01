#!/usr/bin/env python3
"""Generate taxonomy descriptions and boundaries using Claude via batch API."""

import json as _json

from classivore.taxonomy.loader import build_hierarchy, get_children, get_siblings

SYSTEM_PROMPT = """\
You are a taxonomy expert. For each category, produce a JSON object with exactly two keys:

- "description": One sentence defining what content belongs in this category.
- "boundary": One sentence explaining how this category is distinguished from \
its sibling categories.

Respond with only the JSON object. No markdown, no commentary."""


def build_prompt(category, siblings, children):
    """Build the user message for a single category enrichment request.

    Args:
        category: A category dict.
        siblings: List of sibling category names.
        children: List of child category names.

    Returns:
        List with a single user message dict.
    """
    parts = [
        f"Category: {category['display_name']}",
        f"Full path: {' > '.join(category['path'])}",
    ]

    if siblings:
        parts.append(f"Sibling categories: {', '.join(siblings)}")
    else:
        parts.append("Sibling categories: None (only child or root)")

    if children:
        parts.append(f"Child categories: {', '.join(children)}")
    else:
        parts.append("Child categories: None (leaf node)")

    return [{"role": "user", "content": "\n".join(parts)}]


def build_batch_requests(categories, hierarchy, config):
    """Build batch API requests for categories that need enrichment.

    Skips categories that already have a description (supports resume).

    Args:
        categories: List of category dicts.
        hierarchy: Dict from build_hierarchy.
        config: TaxonomyConfig with model and token settings.

    Returns:
        List of batch request dicts.
    """
    requests = []
    for cat in categories:
        if cat["description"]:
            continue

        siblings = get_siblings(cat, hierarchy)
        children = get_children(cat, hierarchy)
        messages = build_prompt(cat, siblings, children)

        requests.append({
            "custom_id": f"cat-{cat['id']}",
            "params": {
                "model": config.enrichment_model,
                "max_tokens": config.enrichment_max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            },
        })

    return requests


def parse_enrichment(message):
    """Extract description and boundary from an API response message.

    Expects a JSON object with "description" and "boundary" keys.
    Falls back to line-based parsing if JSON parsing fails.

    Args:
        message: An Anthropic Message object.

    Returns:
        Tuple of (description, boundaries).
    """
    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text += block.text

    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines_raw = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines_raw = [l for l in lines_raw if not l.strip().startswith("```")]
        text = "\n".join(lines_raw).strip()

    # Try JSON parsing first
    try:
        data = _json.loads(text)
        return (
            data.get("description", "").strip(),
            data.get("boundary", "").strip(),
        )
    except (_json.JSONDecodeError, AttributeError):
        pass

    # Fallback: line-based parsing
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if not line.startswith("#")]

    if len(lines) >= 2:
        return (lines[0], lines[1])
    elif len(lines) == 1:
        return (lines[0], "")
    else:
        return ("", "")


def apply_results(categories, results):
    """Apply enrichment results to category dicts.

    Args:
        categories: List of category dicts (modified in place).
        results: Dict mapping category_id to (description, boundaries) tuples.
    """
    for cat in categories:
        if cat["id"] in results:
            desc, bounds = results[cat["id"]]
            cat["description"] = desc
            cat["boundaries"] = bounds
