#!/usr/bin/env python3
"""Generate taxonomy descriptions and boundaries using Claude via batch API."""

import json as _json

from classivore.logging_config import get_logger
from classivore.taxonomy.loader import build_hierarchy, get_children, get_siblings

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a taxonomy expert. For each category, produce a JSON object with exactly four keys:

- "description": One sentence defining what content belongs in this category.
- "boundary": One sentence explaining how this category is distinguished from \
its sibling categories.
- "aliases": A list of 4-8 strings — alternate names, colloquial terms, \
subgenre/subtopic names, and representative specific examples the web actually \
uses for this category. These should be terms that appear in URLs and page \
content, not abstract synonyms.
- "difficulty": One of "easy", "medium", or "hard". Reflects how much \
high-quality editorial web content exists for this category. Easy = abundant \
editorial coverage (broad consumer topics). Hard = niche or dominated by \
product listings/thin content. Depth alone does not determine difficulty.

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
        if cat["description"] and cat.get("aliases"):
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
    """Extract description, boundary, aliases, and difficulty from an API response.

    Expects a JSON object with "description", "boundary", "aliases", and
    "difficulty" keys. Falls back to line-based parsing if JSON parsing fails.

    Args:
        message: An Anthropic Message object.

    Returns:
        Tuple of (description, boundaries, aliases, difficulty).
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

    valid_difficulties = {"easy", "medium", "hard"}

    # Try JSON parsing first
    try:
        data = _json.loads(text)
        aliases = data.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        difficulty = data.get("difficulty", "medium")
        if difficulty not in valid_difficulties:
            difficulty = "medium"
        return (
            data.get("description", "").strip(),
            data.get("boundary", "").strip(),
            aliases,
            difficulty,
        )
    except (_json.JSONDecodeError, AttributeError) as e:
        logger.debug("enricher_json_parse_failed_falling_back_to_lines", error=str(e))

    # Fallback: line-based parsing
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if not line.startswith("#")]

    if len(lines) >= 2:
        return (lines[0], lines[1], [], "medium")
    elif len(lines) == 1:
        return (lines[0], "", [], "medium")
    else:
        return ("", "", [], "medium")


HINTS_SYSTEM_PROMPT = (
    "You are a web research expert. Return only a JSON array of "
    "domain names, nothing else. No markdown, no commentary."
)


def generate_domain_hints(tier1_names, config):
    """Generate domain hints for tier1 categories via direct API calls.

    Makes one API call per tier1 category. Not batched — the list is short
    (typically 20-30 tier1 categories) and we want results immediately.

    Args:
        tier1_names: List of tier1 category name strings.
        config: TaxonomyConfig with enrichment_model setting.

    Returns:
        Dict mapping tier1_name → list of domain strings.
    """
    from classivore.batch import get_api_client

    client = get_api_client()
    results = {}

    for name in tier1_names:
        user_msg = (
            f"List 8 authoritative websites that publish high-quality editorial "
            f"articles, features, guides, and analysis about: {name}\n\n"
            f"Focus on: journalism, expert editorial content, original reporting.\n"
            f"Exclude: Wikipedia, aggregators, social media, e-commerce sites, "
            f"streaming services, government sites.\n\n"
            f'Return exactly a JSON array of bare domain names like:\n'
            f'["example.com", "another.com"]'
        )

        try:
            response = client.messages.create(
                model=config.enrichment_model,
                max_tokens=200,
                system=HINTS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

            # Strip code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            domains = _json.loads(text)
            if not isinstance(domains, list):
                domains = []

            # Clean: strip www. prefix, deduplicate
            cleaned = []
            seen = set()
            for d in domains:
                if not isinstance(d, str):
                    continue
                d = d.removeprefix("www.").strip().lower()
                if d and d not in seen:
                    seen.add(d)
                    cleaned.append(d)

            results[name] = cleaned
            logger.info("hints_generated", tier1=name, count=len(cleaned))

        except Exception as e:
            logger.warning("hints_generation_failed", tier1=name, error=str(e))
            results[name] = []

    return results


def apply_results(categories, results):
    """Apply enrichment results to category dicts.

    Args:
        categories: List of category dicts (modified in place).
        results: Dict mapping category_id to (description, boundaries, aliases, difficulty) tuples.
    """
    for cat in categories:
        if cat["id"] in results:
            desc, bounds, aliases, difficulty = results[cat["id"]]
            cat["description"] = desc
            cat["boundaries"] = bounds
            cat["aliases"] = aliases
            cat["difficulty"] = difficulty
