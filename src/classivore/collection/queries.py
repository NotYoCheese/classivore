#!/usr/bin/env python3
"""Search query generation for content discovery.

Generates search queries in two tiers:
1. Template queries (free) — uses category name, description keywords, and
   tier-1 ancestor to construct 3 diverse queries per category.
2. LLM queries (batch API) — when templates are exhausted, uses Haiku to
   generate 5 creative queries per category, informed by description,
   boundaries, siblings, and previously tried queries.
"""

import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with", "not", "but", "they", "their", "which", "who",
    "will", "would", "can", "could", "do", "does", "did", "had", "have",
    "may", "might", "shall", "should", "about", "also", "been", "being",
    "between", "both", "each", "into", "more", "most", "other", "over",
    "such", "than", "them", "then", "these", "those", "through", "under",
    "very", "what", "when", "where",
}

QUERY_SYSTEM_PROMPT = """You generate search queries to find high-quality articles for a content taxonomy.
Each query should find articles, guides, analyses, or expert content — NOT product listings, marketplaces, or shopping pages.
Return exactly 5 queries, one per line, numbered 1-5. No commentary."""


def _extract_keywords(description, max_words=4):
    """Extract meaningful keywords from a description, skipping stopwords."""
    if not description:
        return []
    words = re.findall(r"[a-zA-Z]+", description.lower())
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:max_words]


def generate_template_queries(category, tried=None):
    """Generate template-based search queries for a category.

    Uses category name, description keywords, and tier-1 ancestor.
    Free — no API calls needed.

    Args:
        category: Category dict with name, description, path.
        tried: Set of previously tried query strings to exclude.

    Returns:
        List of query strings not in tried set.
    """
    tried = tried or set()
    name = category["name"]
    description = category.get("description", "")
    path = category.get("path", [name])
    tier1 = path[0] if path else name

    keywords = _extract_keywords(description)
    keyword_str = " ".join(keywords) if keywords else name.lower()

    queries = [
        f'"{name}" article {tier1}',
        f"{keyword_str} guide",
        f"{keyword_str} analysis 2026",
    ]

    # Add a tier-1 scoped query if category is deeper than tier 1
    if len(path) > 1:
        queries.append(f"{name} {tier1} explained")

    return [q for q in queries if q not in tried]


def build_llm_prompt(category, siblings, tried_queries):
    """Build a prompt for LLM query generation.

    Args:
        category: Category dict.
        siblings: List of sibling category names.
        tried_queries: List of already-tried query strings.

    Returns:
        List of message dicts for the Anthropic API.
    """
    name = category["name"]
    description = category.get("description", "")
    boundaries = category.get("boundaries", "")
    path = " > ".join(category.get("path", [name]))

    siblings_str = ", ".join(siblings) if siblings else "None"
    tried_str = "\n".join(f"- {q}" for q in tried_queries) if tried_queries else "None yet"

    content = f"""Generate 5 diverse search queries to find high-quality articles about this taxonomy category:

Category: {name}
Full path: {path}
Description: {description}
Boundaries: {boundaries}
Sibling categories (avoid overlap): {siblings_str}

Previously tried queries (generate different ones):
{tried_str}

Requirements:
- Queries should find articles, analyses, guides, and expert content
- Avoid queries that would return product listings, shopping pages, or marketplaces
- Make queries specific enough to match this category, not its siblings
- Include a mix of query styles: quoted phrases, natural language, topic + format"""

    return [{"role": "user", "content": content}]


def parse_llm_queries(text):
    """Parse LLM response into a list of query strings.

    Handles numbered lists, dashed lists, and plain lines.
    Strips quotes, numbering, and preamble.

    Args:
        text: Raw LLM response text.

    Returns:
        List of query strings.
    """
    if not text or not text.strip():
        return []

    queries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strip numbering: "1. ", "1) "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Strip dashes: "- "
        line = re.sub(r"^[-•]\s*", "", line)
        # Strip surrounding quotes
        line = line.strip('"\'')
        line = line.strip()

        if not line:
            continue

        # Skip preamble lines (ends with colon, or long prose without query keywords)
        if line.endswith(":"):
            continue
        if len(line.split()) > 8 and not any(kw in line.lower() for kw in ["guide", "article", "review", "analysis", "how to"]):
            continue

        queries.append(line)

    return queries
