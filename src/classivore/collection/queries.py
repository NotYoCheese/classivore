#!/usr/bin/env python3
"""Search query generation for content discovery.

Generates search queries in two tiers:
1. Template queries (free) — uses category name, description, boundaries,
   and tier-1 ancestor to construct diverse queries across multiple intent types.
2. LLM queries (batch API) — when templates are exhausted, uses Haiku to
   generate creative queries informed by category context and tried queries.
"""

import re
from datetime import datetime, timezone

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

# Template patterns organized by search intent.
# {name} = category name, {scope} = immediate parent (or tier1 for depth ≤ 2),
# {year} = current year, {kw} = keywords (aliases if available, else description),
# {bkw} = keywords from boundaries.
QUERY_TEMPLATES = [
    # --- Informational ---
    '"{name}" article {scope}',
    "what is {name}",
    "{name} explained",
    "understanding {name}",
    "{name} overview",
    "{name} for beginners",
    "introduction to {name}",
    "history of {name}",
    # --- Editorial / In-depth ---
    '"{name}" feature article',
    "{name} in depth {year}",
    "{name} history and origins",
    "{name} scene explained",
    "{kw} journalism {year}",
    '"{kw}" guide {year}',
    "{kw} documentary",
    "{name} culture and community",
    "{scope} {name} overview",
    # --- How-to / Practical ---
    "{name} tips and advice",
    "{name} best practices",
    "{name} guide {year}",
    "getting started with {name}",
    # --- News / Trends ---
    "{name} trends {year}",
    "{name} news {year}",
    "future of {name}",
    "{name} industry report {year}",
    "{name} statistics {year}",
    # --- Definitional / Authoritative ---
    "{name} definition",
    "types of {name}",
    "{name} categories explained",
    # --- Long-tail / Specific ---
    "{name} case studies",
    "{name} examples",
    "{name} research",
    # --- Description keyword variants ---
    "{kw} guide",
    "{kw} analysis {year}",
    "{kw} explained",
    "{kw} {scope}",
    # --- Boundary keyword variants ---
    "{bkw} article",
    "{bkw} {name}",
    # --- Scope-level context (deeper categories only) ---
    "{name} {scope} explained",
    "{name} in {scope}",
]

QUERY_SYSTEM_PROMPT = """You generate search queries to find high-quality articles for a content taxonomy.
Each query should find articles, guides, analyses, or expert content — NOT product listings, marketplaces, or shopping pages.
Return exactly 5 queries, one per line, numbered 1-5. No commentary."""


def _extract_keywords(text, max_words=4):
    """Extract meaningful keywords from text, skipping stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:max_words]


def generate_template_queries(category, tried=None):
    """Generate template-based search queries for a category.

    Uses category name, description, boundaries, and tier-1 ancestor
    to construct diverse queries across multiple intent types.
    Free — no API calls needed.

    Args:
        category: Category dict with name, description, boundaries, path.
        tried: Set of previously tried query strings to exclude.

    Returns:
        List of query strings not in tried set.
    """
    tried = tried or set()
    name = category["name"]
    description = category.get("description", "")
    boundaries = category.get("boundaries", "")
    path = category.get("path", [name])
    tier1 = path[0] if path else name
    scope = path[-2] if len(path) >= 3 else tier1
    year = str(datetime.now(timezone.utc).year)

    aliases = category.get("aliases", [])
    if aliases:
        kw_str = " ".join(aliases[:3])
        boundary_keywords = _extract_keywords(boundaries)
        bkw_str = " ".join(boundary_keywords) if boundary_keywords else ""
    else:
        desc_keywords = _extract_keywords(description)
        boundary_keywords = _extract_keywords(boundaries)
        kw_str = " ".join(desc_keywords) if desc_keywords else name.lower()
        bkw_str = " ".join(boundary_keywords) if boundary_keywords else ""

    is_deep = len(path) > 1

    queries = []
    seen = set()

    for template in QUERY_TEMPLATES:
        # Skip scope-level templates for root categories
        if not is_deep and "{scope}" in template and "{name}" in template:
            if template in ("{name} {scope} explained", "{name} in {scope}"):
                continue

        # Skip boundary keyword templates if no boundary text
        if "{bkw}" in template and not bkw_str:
            continue

        query = (
            template
            .replace("{name}", name)
            .replace("{scope}", scope)
            .replace("{year}", year)
            .replace("{kw}", kw_str)
            .replace("{bkw}", bkw_str)
        )

        # Deduplicate within this generation
        normalized = query.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        if query not in tried:
            queries.append(query)

    return queries


def build_llm_prompt(category, siblings, tried_queries, domain_hints=None,
                     pages_needed=None, aliases=None):
    """Build a prompt for LLM query generation.

    Args:
        category: Category dict.
        siblings: List of sibling category names.
        tried_queries: List of already-tried query strings.
        domain_hints: Optional list of known-good domains for this category's tier-1.
        pages_needed: Optional number of additional pages needed.
        aliases: Optional list of alias strings for this category.

    Returns:
        List of message dicts for the Anthropic API.
    """
    name = category["name"]
    description = category.get("description", "")
    boundaries = category.get("boundaries", "")
    path = " > ".join(category.get("path", [name]))

    siblings_str = ", ".join(siblings) if siblings else "None"
    tried_str = "\n".join(f"- {q}" for q in tried_queries) if tried_queries else "None yet"

    domain_section = ""
    if domain_hints:
        domain_list = ", ".join(domain_hints)
        domain_section = f"\nKnown good domains for this area: {domain_list}\nConsider site-scoped queries (e.g. 'topic site:domain.com') for 1-2 queries.\n"

    pages_section = ""
    if pages_needed:
        pages_section = f"\nWe need approximately {pages_needed} more articles for this category.\n"

    aliases_section = ""
    if aliases:
        aliases_section = f"\nKnown aliases and related terms: {', '.join(aliases)}"
        aliases_section += "\nInclude some of these terms in your queries where natural.\n"

    content = f"""Generate 5 diverse search queries to find high-quality articles about this taxonomy category:

Category: {name}
Full path: {path}
Description: {description}
Boundaries: {boundaries}
Sibling categories (avoid overlap): {siblings_str}
{domain_section}{pages_section}{aliases_section}
Previously tried queries (generate DIFFERENT ones — vary vocabulary, angle, and specificity):
{tried_str}

Requirements:
- Queries should find articles, analyses, guides, and expert content
- Avoid queries that would return product listings, shopping pages, or marketplaces
- Make queries specific enough to match this category, not its siblings
- Use specific terminology and aliases provided above, not just the category name
- Include a mix of query styles: quoted phrases, natural language, topic + format
- Try angles the previous queries missed: different terminology, adjacent concepts, specific subtopics"""

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
