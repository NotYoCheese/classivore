#!/usr/bin/env python3
"""Collection orchestrator.

Coordinates URL discovery, content retrieval, quality filtering, and corpus
storage. Features per-taxonomy state, circuit breaker, graceful SIGINT handling,
and automatic search provider fallback.

Flow:
1. Load enriched taxonomy, collection state (per-taxonomy), corpus hashes, domain blocklist
2. Seed category counts from existing labels if available
3. Distribute page targets across leaf categories
4. For each unsatisfied category: generate queries → search → retrieve → filter → save
5. Checkpoint state after each query cycle
"""

import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from classivore.collection.commoncrawl import fetch_warc_record, lookup_cdx
from classivore.collection.domains import DomainTracker
from classivore.collection.filters import content_hash, filter_page, is_url_blocked
from classivore.collection.queries import (
    build_llm_prompt,
    generate_template_queries,
    parse_llm_queries,
)
from classivore.collection.scraper import extract_text, fetch_page
from classivore.collection.search import SearchClient
from classivore.collection.state import CollectionState
from classivore.logging_config import get_logger
from classivore.persistence import append_ndjson, iter_ndjson

logger = get_logger(__name__)

CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_PAUSE = 60

# Global interrupt flag for SIGINT handling
_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning("interrupt_received")


def run_collection(config, categories, data_dir, pages=None, resume=True,
                   queries_only=False, use_llm_queries=False, verbose=False):
    """Run the collection pipeline.

    Args:
        config: TaxonomyConfig instance.
        categories: List of category dicts (from load_taxonomy).
        data_dir: Path to data directory.
        pages: Total pages to collect (distributed across categories).
        resume: Whether to resume from existing state.
        queries_only: If True, generate and log queries without fetching.
        verbose: Enable verbose logging.

    Returns:
        Summary dict with collection statistics.
    """
    global _interrupted
    _interrupted = False

    # Per-taxonomy state directory
    collection_dir = Path(data_dir) / "collection" / config.slug
    # Shared directories
    shared_collection_dir = Path(data_dir) / "collection"
    corpus_dir = Path(data_dir) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state (per-taxonomy) and domain tracker (shared)
    state = CollectionState(collection_dir)
    domains = DomainTracker(shared_collection_dir)

    # Load existing content hashes for dedup
    corpus_file = corpus_dir / "pages.json"
    seen_hashes = _load_existing_hashes(corpus_file)

    # Get leaf categories, excluding configured exclusions
    excluded = set(config.excluded_categories)
    leaf_cats = [
        c for c in categories
        if c["is_leaf"] and c["display_name"] not in excluded
    ]

    # Compute per-category targets
    target = config.target_per_category
    if pages:
        target = max(1, pages // len(leaf_cats)) if leaf_cats else 0

    for cat in leaf_cats:
        state.init_category(cat["name"], target=target)

    # Seed from existing labels if available
    _seed_from_labels(state, data_dir, config.slug)

    # Initialize search client
    search_client = SearchClient.from_config(config)
    if not queries_only and search_client.active_provider_count == 0:
        logger.warning("No search providers configured — search will fail")

    # Install SIGINT handler
    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    # Build category lookup for LLM query context
    cat_by_name = {c["name"]: c for c in categories}

    # Main collection loop
    collected_pages = []
    consecutive_failures = 0

    try:
        for cat in leaf_cats:
            if _interrupted or state.is_satisfied(cat["name"]):
                continue

            tried = set(state.categories[cat["name"]]["queries_tried"])

            # Tier 1: template queries (free)
            queries = generate_template_queries(cat, tried=tried)

            if not queries and use_llm_queries:
                # Tier 2: LLM-generated queries when templates exhausted
                queries = _generate_llm_queries(
                    cat, categories, config, tried, queries_only,
                )

            if not queries:
                logger.info("no_new_queries", category=cat["name"])
                continue

            for query in queries:
                if _interrupted or state.is_satisfied(cat["name"]):
                    break

                if state.has_query(cat["name"], query):
                    continue

                if queries_only:
                    state.record_query(cat["name"], query)
                    logger.info("query_generated", category=cat["name"], query=query)
                    continue

                # Search
                results = search_client.search(query)

                if results is None:
                    # Transient failure — don't record query, increment circuit breaker
                    consecutive_failures += 1
                    state.record_search_error()
                    logger.warning("search_failed", consecutive_failures=consecutive_failures)

                    if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                        logger.warning(
                            "circuit_breaker_tripped",
                            consecutive_failures=consecutive_failures,
                            pause_seconds=CIRCUIT_BREAKER_PAUSE,
                        )
                        _save_checkpoint(state, domains, collected_pages, corpus_file)
                        collected_pages = []
                        time.sleep(CIRCUIT_BREAKER_PAUSE)
                        consecutive_failures = 0
                        search_client.reset_exhausted()
                    continue

                # Search succeeded (even if empty) — record query, reset circuit breaker
                consecutive_failures = 0
                state.record_query(cat["name"], query)

                for result in results:
                    if _interrupted or state.is_satisfied(cat["name"]):
                        break

                    url = result["url"]

                    # Skip known URLs
                    if state.is_url_known(url):
                        continue

                    # URL blocklist check
                    block_reason = is_url_blocked(url)
                    if block_reason:
                        state.record_url(url, cat["name"], "filtered", "search")
                        continue

                    # Domain checks
                    domain = urlparse(url).netloc
                    if domains.is_blocked(domain):
                        state.record_url(url, cat["name"], "filtered", "search")
                        continue

                    if state.get_domain_count(cat["name"], domain) >= config.max_per_domain_per_category:
                        continue

                    # Retrieve content: Common Crawl first, then live scrape
                    page = _retrieve_and_filter(
                        url, cat["name"], config, state, domains, seen_hashes,
                    )

                    if page:
                        collected_pages.append(page)

                # Checkpoint after each query cycle
                _save_checkpoint(state, domains, collected_pages, corpus_file)
                collected_pages = []

    finally:
        # Always save on exit (normal, interrupt, or exception)
        _save_checkpoint(state, domains, collected_pages, corpus_file)
        state.save()
        domains.save()
        signal.signal(signal.SIGINT, prev_handler)

    return state.summary()


def _generate_llm_queries(cat, categories, config, tried, queries_only):
    """Generate LLM queries for a category when templates are exhausted.

    Uses the Anthropic API (non-batch, single call) to generate creative
    queries informed by category context and previously tried queries.

    Args:
        cat: Category dict.
        categories: Full list of category dicts (for sibling lookup).
        config: TaxonomyConfig instance.
        tried: Set of already-tried query strings.
        queries_only: If True, skip API call and return empty.

    Returns:
        List of new query strings, or empty list on failure.
    """
    if queries_only:
        return []

    # Find siblings (same parent)
    parent_id = cat.get("parent_id", "")
    siblings = [
        c["name"] for c in categories
        if c.get("parent_id") == parent_id and c["name"] != cat["name"]
    ]

    # Get domain hints for this category's tier-1
    tier1 = cat["path"][0] if cat.get("path") else cat["name"]
    domain_hints = config.domain_hints.get(tier1, []) if hasattr(config, "domain_hints") else []

    # Compute pages still needed
    pages_needed = None
    if hasattr(config, "target_per_category"):
        pages_needed = config.target_per_category

    messages = build_llm_prompt(
        cat,
        siblings=siblings,
        tried_queries=sorted(tried),
        domain_hints=domain_hints or None,
        pages_needed=pages_needed,
    )

    try:
        from classivore.batch import get_api_client
        client = get_api_client()
        response = client.messages.create(
            model=config.query_model,
            max_tokens=300,
            temperature=0.7,
            system=("You generate search queries to find high-quality articles "
                    "for a content taxonomy. Each query should find articles, "
                    "guides, analyses, or expert content — NOT product listings, "
                    "marketplaces, or shopping pages. Return exactly 5 queries, "
                    "one per line, numbered 1-5. No commentary."),
            messages=messages,
        )

        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        queries = parse_llm_queries(text)
        new_queries = [q for q in queries if q not in tried]

        if new_queries:
            logger.info(
                "llm_queries_generated",
                category=cat["name"],
                count=len(new_queries),
            )

        return new_queries

    except Exception as e:
        logger.warning("llm_query_generation_failed", category=cat["name"], error=str(e))
        return []


def _seed_from_labels(state, data_dir, taxonomy_slug):
    """Seed category collected counts from existing labels.

    If labels exist for this taxonomy, count labeled pages per category
    and update state counts (only if greater than current).
    """
    labels_file = Path(data_dir) / "labels" / taxonomy_slug / "labels.json"
    if not labels_file.exists():
        return

    try:
        label_counts = {}
        for entry in iter_ndjson(labels_file):
            for cat_name in entry.get("categories", []):
                label_counts[cat_name] = label_counts.get(cat_name, 0) + 1

        for name, count in label_counts.items():
            cat = state.categories.get(name)
            if cat and count > cat["collected"]:
                cat["collected"] = count
                logger.info("seeded_from_labels", category=name, count=count)

    except Exception as e:
        logger.warning("label_seeding_failed", error=str(e))


def _retrieve_and_filter(url, category, config, state, domains, seen_hashes):
    """Try to retrieve and filter a single URL. Returns page dict or None."""
    html = None
    source = "commoncrawl"

    # Try Common Crawl first
    if config.commoncrawl_crawl_id:
        records = lookup_cdx(url, crawl_id=config.commoncrawl_crawl_id)
        if records:
            html = fetch_warc_record(records[0])

    # Fallback to live scrape
    if not html:
        source = "live_scrape"
        html = fetch_page(url)

    if not html:
        state.record_url(url, category, "failed", source)
        domains.record_result(urlparse(url).netloc, success=False)
        return None

    # Extract text
    text = extract_text(html)
    if not text:
        state.record_url(url, category, "failed", source)
        domains.record_result(urlparse(url).netloc, success=False)
        return None

    # Content filter
    filtered_text, reason = filter_page(text)
    if not filtered_text:
        state.record_url(url, category, "filtered", source)
        domains.record_result(urlparse(url).netloc, success=False)
        return None

    # Dedup
    text_hash = content_hash(filtered_text)
    if text_hash in seen_hashes:
        state.record_url(url, category, "duplicate", source)
        return None

    seen_hashes.add(text_hash)
    state.record_url(url, category, "collected", source)
    domains.record_result(urlparse(url).netloc, success=True)

    return {
        "url": url,
        "text": filtered_text,
        "word_count": len(filtered_text.split()),
        "source": source,
        "category": category,
        "content_hash": text_hash,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_existing_hashes(corpus_file):
    """Load content hashes from existing corpus for dedup."""
    hashes = set()
    if not corpus_file.exists():
        return hashes

    try:
        for page in iter_ndjson(corpus_file):
            if "content_hash" in page:
                hashes.add(page["content_hash"])
    except Exception as e:
        logger.warning("corpus_load_failed", error=str(e))

    return hashes


def _save_checkpoint(state, domains, pages, corpus_file):
    """Save state, domain scores, and append new pages to corpus."""
    append_ndjson(corpus_file, pages)
    state.save()
    domains.save()


def audit_domains(data_dir):
    """Generate domain quality audit report."""
    collection_dir = Path(data_dir) / "collection"
    tracker = DomainTracker(collection_dir)
    return tracker.audit_report()
