#!/usr/bin/env python3
"""Collection orchestrator.

Coordinates URL discovery, content retrieval, quality filtering, and corpus
storage. Manages per-category targets, query cycling, and checkpointing.

Flow:
1. Load enriched taxonomy, collection state, existing corpus hashes, domain blocklist
2. Distribute page targets across leaf categories
3. For each unsatisfied category: generate queries → search → retrieve → filter → save
4. Checkpoint state after each query cycle
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from classivore.collection.commoncrawl import fetch_warc_record, lookup_cdx
from classivore.collection.domains import DomainTracker
from classivore.collection.filters import content_hash, filter_page, is_url_blocked
from classivore.collection.queries import generate_template_queries
from classivore.collection.scraper import extract_text, fetch_page
from classivore.collection.search import search_brave
from classivore.collection.state import CollectionState

logger = logging.getLogger(__name__)


def run_collection(config, categories, data_dir, pages=None, resume=True,
                   queries_only=False, verbose=False):
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
    if verbose:
        logging.basicConfig(level=logging.INFO)

    collection_dir = Path(data_dir) / "collection"
    corpus_dir = Path(data_dir) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state and domain tracker
    state = CollectionState(collection_dir)
    domains = DomainTracker(collection_dir)

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

    # Get Brave API key
    brave_api_key = os.getenv("BRAVE_API_KEY", "")
    if not brave_api_key and not queries_only:
        logger.warning("BRAVE_API_KEY not set — search will fail")

    # Main collection loop
    collected_pages = []
    for cat in leaf_cats:
        if state.is_satisfied(cat["name"]):
            continue

        # Generate template queries
        tried = set(state.categories[cat["name"]]["queries_tried"])
        queries = generate_template_queries(cat, tried=tried)

        if not queries:
            logger.info("No new queries for %s", cat["name"])
            continue

        for query in queries:
            if state.is_satisfied(cat["name"]):
                break

            if state.has_query(cat["name"], query):
                continue

            state.record_query(cat["name"], query)

            if queries_only:
                logger.info("Query [%s]: %s", cat["name"], query)
                continue

            # Search
            results = search_brave(query, api_key=brave_api_key)

            for result in results:
                if state.is_satisfied(cat["name"]):
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

    # Final save
    _save_checkpoint(state, domains, collected_pages, corpus_file)
    state.save()
    domains.save()

    return state.summary()


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
        with open(corpus_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    page = json.loads(line)
                    if "content_hash" in page:
                        hashes.add(page["content_hash"])
    except Exception as e:
        logger.warning("Failed to load existing corpus: %s", e)

    return hashes


def _save_checkpoint(state, domains, pages, corpus_file):
    """Save state, domain scores, and append new pages to corpus."""
    if pages:
        with open(corpus_file, "a") as f:
            for page in pages:
                f.write(json.dumps(page) + "\n")

    state.save()
    domains.save()


def audit_domains(data_dir):
    """Generate domain quality audit report."""
    collection_dir = Path(data_dir) / "collection"
    tracker = DomainTracker(collection_dir)
    return tracker.audit_report()
