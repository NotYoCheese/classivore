#!/usr/bin/env python3
"""Labeling orchestrator.

Two-stage hierarchical LLM labeling via Anthropic Batch API:
1. Stage 1: Tier-1 triage — identify relevant top-level categories
2. Stage 2: Subtree classification — assign specific categories from selected subtrees

Features crash recovery via persistent label state, raw result backups,
and batch ID tracking for resume.
"""

import json
from pathlib import Path

from classivore.batch import (
    aggregate_batch_usage,
    get_api_client,
    iter_succeeded_results,
    poll_until_complete,
    submit_batch,
)
from classivore.labeling.parser import parse_stage1_response, parse_stage2_response
from classivore.labeling.prompts import (
    build_stage1_system,
    build_stage2_system,
    build_user_message,
    get_subtree_categories,
)
from classivore.labeling.state import LabelState
from classivore.logging_config import get_logger
from classivore.persistence import iter_ndjson, load_ndjson

logger = get_logger(__name__)

BATCH_CHUNK_SIZE = 10000

# 1-hour TTL keeps the system prompt cached across slow-processing batches.
# Batches routinely take >5 minutes, so the default 5m TTL would expire
# mid-run and most requests would miss cache.
_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}


def _cacheable_system(system_prompt):
    """Wrap a system prompt string in a cache-controlled content block."""
    return [{"type": "text", "text": system_prompt, "cache_control": _CACHE_CONTROL_1H}]


def _empty_usage_totals():
    return {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_cache_read_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _empty_funnel():
    return {
        "stage1_sent": 0,
        "tier1_hits": 0,
        "stage2_sent": 0,
        "labels_emitted": 0,
        "errors": 0,
    }


def _accumulate_usage(totals, stats):
    for key in (
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_creation_tokens",
        "total_cache_read_tokens",
        "estimated_cost_usd",
    ):
        totals[key] += stats.get(key, 0)


def _finalize_usage(totals):
    total_input = (
        totals["total_input_tokens"]
        + totals["total_cache_creation_tokens"]
        + totals["total_cache_read_tokens"]
    )
    totals["cache_hit_rate"] = (
        totals["total_cache_read_tokens"] / total_input if total_input > 0 else 0.0
    )
    return totals


def run_labeling(config, categories, hierarchy, data_dir, stage="all",
                 dry_run=False, poll_interval=30, verbose=False):
    """Run the two-stage labeling pipeline.

    Args:
        config: TaxonomyConfig instance.
        categories: List of category dicts (from load_taxonomy).
        hierarchy: Hierarchy dict (from build_hierarchy).
        data_dir: Path to data directory.
        stage: Which stage to run ("1", "2", or "all").
        dry_run: If True, show what would be labeled without API calls.
        poll_interval: Seconds between batch status polls.
        verbose: Enable verbose logging.

    Returns:
        Summary dict with labeling statistics.
    """
    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels" / config.slug
    corpus_file = data_dir / "corpus" / "pages.json"

    # Load state
    label_state = LabelState(labels_dir)

    # Load corpus pages
    pages = _load_corpus(corpus_file)
    if not pages:
        logger.warning("no_corpus_pages", path=str(corpus_file))
        return label_state.summary()

    # Seed state from existing labels (handles legacy imports that bypassed LabelState)
    _seed_from_existing_labels(label_state, labels_dir)

    # Register all pages in state
    for page in pages:
        content_hash = page.get("content_hash", "")
        url = page.get("url", "")
        if content_hash:
            label_state.init_page(content_hash, url)
    label_state.save()

    # Filter taxonomy to content categories (exclude metadata tier-1s)
    excluded_tier1 = set(config.excluded_tier1_categories)
    content_categories = [
        c for c in categories
        if c["path"][0] not in excluded_tier1
    ]
    tier1_categories = [c for c in content_categories if c["depth"] == 1]

    if dry_run:
        needing_s1 = len(label_state.pages_needing_stage1())
        needing_s2 = len(label_state.pages_needing_stage2())
        print(f"Corpus pages: {len(pages)}")
        print(f"Content categories: {len(content_categories)} (across {len(tier1_categories)} tier-1)")
        print(f"Needing stage 1: {needing_s1}")
        print(f"Needing stage 2: {needing_s2}")
        print(f"Model: {config.labeling_model}")
        print("Dry run — no API calls made.")
        return label_state.summary()

    # Build page lookup by content_hash
    page_lookup = {p["content_hash"]: p for p in pages if p.get("content_hash")}

    # Get API client
    client = get_api_client()

    stage1_usage = None
    stage2_usage = None
    funnel = _empty_funnel()

    # Stage 1
    if stage in ("all", "1"):
        stage1_usage = _run_stage1(
            client, label_state, page_lookup, tier1_categories,
            config, labels_dir, poll_interval, funnel,
        )

    # Stage 2
    if stage in ("all", "2"):
        stage2_usage = _run_stage2(
            client, label_state, page_lookup, content_categories,
            config, labels_dir, poll_interval, funnel,
        )

    # Write output
    _write_labels(label_state, labels_dir)

    _print_usage_summary(stage1_usage, stage2_usage)

    summary = label_state.summary()
    if stage1_usage:
        summary["stage1_usage"] = stage1_usage
    if stage2_usage:
        summary["stage2_usage"] = stage2_usage
    summary["labeling_metrics"] = {
        **funnel,
        "cache": {
            **({"stage1": stage1_usage} if stage1_usage else {}),
            **({"stage2": stage2_usage} if stage2_usage else {}),
        },
    }
    return summary


def _run_stage1(client, state, page_lookup, tier1_categories, config, labels_dir, poll_interval, funnel):
    """Run stage 1: tier-1 triage. Returns aggregated usage stats dict.

    Mutates `funnel` with stage1_sent / tier1_hits counters.
    """
    usage_totals = _empty_usage_totals()
    needing = state.pages_needing_stage1()
    if not needing:
        logger.info("stage1_all_triaged")
        return _finalize_usage(usage_totals)

    logger.info("stage1_triaging", page_count=len(needing), tier1_count=len(tier1_categories))

    system_prompt = build_stage1_system(tier1_categories)
    system_blocks = _cacheable_system(system_prompt)
    valid_tier1_names = {c["name"] for c in tier1_categories}

    # Build batch requests
    requests = []
    for content_hash in needing:
        page = page_lookup.get(content_hash)
        if not page:
            continue
        user_msg = build_user_message(
            page.get("text", ""),
            url=page.get("url", ""),
            truncation_words=config.text_truncation_words,
        )
        requests.append({
            "custom_id": f"s1-{content_hash}",
            "params": {
                "model": config.labeling_model,
                "max_tokens": config.stage1_max_tokens,
                "temperature": config.labeling_temperature,
                "system": system_blocks,
                "messages": [{"role": "user", "content": user_msg}],
            },
        })

    funnel["stage1_sent"] += len(requests)

    # Process in chunks
    for chunk_start in range(0, len(requests), BATCH_CHUNK_SIZE):
        chunk = requests[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
        logger.info("stage1_submitting_chunk", request_count=len(chunk))

        batch_id = submit_batch(client, chunk)
        if not batch_id:
            continue

        state.stage1_batch_ids.append(batch_id)
        state.save()

        poll_until_complete(client, batch_id, poll_interval=poll_interval)

        # Parse and apply (save raw results inline)
        raw_path = labels_dir / f"stage1_raw_{batch_id}.jsonl"
        with open(raw_path, "w") as raw_file:
            for custom_id, message in iter_succeeded_results(client, batch_id):
                _save_raw_line(raw_file, custom_id, message)
                content_hash = custom_id.removeprefix("s1-")
                tier1_cats = parse_stage1_response(message)

                # Validate names and filter by confidence threshold
                filtered = []
                for c in tier1_cats:
                    name = c.get("name", "")
                    if name not in valid_tier1_names:
                        logger.warning("stage1_invalid_tier1", name=name)
                        continue
                    if c.get("confidence", 0) >= config.tier1_confidence_threshold:
                        filtered.append(c)

                if filtered:
                    state.complete_stage1(content_hash, filtered)
                    funnel["tier1_hits"] += 1
                else:
                    state.complete_stage1(content_hash, [])

        state.save()

        chunk_usage = aggregate_batch_usage(client, batch_id)
        _accumulate_usage(usage_totals, chunk_usage)
        logger.info("stage1_cache_stats", batch_id=batch_id, **chunk_usage)

    return _finalize_usage(usage_totals)


def _run_stage2(client, state, page_lookup, content_categories, config, labels_dir, poll_interval, funnel):
    """Run stage 2: subtree classification. Returns aggregated usage stats dict.

    Mutates `funnel` with stage2_sent / labels_emitted / errors counters.
    """
    usage_totals = _empty_usage_totals()
    needing = state.pages_needing_stage2()
    if not needing:
        logger.info("stage2_all_classified")
        return _finalize_usage(usage_totals)

    logger.info("stage2_classifying", page_count=len(needing))

    # Group pages by their tier-1 set for system prompt reuse
    groups = {}
    for content_hash in needing:
        tier1_names = tuple(sorted(state.get_tier1_for_page(content_hash)))
        if not tier1_names:
            # No tier-1 categories selected — mark as complete with empty labels
            state.complete_stage2(content_hash, [], "No tier-1 categories matched.")
            continue
        if tier1_names not in groups:
            groups[tier1_names] = []
        groups[tier1_names].append(content_hash)

    # Build valid category names for validation
    valid_names = {c["name"] for c in content_categories}

    # Build batch requests
    requests = []
    for tier1_set, hashes in groups.items():
        subtrees = get_subtree_categories(content_categories, set(tier1_set))
        system_prompt = build_stage2_system(
            subtrees,
            max_labels=config.max_labels,
            min_confidence=config.min_confidence,
        )
        system_blocks = _cacheable_system(system_prompt)

        for content_hash in hashes:
            page = page_lookup.get(content_hash)
            if not page:
                continue
            user_msg = build_user_message(
                page.get("text", ""),
                url=page.get("url", ""),
                truncation_words=config.text_truncation_words,
            )
            requests.append({
                "custom_id": f"s2-{content_hash}",
                "params": {
                    "model": config.labeling_model,
                    "max_tokens": config.stage2_max_tokens,
                    "temperature": config.labeling_temperature,
                    "system": system_blocks,
                    "messages": [{"role": "user", "content": user_msg}],
                },
            })

    funnel["stage2_sent"] += len(requests)

    # Process in chunks
    for chunk_start in range(0, len(requests), BATCH_CHUNK_SIZE):
        chunk = requests[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
        logger.info("stage2_submitting_chunk", request_count=len(chunk))

        batch_id = submit_batch(client, chunk)
        if not batch_id:
            continue

        state.stage2_batch_ids.append(batch_id)
        state.save()

        poll_until_complete(client, batch_id, poll_interval=poll_interval)

        raw_path = labels_dir / f"stage2_raw_{batch_id}.jsonl"
        with open(raw_path, "w") as raw_file:
            for custom_id, message in iter_succeeded_results(client, batch_id):
                _save_raw_line(raw_file, custom_id, message)
                content_hash = custom_id.removeprefix("s2-")
                result = parse_stage2_response(
                    message, valid_names,
                    min_confidence=config.min_confidence,
                    max_labels=config.max_labels,
                )

                if "error" in result:
                    state.mark_error(content_hash, result["error"])
                    funnel["errors"] += 1
                else:
                    state.complete_stage2(
                        content_hash,
                        result["categories"],
                        result.get("reasoning", ""),
                    )
                    funnel["labels_emitted"] += len(result["categories"])

        state.save()

        chunk_usage = aggregate_batch_usage(client, batch_id)
        _accumulate_usage(usage_totals, chunk_usage)
        logger.info("stage2_cache_stats", batch_id=batch_id, **chunk_usage)

    return _finalize_usage(usage_totals)


def _seed_from_existing_labels(state, labels_dir):
    """Seed label state from existing labels.json.

    Handles legacy imports that wrote labels directly without going
    through LabelState. Pages already tracked in state are skipped.
    """
    labels_file = labels_dir / "labels.json"
    if not labels_file.exists():
        return

    seeded = 0
    for entry in iter_ndjson(labels_file):
        content_hash = entry.get("content_hash", "")
        if not content_hash:
            continue
        if content_hash in state.pages:
            continue

        url = entry.get("url", "")
        categories = entry.get("categories", [])
        labels = [{"name": c, "confidence": 1.0} for c in categories]

        state.pages[content_hash] = {
            "url": url,
            "status": "stage2_complete",
            "tier1_categories": None,
            "labels": labels,
            "reasoning": "seeded from existing labels",
            "error": None,
        }
        seeded += 1

    if seeded:
        logger.info("seeded_from_existing_labels", count=seeded)
        state.save()


def _load_corpus(corpus_file):
    """Load corpus pages from NDJSON file."""
    return load_ndjson(Path(corpus_file))


def _save_raw_line(raw_file, custom_id, message):
    """Append a single raw result line to the JSONL backup file."""
    try:
        text = "".join(b.text for b in message.content if hasattr(b, "text"))
        raw_file.write(json.dumps({"custom_id": custom_id, "text": text}) + "\n")
    except Exception as e:
        logger.warning("raw_result_save_failed", custom_id=custom_id, error=str(e))


def _print_usage_summary(stage1_usage, stage2_usage):
    """Print a human-readable cache and cost summary to stdout."""
    stages = [("Stage 1", stage1_usage), ("Stage 2", stage2_usage)]
    stages = [(name, u) for name, u in stages if u is not None]
    if not stages:
        return

    print("\nLabeling usage (prompt cache enabled, 1h TTL):")
    total_cost = 0.0
    for name, u in stages:
        total_cost += u["estimated_cost_usd"]
        print(
            f"  {name}: "
            f"input={u['total_input_tokens']:,}  "
            f"output={u['total_output_tokens']:,}  "
            f"cache_write={u['total_cache_creation_tokens']:,}  "
            f"cache_read={u['total_cache_read_tokens']:,}  "
            f"hit_rate={u['cache_hit_rate']*100:.1f}%  "
            f"cost=${u['estimated_cost_usd']:.4f}"
        )
    print(f"  Total estimated cost: ${total_cost:.4f}")


def _write_labels(state, labels_dir):
    """Write final labels as NDJSON for collection seeding."""
    output_path = labels_dir / "labels.json"
    count = 0
    with open(output_path, "w") as f:
        for content_hash, page in state.pages.items():
            if page["status"] != "stage2_complete":
                continue
            labels = page.get("labels", [])
            entry = {
                "url": page["url"],
                "content_hash": content_hash,
                "categories": [c["name"] for c in labels],
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    logger.info("labels_written", count=count, path=str(output_path))
