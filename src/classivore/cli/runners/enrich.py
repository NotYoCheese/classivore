#!/usr/bin/env python3
"""Runner for `classivore enrich` — LLM-augment a taxonomy with batch API."""


def run(args):
    from classivore.batch import (
        get_api_client,
        iter_succeeded_results,
        poll_until_complete,
        submit_batch,
    )
    from classivore.config.settings import load_taxonomy_config
    from classivore.taxonomy.enricher import (
        apply_results,
        build_batch_requests,
        parse_enrichment,
    )
    from classivore.taxonomy.loader import (
        apply_enriched_if_present,
        build_hierarchy,
        enriched_taxonomy_path,
        load_taxonomy,
        save_enriched_taxonomy,
    )

    config = load_taxonomy_config(args.taxonomy)
    enriched_path = enriched_taxonomy_path(config)
    if apply_enriched_if_present(config):
        print(f"Resuming from {enriched_path.name}")
    categories = load_taxonomy(config)

    excluded_tier1 = set(config.excluded_tier1_categories)
    enrichable = [
        c for c in categories
        if not c["path"] or c["path"][0] not in excluded_tier1
    ]

    hierarchy = build_hierarchy(enrichable)
    requests = build_batch_requests(enrichable, hierarchy, config)

    already_enriched = len(categories) - len(requests)
    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"  Already enriched: {already_enriched}")
    print(f"  To enrich: {len(requests)}")

    if not requests:
        print("Nothing to enrich.")
        return

    if args.dry_run:
        print("Dry run — no API calls made.")
        return

    client = get_api_client()

    print(f"\nSubmitting batch ({len(requests)} requests)...")
    batch_id = submit_batch(client, requests)
    print(f"Batch ID: {batch_id}")

    print(f"Polling every {args.poll_interval}s...")
    poll_until_complete(
        client, batch_id, poll_interval=args.poll_interval, verbose=args.verbose,
    )

    print("\nProcessing results...")
    results = {}
    for custom_id, message in iter_succeeded_results(client, batch_id):
        cat_id = custom_id.removeprefix("cat-")
        results[cat_id] = parse_enrichment(message)

    apply_results(categories, results)
    save_enriched_taxonomy(categories, enriched_path)
    print(f"\nSaved enriched taxonomy to {enriched_path}")
    print(f"  Enriched: {len(results)} categories")

    if args.review:
        _review_enrichments(categories, results)


def _review_enrichments(categories, results):
    enriched = [c for c in categories if c["id"] in results]
    print(f"\nReview {len(enriched)} enriched categories (Enter for next, q to quit):\n")

    for cat in enriched:
        print(f"  {cat['display_name']}")
        print(f"    Description: {cat['description']}")
        print(f"    Boundary:    {cat['boundaries']}")

        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nReview ended.")
            return

        if choice == "q":
            print("Review ended.")
            return
