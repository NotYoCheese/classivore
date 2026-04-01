#!/usr/bin/env python3
"""
Classivore CLI — taxonomy-agnostic text classification pipeline.

Usage:
    classivore <command> [options]

Commands:
    enrich      Enrich taxonomy with LLM-generated descriptions
    collect     Collect training data from the web
    validate    Validate data quality
    label       Label scraped data with LLM
    train       Train classification model
    classify    Run inference on text
    agent       Run data expansion agent
    serve       Start local API server
    taxonomy    Show taxonomy info and stats
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="classivore",
        description="Taxonomy-agnostic text classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Each command module registers its own subparser
    _register_enrich(subparsers)
    _register_collect(subparsers)
    _register_validate(subparsers)
    _register_label(subparsers)
    _register_train(subparsers)
    _register_classify(subparsers)
    _register_agent(subparsers)
    _register_serve(subparsers)
    _register_taxonomy(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


def _add_common_args(subparser):
    """Add arguments common to all commands."""
    subparser.add_argument(
        "--taxonomy", "-t", default="iab-2.2",
        help="Taxonomy slug (default: iab-2.2)",
    )
    subparser.add_argument(
        "--data-dir", default=None,
        help="Override data directory",
    )
    subparser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Increase output detail",
    )


def _register_enrich(subparsers):
    p = subparsers.add_parser("enrich", help="Enrich taxonomy with descriptions")
    _add_common_args(p)
    p.add_argument("--review", action="store_true", help="Interactive review after enrichment")
    p.add_argument("--dry-run", action="store_true", help="Show what would be enriched")
    p.add_argument("--poll-interval", type=int, default=30, help="Batch poll interval in seconds")
    p.set_defaults(func=_cmd_enrich)


def _register_collect(subparsers):
    p = subparsers.add_parser("collect", help="Collect training data")
    _add_common_args(p)
    p.add_argument("--pages", type=int, default=100, help="Number of pages to collect")
    p.set_defaults(func=_cmd_collect)


def _register_validate(subparsers):
    p = subparsers.add_parser("validate", help="Validate data quality")
    _add_common_args(p)
    p.add_argument("--labeled", action="store_true", help="Validate labeled data")
    p.add_argument("--skip-noise", action="store_true", help="Skip label noise scoring (faster)")
    p.add_argument("--no-color", action="store_true", help="Disable colored output")
    p.set_defaults(func=_cmd_validate)


def _register_label(subparsers):
    p = subparsers.add_parser("label", help="Label data with LLM")
    _add_common_args(p)
    p.add_argument("--provider", choices=["anthropic", "bedrock"], default="anthropic")
    p.set_defaults(func=_cmd_label)


def _register_train(subparsers):
    p = subparsers.add_parser("train", help="Train classification model")
    _add_common_args(p)
    p.add_argument("--model-base", default=None, help="Override base model")
    p.set_defaults(func=_cmd_train)


def _register_classify(subparsers):
    p = subparsers.add_parser("classify", help="Run inference")
    _add_common_args(p)
    p.add_argument("--text", type=str, help="Text to classify")
    p.add_argument("--file", type=str, help="JSON file to classify")
    p.add_argument("--output", type=str, help="Output file for results")
    p.add_argument("--interactive", action="store_true", help="Interactive mode")
    p.set_defaults(func=_cmd_classify)


def _register_agent(subparsers):
    p = subparsers.add_parser("agent", help="Run data expansion agent")
    _add_common_args(p)
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="Simulate without API calls")
    p.set_defaults(func=_cmd_agent)


def _register_serve(subparsers):
    p = subparsers.add_parser("serve", help="Start local API server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.set_defaults(func=_cmd_serve)


def _register_taxonomy(subparsers):
    p = subparsers.add_parser("taxonomy", help="Show taxonomy info")
    _add_common_args(p)
    p.set_defaults(func=_cmd_taxonomy)


# Stub command handlers — each will be implemented in its own module
def _cmd_enrich(args):
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
        build_hierarchy,
        load_taxonomy,
        save_enriched_taxonomy,
    )

    config = load_taxonomy_config(args.taxonomy)

    # Resume from enriched file if it exists, otherwise load raw taxonomy
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        print(f"Resuming from {enriched_path.name}")
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)

    hierarchy = build_hierarchy(categories)
    requests = build_batch_requests(categories, hierarchy, config)

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
    """Interactive review of newly enriched categories."""
    enriched = [c for c in categories if c["id"] in results]
    print(f"\nReview {len(enriched)} enriched categories ([a]ccept / [s]kip / [q]uit):\n")

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


def _cmd_collect(args):
    print(f"TODO: collect {args.pages} pages for {args.taxonomy}")


def _cmd_validate(args):
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.validation.formatter import format_report
    from classivore.validation.runner import validate_labeled, validate_scraped

    data_dir = get_data_dir(args.data_dir)

    if args.labeled:
        # Try to load taxonomy categories for coverage checks
        taxonomy_categories = None
        try:
            config = load_taxonomy_config(args.taxonomy)
            taxonomy_file = config.taxonomy_file
            if taxonomy_file.exists():
                import csv
                with open(taxonomy_file) as f:
                    reader = csv.DictReader(f)
                    taxonomy_categories = [
                        row[config.name_column]
                        for row in reader
                        if row.get("is_leaf", "true").lower() == "true"
                    ]
        except Exception:
            pass  # taxonomy loading is optional for validation

        report = validate_labeled(
            data_dir=data_dir,
            taxonomy_slug=args.taxonomy,
            taxonomy_categories=taxonomy_categories,
            skip_noise=args.skip_noise,
        )
    else:
        report = validate_scraped(data_dir)

    print(format_report(report, use_color=not args.no_color))


def _cmd_label(args):
    print(f"TODO: label data for {args.taxonomy} with {args.provider}")


def _cmd_train(args):
    print(f"TODO: train model for {args.taxonomy}")


def _cmd_classify(args):
    print(f"TODO: classify with {args.taxonomy}")


def _cmd_agent(args):
    print(f"TODO: run agent for {args.taxonomy} (max {args.max_iterations} iterations)")


def _cmd_serve(args):
    print(f"TODO: serve API on {args.host}:{args.port}")


def _cmd_taxonomy(args):
    print(f"TODO: show taxonomy info for {args.taxonomy}")


if __name__ == "__main__":
    main()
