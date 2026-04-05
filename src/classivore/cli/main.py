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

    # Configure logging before dispatching to command handler
    from classivore.logging_config import configure_logging
    configure_logging(verbose=getattr(args, "verbose", False))

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
    p.add_argument("--pages", type=int, default=None, help="Total pages to collect (distributed across categories)")
    p.add_argument("--resume", action="store_true", default=True, help="Resume from existing state (default)")
    p.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring existing state")
    p.add_argument("--queries-only", action="store_true", help="Generate queries without fetching content")
    p.add_argument("--audit-domains", action="store_true", help="Show domain quality report and exit")
    p.add_argument("--status", action="store_true", help="Show collection status dashboard and exit")
    p.set_defaults(func=_cmd_collect)


def _register_validate(subparsers):
    p = subparsers.add_parser("validate", help="Validate data quality")
    _add_common_args(p)
    p.add_argument("--labeled", action="store_true", help="Validate labeled data")
    p.add_argument("--skip-noise", action="store_true", help="Skip label noise scoring (faster)")
    p.add_argument("--no-color", action="store_true", help="Disable colored output")
    p.set_defaults(func=_cmd_validate)


def _register_label(subparsers):
    p = subparsers.add_parser("label", help="Label corpus pages with LLM")
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", help="Show what would be labeled without API calls")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all", help="Run specific stage (default: all)")
    p.add_argument("--poll-interval", type=int, default=30, help="Batch poll interval in seconds")
    p.add_argument("--status", action="store_true", help="Show labeling progress and exit")
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
    p.add_argument("--max-iterations", type=int, default=10,
                   help="Maximum collect/label cycles (default: 10)")
    p.add_argument("--target", type=int, default=None,
                   help="Target labeled pages per category (default: from config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show coverage analysis without collecting or labeling")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Batch poll interval in seconds")
    p.add_argument("--status", action="store_true",
                   help="Show agent run history and current coverage")
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
    from classivore.collection import audit_domains, run_collection
    from classivore.collection.dashboard import format_status_dashboard
    from classivore.collection.domains import DomainTracker
    from classivore.collection.state import CollectionState
    from classivore.config.settings import get_data_dir, load_taxonomy_config

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    if args.audit_domains:
        print(audit_domains(data_dir))
        return

    if args.status:
        from pathlib import Path
        collection_dir = Path(data_dir) / "collection" / config.slug
        shared_dir = Path(data_dir) / "collection"
        corpus_file = Path(data_dir) / "corpus" / "pages.json"
        state = CollectionState(collection_dir)
        domains = DomainTracker(shared_dir)
        print(format_status_dashboard(state, domains, corpus_file=corpus_file, taxonomy_slug=config.slug))
        return

    # Load enriched taxonomy if available
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)

    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"Data dir: {data_dir}")

    resume = args.resume and not args.no_resume

    summary = run_collection(
        config=config,
        categories=categories,
        data_dir=data_dir,
        pages=args.pages,
        resume=resume,
        queries_only=args.queries_only,
        verbose=args.verbose,
    )

    print(f"\nCollection complete:")
    print(f"  Categories: {summary['satisfied_categories']}/{summary['total_categories']} satisfied")
    print(f"  Pages: {summary['total_collected']}/{summary['total_target']}")


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
            enriched_path = config.enriched_file or (
                config.taxonomy_file.parent / "taxonomy_enriched.csv"
            )
            taxonomy_file = enriched_path if enriched_path.exists() else config.taxonomy_file
            if taxonomy_file.exists():
                import csv
                with open(taxonomy_file) as f:
                    reader = csv.DictReader(f)
                    # Include all categories (leaf and non-leaf) since
                    # labeling can assign non-leaf categories
                    taxonomy_categories = [
                        row[config.name_column]
                        for row in reader
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
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.labeling import run_labeling
    from classivore.labeling.state import LabelState
    from classivore.taxonomy.loader import build_hierarchy, load_taxonomy

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    if args.status:
        from pathlib import Path
        state = LabelState(Path(data_dir) / "labels" / config.slug)
        print(state.summary_str())
        return

    # Load enriched taxonomy
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)
    hierarchy = build_hierarchy(categories)

    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"Data dir: {data_dir}")

    summary = run_labeling(
        config=config,
        categories=categories,
        hierarchy=hierarchy,
        data_dir=data_dir,
        stage=args.stage,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
    )

    triaged = summary["stage1_complete"] + summary["stage2_complete"]
    print(f"\nLabeling complete:")
    print(f"  Triaged:  {triaged} pages")
    print(f"  Labeled:  {summary['stage2_complete']} pages")
    print(f"  Pending:  {summary['unlabeled']} pages")
    print(f"  Errors:   {summary['error']}")


def _cmd_train(args):
    print(f"TODO: train model for {args.taxonomy}")


def _cmd_classify(args):
    print(f"TODO: classify with {args.taxonomy}")


def _cmd_agent(args):
    from classivore.agent.coverage import analyze_coverage
    from classivore.agent.runner import run_agent
    from classivore.agent.state import AgentState
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import build_hierarchy, load_taxonomy

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    if args.status:
        from pathlib import Path
        agent_dir = Path(data_dir) / "agent" / config.slug
        labels_dir = Path(data_dir) / "labels" / config.slug
        agent_state = AgentState(agent_dir)
        summary = agent_state.summary()

        print(f"Agent Status: {config.name}")
        print("=" * 50)
        print(f"  Iterations completed: {summary['iterations_completed']}")
        print(f"  Total collected:      {summary['total_pages_collected']}")
        print(f"  Total labeled:        {summary['total_pages_labeled']}")
        if summary['started_at']:
            print(f"  Started:              {summary['started_at']}")
        if summary['last_checkpoint_at']:
            print(f"  Last update:          {summary['last_checkpoint_at']}")

        # Show current coverage
        enriched_path = config.enriched_file or (
            config.taxonomy_file.parent / "taxonomy_enriched.csv"
        )
        if enriched_path.exists():
            config.taxonomy_file = enriched_path
        categories = load_taxonomy(config)
        target = args.target or config.target_per_category

        report = analyze_coverage(
            categories, labels_dir, target,
            excluded_categories=set(config.excluded_categories),
            excluded_tier1=set(config.excluded_tier1_categories),
        )
        print(f"\n  Coverage: {report.satisfied_categories}/{report.total_categories} "
              f"categories at target ({report.coverage_pct:.1f}%)")
        print(f"  Total labeled pages:  {report.total_labeled_pages}")
        print(f"  Categories to fill:   {len(report.gaps)}")
        return

    # Load enriched taxonomy
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)
    hierarchy = build_hierarchy(categories)

    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"Data dir: {data_dir}")

    summary = run_agent(
        config=config,
        categories=categories,
        hierarchy=hierarchy,
        data_dir=data_dir,
        max_iterations=args.max_iterations,
        target_per_category=args.target,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
    )

    if not args.dry_run:
        print(f"\nAgent complete:")
        print(f"  Iterations: {summary.get('iterations_completed', 0)}")
        print(f"  Collected:  {summary.get('total_pages_collected', 0)} pages")
        print(f"  Labeled:    {summary.get('total_pages_labeled', 0)} pages")


def _cmd_serve(args):
    print(f"TODO: serve API on {args.host}:{args.port}")


def _cmd_taxonomy(args):
    print(f"TODO: show taxonomy info for {args.taxonomy}")


if __name__ == "__main__":
    main()
