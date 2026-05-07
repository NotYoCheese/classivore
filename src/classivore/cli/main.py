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
    init        Initialize a new taxonomy (validate, enrich, configure)
    hints       Generate domain hints for tier1 categories
    publish     Publish trained model to HuggingFace Hub
    hf          HuggingFace repo management
    taxonomy    Show taxonomy info and stats
"""

import argparse
import sys

from classivore.logging_config import get_logger

logger = get_logger(__name__)


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
    _register_init(subparsers)
    _register_hints(subparsers)
    _register_publish(subparsers)
    _register_hf(subparsers)
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
    from classivore.cli.runners.enrich import run

    p = subparsers.add_parser("enrich", help="Enrich taxonomy with descriptions")
    _add_common_args(p)
    p.add_argument("--review", action="store_true", help="Interactive review after enrichment")
    p.add_argument("--dry-run", action="store_true", help="Show what would be enriched")
    p.add_argument("--poll-interval", type=int, default=30, help="Batch poll interval in seconds")
    p.set_defaults(func=run)


def _register_collect(subparsers):
    p = subparsers.add_parser("collect", help="Collect training data")
    _add_common_args(p)
    p.add_argument("--resume", action="store_true", default=True, help="Resume from existing state (default)")
    p.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring existing state")
    p.add_argument("--queries-only", action="store_true", help="Generate queries without fetching content")
    p.add_argument("--audit-domains", action="store_true", help="Show domain quality report and exit")
    p.add_argument("--status", action="store_true", help="Show collection status dashboard and exit")
    p.add_argument("--json", action="store_true", help="Print run record as JSON instead of summary")
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
    p.add_argument("--json", action="store_true", help="Print run record as JSON instead of summary")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap each stage at N pages this run (for cost-bounded sampling). Resumes naturally on subsequent runs.")
    p.add_argument(
        "--prompt-cache", dest="prompt_cache", action=argparse.BooleanOptionalAction,
        default=None,
        help="Override prompt_cache config (use --no-prompt-cache to force off).",
    )
    p.set_defaults(func=_cmd_label)


def _register_train(subparsers):
    p = subparsers.add_parser("train", help="Train classification model")
    _add_common_args(p)
    p.add_argument("--model-base", default=None, help="Override base model")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--device", default=None, help="Force device (cuda, mps, cpu)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Show data stats and config without training")
    p.set_defaults(func=_cmd_train)


def _register_classify(subparsers):
    from classivore.cli.runners.classify import run

    p = subparsers.add_parser("classify", help="Run inference on text")
    _add_common_args(p)
    p.add_argument("--text", type=str, help="Text to classify")
    p.add_argument("--file", type=str, help="JSON/NDJSON file to classify")
    p.add_argument("--output", "-o", type=str, help="Output file (default: stdout)")
    p.add_argument("--interactive", action="store_true", help="Interactive stdin mode")
    p.add_argument("--model-dir", type=str, help="Path to model directory")
    p.set_defaults(func=run)


def _register_agent(subparsers):
    from classivore.cli.runners.agent import run

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
    p.add_argument("--json", action="store_true",
                   help="Print run record as JSON instead of summary")
    p.set_defaults(func=run)


def _register_init(subparsers):
    from classivore.cli.runners.init import run

    p = subparsers.add_parser("init", help="Initialize a new taxonomy")
    p.add_argument("--csv", required=True, help="Path to raw taxonomy CSV")
    p.add_argument("--name", required=True, help="Taxonomy name (e.g. 'IAB Content Taxonomy')")
    p.add_argument("--version", required=True, help="Version string (e.g. '2.2')")
    p.add_argument("--slug", required=True, help="Short identifier (e.g. 'iab-2.2')")
    p.add_argument("--taxonomies-dir", default="./taxonomies", help="Taxonomies directory")
    p.add_argument("--id-col", default="id", help="ID column name (default: id)")
    p.add_argument("--name-col", default="name", help="Name column name (default: name)")
    p.add_argument("--parent-col", default="parent_id", help="Parent column name (default: parent_id)")
    p.add_argument("--skip-enrichment", action="store_true", help="Skip batch API enrichment")
    p.add_argument("--skip-hints", action="store_true", help="Skip domain hint generation")
    p.add_argument("--dry-run", action="store_true", help="Validate and report only")
    p.add_argument("--verbose", "-v", action="store_true", help="Increase output detail")
    p.set_defaults(func=run)


def _register_hints(subparsers):
    from classivore.cli.runners.hints import run

    p = subparsers.add_parser("hints", help="Generate domain hints for tier1 categories")
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    p.set_defaults(func=run)


def _register_publish(subparsers):
    p = subparsers.add_parser("publish", help="Publish trained model to HuggingFace Hub")
    p.add_argument("--model-path", required=True, help="Path to trained model directory")
    p.add_argument("--repo-id", required=True, help="HuggingFace repo (e.g. classivore/iab22-deberta-large)")
    p.add_argument("--version", required=True, help="Semver tag (e.g. v1.0.0)")
    p.add_argument("--token", default=None, help="HuggingFace token (falls back to HUGGINGFACE_TOKEN env)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    p.add_argument("--verbose", "-v", action="store_true", help="Increase output detail")
    p.set_defaults(func=_cmd_publish)


def _register_hf(subparsers):
    p = subparsers.add_parser("hf", help="HuggingFace repo management")
    hf_sub = p.add_subparsers(dest="hf_command", help="HuggingFace subcommands")

    init_p = hf_sub.add_parser("init", help="Create HuggingFace repo")
    init_p.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    init_p.add_argument("--token", default=None, help="HuggingFace token (falls back to HUGGINGFACE_TOKEN env)")
    init_p.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    init_p.add_argument("--verbose", "-v", action="store_true", help="Increase output detail")
    init_p.set_defaults(func=_cmd_hf_init)


def _register_taxonomy(subparsers):
    from classivore.cli.runners.taxonomy import run

    p = subparsers.add_parser("taxonomy", help="Show taxonomy info")
    _add_common_args(p)
    p.set_defaults(func=run)


def _record_run(command, taxonomy, cli_args, data_dir, emit_json, run):
    """Wrap a command body in a RunRecorder + report block.

    Args:
        command: Command name (e.g. "collect").
        taxonomy: Taxonomy slug.
        cli_args: Dict of parsed CLI args (typically vars(args)).
        data_dir: Path to the data directory; runs.jsonl lives under it.
        emit_json: If True, print the run record as JSON instead of the report.
        run: Callable taking the RunRecorder; returns the underlying summary
            (which is also returned by _record_run itself).

    Returns:
        Whatever `run` returned (typically the command's summary dict).
    """
    import json as _json
    from pathlib import Path as _Path

    from classivore.runs import (
        RunRecorder,
        default_runs_path,
        format_summary,
        load_runs,
        sum_metrics,
    )

    runs_path = default_runs_path(_Path(data_dir))
    safe_args = {k: v for k, v in cli_args.items() if k != "func"}

    summary = None
    with RunRecorder(command=command, taxonomy=taxonomy, args=safe_args, runs_path=runs_path) as recorder:
        summary = run(recorder)

    record = recorder.record
    all_runs = load_runs(runs_path, command=command, taxonomy=taxonomy)
    all_time = sum_metrics(all_runs)

    if emit_json:
        print(_json.dumps({"record": record, "all_time": all_time}, indent=2))
    else:
        print(format_summary(record, all_time))

    return summary


# Stub command handlers — each will be implemented in its own module
def _cmd_collect(args):
    from classivore.collection import audit_domains, run_collection
    from classivore.collection.dashboard import format_status_dashboard
    from classivore.collection.domains import DomainTracker
    from classivore.collection.state import CollectionState
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import apply_enriched_if_present, load_taxonomy

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
        labels_dir = Path(data_dir) / "labels" / config.slug
        state = CollectionState(collection_dir)
        domains = DomainTracker(shared_dir)
        print(format_status_dashboard(
            state, domains,
            corpus_file=corpus_file,
            taxonomy_slug=config.slug,
            labels_dir=labels_dir,
            target_per_category=config.target_per_category,
        ))
        return

    apply_enriched_if_present(config)
    categories = load_taxonomy(config)

    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"Data dir: {data_dir}")

    resume = args.resume and not args.no_resume

    summary = _record_run(
        command="collect",
        taxonomy=args.taxonomy,
        cli_args=vars(args),
        data_dir=data_dir,
        emit_json=getattr(args, "json", False),
        run=lambda recorder: _do_collect(
            recorder, config, categories, data_dir, resume, args,
        ),
    )

    if not getattr(args, "json", False):
        print(f"\nCollection complete:")
        print(f"  Categories: {summary['satisfied_categories']}/{summary['total_categories']} satisfied")
        print(f"  Pages: {summary['total_collected']}/{summary['total_target']}")


def _do_collect(recorder, config, categories, data_dir, resume, args):
    from classivore.collection import run_collection

    summary = run_collection(
        config=config,
        categories=categories,
        data_dir=data_dir,
        resume=resume,
        queries_only=args.queries_only,
        verbose=args.verbose,
    )
    if "metrics" in summary:
        recorder.metrics.update(summary["metrics"])
    return summary


def _cmd_validate(args):
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import apply_enriched_if_present
    from classivore.validation.formatter import format_report
    from classivore.validation.runner import validate_labeled, validate_scraped

    data_dir = get_data_dir(args.data_dir)

    if args.labeled:
        # Try to load taxonomy categories for coverage checks
        taxonomy_categories = None
        try:
            config = load_taxonomy_config(args.taxonomy)
            apply_enriched_if_present(config)
            if config.taxonomy_file.exists():
                import csv
                with open(config.taxonomy_file) as f:
                    reader = csv.DictReader(f)
                    # Include all categories (leaf and non-leaf) since
                    # labeling can assign non-leaf categories
                    taxonomy_categories = [
                        row[config.name_column]
                        for row in reader
                    ]
        except Exception as e:
            # Taxonomy loading is optional for validation — fall through.
            logger.debug("validation_taxonomy_load_skipped", taxonomy=args.taxonomy, error=str(e))

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
    from classivore.labeling.state import LabelState
    from classivore.taxonomy.loader import (
        apply_enriched_if_present,
        build_hierarchy,
        load_taxonomy,
    )

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    if args.prompt_cache is not None:
        config.prompt_cache = args.prompt_cache

    if args.status:
        from pathlib import Path
        state = LabelState(Path(data_dir) / "labels" / config.slug)
        print(state.summary_str())
        return

    apply_enriched_if_present(config)
    categories = load_taxonomy(config)
    hierarchy = build_hierarchy(categories)

    print(f"Taxonomy: {config.name} ({len(categories)} categories)")
    print(f"Data dir: {data_dir}")

    if args.dry_run:
        from classivore.labeling import run_labeling
        run_labeling(
            config=config, categories=categories, hierarchy=hierarchy,
            data_dir=data_dir, stage=args.stage, dry_run=True,
            poll_interval=args.poll_interval, verbose=args.verbose,
            limit=args.limit,
        )
        return

    summary = _record_run(
        command="label",
        taxonomy=args.taxonomy,
        cli_args=vars(args),
        data_dir=data_dir,
        emit_json=getattr(args, "json", False),
        run=lambda recorder: _do_label(
            recorder, config, categories, hierarchy, data_dir, args,
        ),
    )

    if not getattr(args, "json", False):
        triaged = summary["stage1_complete"] + summary["stage2_complete"]
        print(f"\nLabeling complete:")
        print(f"  Triaged:  {triaged} pages")
        print(f"  Labeled:  {summary['stage2_complete']} pages")
        print(f"  Pending:  {summary['unlabeled']} pages")
        print(f"  Errors:   {summary['error']}")


def _do_label(recorder, config, categories, hierarchy, data_dir, args):
    from classivore.labeling import run_labeling

    summary = run_labeling(
        config=config,
        categories=categories,
        hierarchy=hierarchy,
        data_dir=data_dir,
        stage=args.stage,
        dry_run=False,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
        limit=args.limit,
    )
    if "labeling_metrics" in summary:
        recorder.metrics["labeling"] = summary["labeling_metrics"]
    return summary


def _cmd_train(args):
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import apply_enriched_if_present
    from classivore.training.trainer import train_model

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    # Apply overrides
    if args.model_base:
        config.model_base = args.model_base

    apply_enriched_if_present(config)

    result = train_model(
        config=config,
        data_dir=data_dir,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nTraining complete:")
        print(f"  Model saved to: {result['model_path']}")
        print(f"  Training time:  {result['training_time']}s")
        metrics = result.get("metrics", {})
        if "eval_f1_micro" in metrics:
            print(f"  Val F1 micro:   {metrics['eval_f1_micro']:.4f}")
            print(f"  Val F1 macro:   {metrics['eval_f1_macro']:.4f}")
def _cmd_publish(args):
    import os

    from dotenv import load_dotenv

    from classivore.publishing.hub import publish_model

    load_dotenv()
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Error: No HuggingFace token. Set --token or HUGGINGFACE_TOKEN env var.")
        sys.exit(1)

    try:
        result = publish_model(
            model_path=args.model_path,
            repo_id=args.repo_id,
            version=args.version,
            token=token,
            dry_run=args.dry_run,
        )
        if result:
            print(f"\nPublished successfully: {result}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_hf_init(args):
    import os

    from dotenv import load_dotenv

    from classivore.publishing.hub import init_repo

    load_dotenv()
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Error: No HuggingFace token. Set --token or HUGGINGFACE_TOKEN env var.")
        sys.exit(1)

    init_repo(args.repo_id, token, private=not args.public)
    print(f"Repo ready: {args.repo_id}")
