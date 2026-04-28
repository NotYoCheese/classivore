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
    serve       Start local API server
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
    p = subparsers.add_parser("classify", help="Run inference on text")
    _add_common_args(p)
    p.add_argument("--text", type=str, help="Text to classify")
    p.add_argument("--file", type=str, help="JSON/NDJSON file to classify")
    p.add_argument("--output", "-o", type=str, help="Output file (default: stdout)")
    p.add_argument("--interactive", action="store_true", help="Interactive stdin mode")
    p.add_argument("--model-dir", type=str, help="Path to model directory")
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
    p.add_argument("--json", action="store_true",
                   help="Print run record as JSON instead of summary")
    p.set_defaults(func=_cmd_agent)


def _register_init(subparsers):
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
    p.set_defaults(func=_cmd_init)


def _register_hints(subparsers):
    p = subparsers.add_parser("hints", help="Generate domain hints for tier1 categories")
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    p.set_defaults(func=_cmd_hints)


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


def _register_serve(subparsers):
    p = subparsers.add_parser("serve", help="Start local API server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.set_defaults(func=_cmd_serve)


def _register_taxonomy(subparsers):
    p = subparsers.add_parser("taxonomy", help="Show taxonomy info")
    _add_common_args(p)
    p.set_defaults(func=_cmd_taxonomy)


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

    # Filter out excluded tier1 categories before enrichment
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
    from classivore.taxonomy.loader import load_taxonomy

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
    from classivore.taxonomy.loader import build_hierarchy, load_taxonomy

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    if args.prompt_cache is not None:
        config.prompt_cache = args.prompt_cache

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

    if args.dry_run:
        from classivore.labeling import run_labeling
        run_labeling(
            config=config, categories=categories, hierarchy=hierarchy,
            data_dir=data_dir, stage=args.stage, dry_run=True,
            poll_interval=args.poll_interval, verbose=args.verbose,
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
    )
    if "labeling_metrics" in summary:
        recorder.metrics["labeling"] = summary["labeling_metrics"]
    return summary


def _cmd_train(args):
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.training.trainer import train_model

    config = load_taxonomy_config(args.taxonomy)
    data_dir = get_data_dir(args.data_dir)

    # Apply overrides
    if args.model_base:
        config.model_base = args.model_base

    # Load enriched taxonomy
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path

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


def _cmd_classify(args):
    import json
    from pathlib import Path

    from classivore.config.settings import get_models_dir
    from classivore.inference import Classifier

    # Resolve model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        models_dir = get_models_dir()
        slug_dir = models_dir / args.taxonomy
        if not slug_dir.is_dir():
            print(f"Error: no models found for taxonomy '{args.taxonomy}' in {models_dir}")
            sys.exit(1)
        subdirs = sorted(d for d in slug_dir.iterdir() if d.is_dir() and d.name != "checkpoints")
        if not subdirs:
            print(f"Error: no model runs found in {slug_dir}")
            sys.exit(1)
        model_dir = subdirs[-1]

    print(f"Loading model from {model_dir}...")
    try:
        classifier = Classifier(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.text:
        results = classifier.predict(args.text)
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"text": args.text, "predictions": results}, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            _print_classify_results(results)

    elif args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}")
            sys.exit(1)

        # Read input: JSON array or NDJSON
        raw = input_path.read_text().strip()
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(line) for line in raw.split("\n") if line.strip()]

        texts = [r["text"] for r in records]
        all_results = classifier.predict_batch(texts)

        # Merge predictions back into records
        output_records = []
        for record, preds in zip(records, all_results):
            out = {k: v for k, v in record.items() if k != "text"}
            out["text"] = record["text"]
            out["predictions"] = preds
            output_records.append(out)

        # Write output
        output_lines = [json.dumps(r) for r in output_records]
        output_text = "\n".join(output_lines) + "\n"
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"Classified {len(records)} texts → {args.output}")
        else:
            print(output_text, end="")

    elif args.interactive:
        print("Classivore interactive mode. Enter text, press Enter. Ctrl-D to exit.")
        try:
            while True:
                text = input("> ")
                if not text.strip():
                    continue
                results = classifier.predict(text)
                _print_classify_results(results)
        except EOFError:
            print()

    else:
        print("Error: specify --text, --file, or --interactive")
        sys.exit(1)


def _print_classify_results(results):
    """Print classification results as a formatted table."""
    if not results:
        print("  No categories above threshold.")
        return

    name_width = max(len(r["name"]) for r in results)
    name_width = max(name_width, 8)  # minimum "Category" header width

    line = "\u2500" * (name_width + 40)
    print(f"\n  {'Category'.ljust(name_width)}  {'Confidence':>10}  Path")
    print(f"  {line}")
    for r in results:
        path_str = " > ".join(r["path"]) if r.get("path") else ""
        print(f"  {r['name'].ljust(name_width)}  {r['confidence']:>10.4f}  {path_str}")
    print()


def _cmd_agent(args):
    from classivore.agent.coverage import analyze_coverage
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

    if args.dry_run:
        from classivore.agent.runner import run_agent
        run_agent(
            config=config, categories=categories, hierarchy=hierarchy,
            data_dir=data_dir, max_iterations=args.max_iterations,
            target_per_category=args.target, dry_run=True,
            poll_interval=args.poll_interval, verbose=args.verbose,
        )
        return

    summary = _record_run(
        command="agent",
        taxonomy=args.taxonomy,
        cli_args=vars(args),
        data_dir=data_dir,
        emit_json=getattr(args, "json", False),
        run=lambda recorder: _do_agent(
            recorder, config, categories, hierarchy, data_dir, args,
        ),
    )

    if not getattr(args, "json", False):
        print(f"\nAgent complete:")
        print(f"  Iterations: {summary.get('iterations_completed', 0)}")
        print(f"  Collected:  {summary.get('total_pages_collected', 0)} pages")
        print(f"  Labeled:    {summary.get('total_pages_labeled', 0)} pages")


def _do_agent(recorder, config, categories, hierarchy, data_dir, args):
    from classivore.agent.runner import run_agent

    summary = run_agent(
        config=config,
        categories=categories,
        hierarchy=hierarchy,
        data_dir=data_dir,
        max_iterations=args.max_iterations,
        target_per_category=args.target,
        dry_run=False,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
    )
    if "metrics" in summary:
        recorder.metrics.update(summary["metrics"])
    return summary


def _cmd_init(args):
    from pathlib import Path

    import yaml

    from classivore.taxonomy.onboarding import (
        generate_default_config,
        normalize_taxonomy_csv,
        print_onboarding_report,
        validate_csv,
        write_config,
    )

    csv_path = Path(args.csv)
    taxonomies_dir = Path(args.taxonomies_dir)
    slug = args.slug

    # 1. Validate CSV
    print(f"Validating {csv_path}...")
    stats = validate_csv(csv_path, args.id_col, args.name_col, args.parent_col)

    if stats.get("errors"):
        print(f"\nValidation errors:")
        for err in stats["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    if stats.get("warnings"):
        print(f"\nWarnings:")
        for warn in stats["warnings"]:
            print(f"  - {warn}")

    print(f"\nValidation passed:")
    print(f"  Categories: {stats['total']}")
    print(f"  Leaves: {stats['leaves']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Tier1 categories: {len(stats['tier1_names'])}")

    if args.dry_run:
        print(f"\nDry run — no files written.")
        return

    # 2. Create taxonomy directory
    tax_dir = taxonomies_dir / slug
    tax_dir.mkdir(parents=True, exist_ok=True)

    # 3. Normalize CSV — compute path/depth/is_leaf/children_count from parent_id
    dest_csv = tax_dir / "taxonomy.csv"
    normalize_taxonomy_csv(
        csv_path, dest_csv,
        id_col=args.id_col, name_col=args.name_col, parent_col=args.parent_col,
    )
    print(f"\nWrote normalized taxonomy CSV to {dest_csv}")

    # 4. Generate and write config
    config_dict = generate_default_config(
        slug=slug, name=args.name, version=args.version,
        taxonomy_file="taxonomy.csv",
        id_col=args.id_col, name_col=args.name_col, parent_col=args.parent_col,
    )
    config_path = tax_dir / "config.yaml"
    write_config(config_dict, config_path)
    print(f"Generated config at {config_path}")

    # 5. Enrichment (unless skipped)
    if not args.skip_enrichment:
        print(f"\nRunning enrichment...")
        from classivore.batch import (
            get_api_client,
            iter_succeeded_results,
            poll_until_complete,
            submit_batch,
        )
        from classivore.config.settings import TaxonomyConfig
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

        config = TaxonomyConfig(config_path)
        categories = load_taxonomy(config)
        hierarchy = build_hierarchy(categories)
        requests = build_batch_requests(categories, hierarchy, config)

        if requests:
            print(f"  Submitting {len(requests)} enrichment requests...")
            client = get_api_client()
            batch_id = submit_batch(client, requests)
            print(f"  Batch ID: {batch_id}")
            print(f"  Polling...")
            poll_until_complete(client, batch_id, poll_interval=30,
                                verbose=getattr(args, "verbose", False))

            results = {}
            for custom_id, message in iter_succeeded_results(client, batch_id):
                cat_id = custom_id.removeprefix("cat-")
                results[cat_id] = parse_enrichment(message)

            apply_results(categories, results)
            enriched_path = tax_dir / "taxonomy_enriched.csv"
            save_enriched_taxonomy(categories, enriched_path)
            print(f"  Enriched {len(results)} categories → {enriched_path}")
        else:
            print("  All categories already enriched.")
    else:
        print("\nSkipping enrichment (--skip-enrichment)")

    # 6. Domain hints (unless skipped)
    if not args.skip_hints:
        print(f"\nGenerating domain hints...")
        from classivore.config.settings import TaxonomyConfig
        from classivore.taxonomy.enricher import generate_domain_hints
        from classivore.taxonomy.loader import load_taxonomy

        config = TaxonomyConfig(config_path)
        enriched_path = tax_dir / "taxonomy_enriched.csv"
        if enriched_path.exists():
            config.taxonomy_file = enriched_path
        categories = load_taxonomy(config)

        excluded_tier1 = set(config.excluded_tier1_categories)
        tier1_names = sorted({
            c["path"][0] for c in categories if c.get("path")
        } - excluded_tier1)

        new_hints = generate_domain_hints(tier1_names, config)

        # Merge into config.yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        raw["domain_hints"] = {k: v for k, v in new_hints.items() if v}
        with open(config_path, "w") as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False)

        total_domains = sum(len(v) for v in new_hints.values())
        print(f"  Generated hints for {len(new_hints)} tier1 categories ({total_domains} domains)")
    else:
        print("\nSkipping domain hints (--skip-hints)")

    # 7. Onboarding report
    from classivore.config.settings import TaxonomyConfig
    from classivore.taxonomy.loader import load_taxonomy

    config = TaxonomyConfig(config_path)
    enriched_path = tax_dir / "taxonomy_enriched.csv"
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)
    print_onboarding_report(categories, tax_dir)

    # 8. Next steps
    print(f"  Next steps:")
    print(f"    1. Review {config_path}")
    print(f"       - Adjust targets_by_difficulty if needed")
    print(f"       - Add to excluded_categories any categories flagged above")
    print(f"       - Add to excluded_tier1_categories any metadata tier1s")
    print(f"    2. Run: classivore collect --taxonomy {slug}")
    print(f"    3. Run: classivore label --taxonomy {slug}")
    print()


def _cmd_hints(args):
    import yaml

    from classivore.config.settings import load_taxonomy_config
    from classivore.taxonomy.enricher import generate_domain_hints
    from classivore.taxonomy.loader import load_taxonomy

    config = load_taxonomy_config(args.taxonomy)

    # Load enriched taxonomy if available
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)

    # Get unique tier1 names, excluding metadata categories
    excluded_tier1 = set(config.excluded_tier1_categories)
    tier1_names = sorted({
        c["path"][0] for c in categories if c.get("path")
    } - excluded_tier1)

    # Find which tier1s are missing hints
    existing = config.domain_hints
    missing = [name for name in tier1_names if name not in existing]

    print(f"Taxonomy: {config.name}")
    print(f"  Tier1 categories: {len(tier1_names)}")
    print(f"  Already have hints: {len(tier1_names) - len(missing)}")
    print(f"  Need hints: {len(missing)}")

    if not missing:
        print("\nAll tier1 categories already have domain hints.")
        return

    if args.dry_run:
        print("\nDry run — would generate hints for:")
        for name in missing:
            print(f"  {name}")
        return

    # Generate hints
    new_hints = generate_domain_hints(missing, config)

    # Merge into existing config
    merged_hints = dict(existing)
    total_domains = 0
    for name, domains in new_hints.items():
        if domains:
            merged_hints[name] = domains
            total_domains += len(domains)

    # Write updated config.yaml preserving all other fields
    with open(config.config_path) as f:
        raw = yaml.safe_load(f)
    raw["domain_hints"] = merged_hints
    with open(config.config_path, "w") as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated hints for {len(new_hints)} tier1 categories")
    print(f"  Total domains added: {total_domains}")
    print(f"  Config updated: {config.config_path}")


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


def _cmd_serve(args):
    print(f"TODO: serve API on {args.host}:{args.port}")


def _cmd_taxonomy(args):
    from collections import defaultdict

    from classivore.config.settings import load_taxonomy_config
    from classivore.taxonomy.loader import load_taxonomy

    config = load_taxonomy_config(args.taxonomy)
    enriched_path = config.enriched_file or (
        config.taxonomy_file.parent / "taxonomy_enriched.csv"
    )
    if enriched_path.exists():
        config.taxonomy_file = enriched_path
    categories = load_taxonomy(config)

    excluded = set(config.excluded_categories)
    excluded_tier1 = set(config.excluded_tier1_categories)

    # --- Category Counts ---
    total = len(categories)
    leaves = [c for c in categories if c["is_leaf"]]
    non_leaves = total - len(leaves)
    excluded_count = sum(1 for c in categories if c["display_name"] in excluded)

    # --- Depth Distribution ---
    depth_counts = defaultdict(int)
    for c in categories:
        depth_counts[c["depth"]] += 1

    # --- Enrichment Coverage ---
    valid_difficulties = {"easy", "medium", "hard"}
    has_desc = sum(1 for c in categories if c["description"].strip())
    has_aliases = sum(1 for c in categories if c.get("aliases"))
    has_diff = sum(1 for c in categories if c.get("difficulty") in valid_difficulties)
    missing_any = sum(
        1 for c in categories
        if not c["description"].strip()
        or not c.get("aliases")
        or c.get("difficulty") not in valid_difficulties
    )

    # --- Difficulty Breakdown (leaves only) ---
    diff_counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    for c in leaves:
        d = c.get("difficulty", "")
        if d in valid_difficulties:
            diff_counts[d] += 1
        else:
            diff_counts["unknown"] += 1
    n_leaves = len(leaves) or 1

    # --- Tier-1 Summary ---
    tier1_data = defaultdict(lambda: {"total": 0, "leaves": 0, "hard": 0, "excl": 0})
    for c in categories:
        t1 = c["path"][0] if c.get("path") else c["name"]
        if t1 in excluded_tier1:
            continue
        tier1_data[t1]["total"] += 1
        if c["is_leaf"]:
            tier1_data[t1]["leaves"] += 1
            if c.get("difficulty") == "hard":
                tier1_data[t1]["hard"] += 1
        if c["display_name"] in excluded:
            tier1_data[t1]["excl"] += 1

    # --- Print ---
    sep = "\u2500" * 55
    print(f"\n{sep}")
    print(f"Taxonomy: {config.name} v{config.version}  (slug: {config.slug})")
    print(f"{sep}\n")

    print(f"Category Counts")
    print(f"  Total categories:      {total:>4}")
    print(f"  Leaf categories:       {len(leaves):>4}")
    print(f"  Non-leaf categories:   {non_leaves:>4}")
    print(f"  Excluded from training:{excluded_count:>4}   (from config.excluded_categories)")

    print(f"\nDepth Distribution")
    for depth in sorted(depth_counts):
        pct = depth_counts[depth] / total * 100
        print(f"  Depth {depth}:  {depth_counts[depth]:>4}  ({pct:>5.1f}%)")

    print(f"\nEnrichment Coverage")
    print(f"  With description:  {has_desc:>4} ({has_desc/total*100:>5.1f}%)")
    print(f"  With aliases:      {has_aliases:>4} ({has_aliases/total*100:>5.1f}%)")
    print(f"  With difficulty:   {has_diff:>4} ({has_diff/total*100:>5.1f}%)")
    print(f"  Missing any field: {missing_any:>4}")

    print(f"\nDifficulty Breakdown  (leaf categories only)")
    print(f"  Easy:    {diff_counts['easy']:>4}  ({diff_counts['easy']/n_leaves*100:>5.1f}%)")
    print(f"  Medium:  {diff_counts['medium']:>4}  ({diff_counts['medium']/n_leaves*100:>5.1f}%)")
    print(f"  Hard:    {diff_counts['hard']:>4}  ({diff_counts['hard']/n_leaves*100:>5.1f}%)")
    if diff_counts["unknown"]:
        print(f"  Unknown: {diff_counts['unknown']:>4}  ({diff_counts['unknown']/n_leaves*100:>5.1f}%)")

    tier1_sorted = sorted(tier1_data.items())
    name_width = max((len(name) for name, _ in tier1_sorted), default=20)
    print(f"\nTier-1 Summary  ({len(tier1_sorted)} top-level categories)")
    header = f"  {'Tier-1 Name'.ljust(name_width)}  {'Total':>5}  {'Leaves':>6}  {'Hard':>4}  {'Excl':>4}"
    print(header)
    hline = "\u2500" * (len(header) - 2)
    print(f"  {hline}")
    for name, data in tier1_sorted:
        print(f"  {name.ljust(name_width)}  {data['total']:>5}  {data['leaves']:>6}  {data['hard']:>4}  {data['excl']:>4}")

    # --- Coverage / Unsatisfied Categories ---
    from pathlib import Path

    from classivore.agent.coverage import analyze_coverage
    from classivore.config.settings import get_data_dir

    data_dir = get_data_dir(getattr(args, "data_dir", None))
    labels_dir = Path(data_dir) / "labels" / config.slug
    target = config.target_per_category

    report = analyze_coverage(
        categories, labels_dir, target,
        excluded_categories=None,
        excluded_tier1=excluded_tier1,
    )

    # Build lookup from category name → display_name for excluded check
    name_to_display = {c["name"]: c["display_name"] for c in categories}
    excluded_names = {c["name"] for c in categories if c["display_name"] in excluded}

    if report.gaps:
        print(f"\nUnsatisfied Categories  ({len(report.gaps)} below target of {target})")
        # Show tier1: name for clarity
        gap_labels = []
        for g in report.gaps:
            label = f"{g.tier1_name}: {g.name}" if g.tier1_name else g.name
            gap_labels.append((label, g))
        label_width = max(len(label) for label, _ in gap_labels)
        gline = "\u2500" * (label_width + 25)
        print(f"  {'Category'.ljust(label_width)}  {'Labels':>6}  {'Target':>6}  {'Deficit':>7}")
        print(f"  {gline}")
        for label, g in gap_labels:
            marker = " *" if g.name in excluded_names else ""
            print(f"  {label.ljust(label_width)}  {g.current_count:>6}  {g.target_count:>6}  {g.deficit:>7}{marker}")
        if excluded:
            print(f"\n  * excluded from collection (config.excluded_categories)")
    else:
        print(f"\nAll categories at target ({target} labels)")

    print(f"  Total labeled pages: {report.total_labeled_pages}")
    print(f"  Satisfied: {report.satisfied_categories}/{report.total_categories}")

    # --- Exclusions ---
    if excluded or excluded_tier1:
        print(f"\nExclusions")
        if excluded_tier1:
            print(f"  Excluded tier-1 categories ({len(excluded_tier1)}):")
            for name in sorted(excluded_tier1):
                print(f"    {name}")
        if excluded:
            print(f"  Excluded categories ({len(excluded)}):")
            for name in sorted(excluded):
                print(f"    {name}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
