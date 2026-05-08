#!/usr/bin/env python3
"""Runner for `classivore agent` — automated collect/label expansion loop."""


def run(args):
    from classivore.agent.coverage import analyze_coverage
    from classivore.agent.state import AgentState
    from classivore.cli.main import _record_run
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import (
        apply_enriched_if_present,
        build_hierarchy,
        load_taxonomy,
    )

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

        apply_enriched_if_present(config)
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

    apply_enriched_if_present(config)
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
