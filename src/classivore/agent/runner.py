#!/usr/bin/env python3
"""Agent orchestrator — iterates collect → label → evaluate loops.

Prioritizes categories with the fewest labeled pages. Stops when
coverage targets are met, budget is exhausted, or yield drops to zero.
"""

import copy
import traceback
from pathlib import Path

import structlog

from classivore.agent.coverage import analyze_coverage
from classivore.agent.state import AgentState
from classivore.logging_config import get_logger
from classivore.models import AgentConfig, IterationPlan, IterationResult

logger = get_logger(__name__)

MAX_CATEGORIES_PER_ITERATION = 100


def run_agent(
    config,
    categories,
    hierarchy,
    data_dir,
    max_iterations=10,
    target_per_category=None,
    dry_run=False,
    poll_interval=30,
    verbose=False,
):
    """Run the data expansion agent.

    Iterates: analyze coverage → collect targeted pages → label new pages → evaluate.
    Prioritizes categories with the fewest labeled pages.

    Args:
        config: TaxonomyConfig instance.
        categories: List of category dicts (from load_taxonomy).
        hierarchy: Hierarchy dict (from build_hierarchy).
        data_dir: Path to data directory.
        max_iterations: Maximum number of collect/label cycles.
        target_per_category: Override target labeled pages per category.
        dry_run: Show coverage analysis without collecting or labeling.
        poll_interval: Seconds between batch status polls.
        verbose: Enable verbose logging.

    Returns:
        Agent summary dict.
    """
    structlog.contextvars.bind_contextvars(
        job_type="agent",
        taxonomy=config.slug,
    )

    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels" / config.slug
    agent_dir = data_dir / "agent" / config.slug

    target = target_per_category or config.target_per_category
    excluded_cats = set(config.excluded_categories)
    excluded_tier1 = set(config.excluded_tier1_categories)

    agent_config = AgentConfig(
        max_iterations=max_iterations,
        target_per_category=target,
    )

    # Aggregated metrics across iterations (for run recorder).
    aggregated_metrics: dict = {}

    # Initial coverage analysis
    report = analyze_coverage(
        categories, labels_dir, target,
        excluded_categories=excluded_cats,
        excluded_tier1=excluded_tier1,
    )
    initial_satisfied = report.satisfied_categories
    initial_total = report.total_categories

    logger.info(
        "coverage_analysis",
        total_categories=report.total_categories,
        covered=report.covered_categories,
        satisfied=report.satisfied_categories,
        gaps=len(report.gaps),
        coverage_pct=f"{report.coverage_pct:.1f}%",
        total_labeled=report.total_labeled_pages,
    )

    if dry_run:
        _print_coverage_report(report, agent_config)
        return {"dry_run": True, "coverage": report}

    # Load agent state (supports resume)
    agent_state = AgentState(agent_dir)
    start_iteration = agent_state.current_iteration()

    # Main loop — max_iterations is relative (run N more from now)
    for i in range(start_iteration, start_iteration + max_iterations):
        structlog.contextvars.bind_contextvars(iteration=i)

        # Check stop conditions
        should_stop, reason = agent_state.should_stop(agent_config, start_iteration)
        if should_stop:
            logger.info("agent_stopped", reason=reason)
            break

        # 1. What do we have?
        if i > 0:
            report = analyze_coverage(
                categories, labels_dir, target,
                excluded_categories=excluded_cats,
                excluded_tier1=excluded_tier1,
            )

        if not report.gaps:
            logger.info("agent_stopped", reason="no gaps remaining")
            break

        # 2. How many new pages does each category need?
        gaps = report.gaps[:MAX_CATEGORIES_PER_ITERATION]
        category_targets = {
            g.name: g.target_count - g.current_count for g in gaps
        }
        plan = IterationPlan(
            iteration=i,
            target_categories=list(category_targets.keys()),
        )
        logger.info(
            "iteration_start",
            target_categories=len(category_targets),
            agent_iteration=i,
        )
        agent_state.start_iteration(plan)

        # 3. Collect pages for those categories
        collection_summary = _run_collection(
            config, categories, data_dir, plan, category_targets, verbose,
        )
        if "metrics" in collection_summary:
            _merge_metrics(aggregated_metrics, collection_summary["metrics"])
        if "error_info" in collection_summary:
            aggregated_metrics.setdefault("errors", []).append({
                "iteration": i, "stage": "collection",
                **collection_summary["error_info"],
            })

        # 4. Label everything new
        labeling_summary = _run_labeling(
            config, categories, hierarchy, data_dir,
            poll_interval, verbose,
        )
        if "labeling_metrics" in labeling_summary:
            _merge_metrics(aggregated_metrics, {"labeling": labeling_summary["labeling_metrics"]})
        if "error_info" in labeling_summary:
            aggregated_metrics.setdefault("errors", []).append({
                "iteration": i, "stage": "labeling",
                **labeling_summary["error_info"],
            })

        # 5. How did we do?
        post_report = analyze_coverage(
            categories, labels_dir, target,
            excluded_categories=excluded_cats,
            excluded_tier1=excluded_tier1,
        )
        new_labels = max(0, post_report.total_labeled_pages - report.total_labeled_pages)

        result = IterationResult(
            iteration=i,
            pages_collected=collection_summary.get("total_collected", 0),
            pages_labeled=new_labels,
            categories_satisfied_before=report.satisfied_categories,
            categories_satisfied_after=post_report.satisfied_categories,
            gaps_before=len(report.gaps),
            gaps_after=len(post_report.gaps),
        )
        agent_state.complete_iteration(result)

        logger.info(
            "iteration_complete",
            pages_collected=result.pages_collected,
            pages_labeled=result.pages_labeled,
            satisfied_before=result.categories_satisfied_before,
            satisfied_after=result.categories_satisfied_after,
            gaps_remaining=result.gaps_after,
        )

        report = post_report

    summary = agent_state.summary()

    # Final coverage snapshot for the run report
    final_report = analyze_coverage(
        categories, labels_dir, target,
        excluded_categories=excluded_cats,
        excluded_tier1=excluded_tier1,
    )
    aggregated_metrics["coverage"] = {
        "at_target_before": initial_satisfied,
        "at_target_after": final_report.satisfied_categories,
        "total_categories": final_report.total_categories,
        "thin_remaining": len(final_report.gaps),
    }
    summary["metrics"] = aggregated_metrics

    logger.info("agent_complete", iterations=summary.get("iterations_completed"))
    return summary


def _merge_metrics(target, source):
    """Recursively sum numeric leaves from source into target."""
    for key, value in source.items():
        if isinstance(value, dict):
            sub = target.setdefault(key, {})
            if not isinstance(sub, dict):
                continue
            _merge_metrics(sub, value)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value



def _run_collection(config, categories, data_dir, plan, category_targets, verbose):
    """Run targeted collection for gap categories."""
    from classivore.collection import run_collection

    # Focus collection on target categories by excluding everything else
    modified_config = copy.copy(config)
    all_leaf_names = {
        c["display_name"] for c in categories if c["is_leaf"]
    }
    target_display = {
        c["display_name"] for c in categories
        if c["name"] in category_targets
    }
    non_target = all_leaf_names - target_display
    modified_config.excluded_categories = list(
        set(config.excluded_categories) | non_target
    )

    try:
        summary = run_collection(
            config=modified_config,
            categories=categories,
            data_dir=str(data_dir),
            resume=True,
            agent_iteration=plan.iteration,
            category_targets=category_targets,
            fresh_state=True,
            verbose=verbose,
        )
        return summary
    except Exception as e:
        # run_collection captures its own exceptions and returns a partial
        # summary. Anything reaching here is a setup-level failure.
        logger.exception("collection_invocation_failed", error=str(e))
        return {
            "total_collected": 0,
            "total_target": 0,
            "error_info": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }


def _run_labeling(config, categories, hierarchy, data_dir, poll_interval, verbose):
    """Run labeling on all unlabeled corpus pages."""
    from classivore.labeling import run_labeling

    try:
        summary = run_labeling(
            config=config,
            categories=categories,
            hierarchy=hierarchy,
            data_dir=str(data_dir),
            poll_interval=poll_interval,
            verbose=verbose,
        )
        return summary
    except Exception as e:
        # run_labeling captures its own exceptions and returns a partial
        # summary. Anything reaching here is a setup-level failure.
        logger.exception("labeling_invocation_failed", error=str(e))
        return {
            "stage2_complete": 0,
            "error_info": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }


def _print_coverage_report(report, config):
    """Print human-readable coverage report for dry run."""
    print(f"\nCoverage Analysis")
    print(f"{'=' * 60}")
    print(f"  Total categories:    {report.total_categories}")
    print(f"  With labels:         {report.covered_categories}")
    print(f"  Meeting target ({config.target_per_category}):  {report.satisfied_categories}")
    print(f"  Coverage:            {report.coverage_pct:.1f}%")
    print(f"  Total labeled pages: {report.total_labeled_pages}")
    print(f"  Categories to fill:  {len(report.gaps)}")

    if report.gaps:
        print(f"\n  Worst gaps (fewest labels first):")
        for gap in report.gaps[:20]:
            bar = '#' * min(gap.current_count, 40)
            print(f"    {gap.name:<40} {gap.current_count:>4}/{gap.target_count} {bar}")

        if len(report.gaps) > 20:
            print(f"    ... and {len(report.gaps) - 20} more")
