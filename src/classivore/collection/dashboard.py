#!/usr/bin/env python3
"""Collection status dashboard.

Formats collection state and domain tracker data into a human-readable
status report with coverage histogram, velocity, ETA, and error breakdown.

Coverage data comes from labels.json (source of truth). Operational metrics
(velocity, errors, domains) come from CollectionState.
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from classivore.logging_config import get_logger
from classivore.persistence import iter_ndjson

logger = get_logger(__name__)


def format_status_dashboard(state, domains, corpus_file=None, taxonomy_slug="",
                            labels_dir=None, target_per_category=None):
    """Format a collection status dashboard.

    Args:
        state: CollectionState instance.
        domains: DomainTracker instance.
        corpus_file: Path to corpus pages.json (for total count).
        taxonomy_slug: Taxonomy slug for display.
        labels_dir: Path to labels directory (for coverage from labels.json).
        target_per_category: Target labels per category (for coverage display).

    Returns:
        Formatted string for terminal output.
    """
    lines = []
    title = f"Collection Status ({taxonomy_slug})" if taxonomy_slug else "Collection Status"
    lines.append(title)
    lines.append("=" * 60)

    # --- Run info ---
    _add_run_info(lines, state)

    # --- Coverage from labels (source of truth) ---
    if labels_dir and target_per_category:
        _add_label_coverage(lines, labels_dir, target_per_category)

    # --- Corpus total ---
    if corpus_file and Path(corpus_file).exists():
        try:
            total_corpus = sum(1 for line in open(corpus_file) if line.strip())
            lines.append(f"           {total_corpus} total corpus pages (shared)")
        except OSError as e:
            logger.debug("dashboard_corpus_count_failed", path=str(corpus_file), error=str(e))

    # --- Coverage histogram from labels ---
    if labels_dir and target_per_category:
        _add_label_histogram(lines, labels_dir, target_per_category)

    # --- Velocity ---
    _add_velocity(lines, state)

    # --- Errors ---
    _add_errors(lines, state)

    # --- Domain summary ---
    _add_domain_summary(lines, domains)

    return "\n".join(lines)


def _add_run_info(lines, state):
    lines.append("")
    if state.started_at:
        try:
            started = datetime.fromisoformat(state.started_at)
            elapsed = datetime.now(timezone.utc) - started
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes = remainder // 60
            lines.append(f"Started:     {state.started_at} ({hours}h {minutes}m ago)")
        except (ValueError, TypeError) as e:
            logger.debug("dashboard_started_at_parse_failed", raw=state.started_at, error=str(e))
            lines.append(f"Started:     {state.started_at}")
    else:
        lines.append("Started:     Not yet started")

    if state.last_checkpoint_at:
        lines.append(f"Last update: {state.last_checkpoint_at}")


def _add_label_coverage(lines, labels_dir, target):
    """Show coverage from labels.json — the source of truth."""
    labels_file = Path(labels_dir) / "labels.json"
    if not labels_file.exists():
        lines.append("")
        lines.append("Progress:  No labels yet")
        return

    label_counts = Counter()
    total_pages = 0
    for entry in iter_ndjson(labels_file):
        total_pages += 1
        for cat in entry.get("categories", []):
            label_counts[cat] += 1

    num_categories = len(label_counts)
    satisfied = sum(1 for c in label_counts.values() if c >= target)

    lines.append("")
    if num_categories > 0:
        pct = satisfied / num_categories * 100 if num_categories else 0
        lines.append(f"Progress:  {satisfied}/{num_categories} categories at target ({pct:.1f}%)")
    else:
        lines.append("Progress:  No labeled categories")
    lines.append(f"           {total_pages} labeled pages")


def _add_label_histogram(lines, labels_dir, target):
    """Coverage histogram from label counts."""
    labels_file = Path(labels_dir) / "labels.json"
    if not labels_file.exists():
        return

    label_counts = Counter()
    for entry in iter_ndjson(labels_file):
        for cat in entry.get("categories", []):
            label_counts[cat] += 1

    if not label_counts:
        return

    buckets = {"0": 0, "1-10": 0, "11-20": 0, "21-50": 0, "50+": 0}
    for count in label_counts.values():
        if count == 0:
            buckets["0"] += 1
        elif count <= 10:
            buckets["1-10"] += 1
        elif count <= 20:
            buckets["11-20"] += 1
        elif count <= 50:
            buckets["21-50"] += 1
        else:
            buckets["50+"] += 1

    lines.append("")
    lines.append("Label Coverage")
    lines.append("-" * 60)

    max_count = max(buckets.values()) if any(buckets.values()) else 1
    for bucket, count in buckets.items():
        bar_len = int(count / max_count * 30) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        label = f"{bucket:>6} labels:"
        lines.append(f"  {label} {count:>4} categories  {bar}")


def _add_velocity(lines, state):
    recent = state.recent_urls(minutes=10)
    lines.append("")
    lines.append("Velocity (last 10 min)")
    lines.append("-" * 60)
    lines.append(f"  Pages collected: {recent}")

    if recent > 0:
        rate = recent / 10.0
        lines.append(f"  Rate: {rate:.1f} pages/min")
    else:
        lines.append("  Rate: no recent activity")


def _add_errors(lines, state):
    ec = state.error_counts
    total_errors = sum(ec.values())
    if total_errors == 0:
        return

    lines.append("")
    lines.append("Errors")
    lines.append("-" * 60)
    lines.append(f"  Search errors:  {ec['search_errors']}")
    lines.append(f"  Fetch failures: {ec['fetch_errors']}")
    lines.append(f"  Filtered:       {ec['filtered']}")
    lines.append(f"  Duplicates:     {ec['duplicates']}")


def _add_domain_summary(lines, domains):
    if not domains.scores and not domains.blocklist:
        return

    lines.append("")
    lines.append("Top Domains (by attempts)")
    lines.append("-" * 60)

    if domains.scores:
        sorted_domains = sorted(
            domains.scores.items(),
            key=lambda x: x[1]["attempts"],
            reverse=True,
        )
        for domain, entry in sorted_domains[:10]:
            rate = entry["successes"] / max(entry["attempts"], 1)
            lines.append(
                f"  {domain:<35} {rate:>5.0%}  {entry['successes']}/{entry['attempts']}"
            )

    blocked_count = len(domains.blocklist)
    auto_blocked = sum(1 for d in domains.scores if domains.is_blocked(d) and d not in domains.blocklist)
    if blocked_count or auto_blocked:
        lines.append(f"  Blocked: {blocked_count} manual, {auto_blocked} auto")
