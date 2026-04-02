#!/usr/bin/env python3
"""Collection status dashboard.

Formats collection state and domain tracker data into a human-readable
status report with coverage histogram, velocity, ETA, and error breakdown.
"""

from datetime import datetime, timezone
from pathlib import Path


def format_status_dashboard(state, domains, corpus_file=None, taxonomy_slug=""):
    """Format a collection status dashboard.

    Args:
        state: CollectionState instance.
        domains: DomainTracker instance.
        corpus_file: Path to corpus pages.json (for total count).
        taxonomy_slug: Taxonomy slug for display.

    Returns:
        Formatted string for terminal output.
    """
    lines = []
    title = f"Collection Status ({taxonomy_slug})" if taxonomy_slug else "Collection Status"
    lines.append(title)
    lines.append("=" * 60)

    # --- Run info ---
    _add_run_info(lines, state)

    # --- Progress ---
    _add_progress(lines, state, corpus_file)

    # --- Coverage histogram ---
    _add_coverage_histogram(lines, state)

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
        except (ValueError, TypeError):
            lines.append(f"Started:     {state.started_at}")
    else:
        lines.append("Started:     Not yet started")

    if state.last_checkpoint_at:
        lines.append(f"Last update: {state.last_checkpoint_at}")


def _add_progress(lines, state, corpus_file):
    summary = state.summary()
    total_cats = summary["total_categories"]
    satisfied = summary["satisfied_categories"]
    collected = summary["total_collected"]
    target = summary["total_target"]

    lines.append("")
    if total_cats > 0:
        pct_cats = satisfied / total_cats * 100
        lines.append(f"Progress:  {satisfied}/{total_cats} categories satisfied ({pct_cats:.1f}%)")
    else:
        lines.append("Progress:  No categories initialized")

    if target > 0:
        pct_pages = collected / target * 100
        lines.append(f"           {collected}/{target} pages collected ({pct_pages:.1f}%)")

    if corpus_file and Path(corpus_file).exists():
        try:
            total_corpus = sum(1 for line in open(corpus_file) if line.strip())
            lines.append(f"           {total_corpus} total corpus pages (shared)")
        except Exception:
            pass


def _add_coverage_histogram(lines, state):
    histogram = state.coverage_histogram()
    if not any(histogram.values()):
        return

    lines.append("")
    lines.append("Coverage")
    lines.append("-" * 60)

    max_count = max(histogram.values()) if any(histogram.values()) else 1
    for bucket, count in histogram.items():
        bar_len = int(count / max_count * 30) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        label = f"{bucket:>5} pages:"
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

        summary = state.summary()
        remaining = summary["total_target"] - summary["total_collected"]
        if remaining > 0 and rate > 0:
            minutes_left = remaining / rate
            if minutes_left < 60:
                lines.append(f"  Est. remaining: ~{int(minutes_left)} minutes")
            else:
                hours_left = minutes_left / 60
                lines.append(f"  Est. remaining: ~{hours_left:.0f} hours")
        elif remaining <= 0:
            lines.append("  Est. remaining: complete")
    else:
        lines.append("  Rate: no recent activity")
        lines.append("  Est. remaining: unknown")


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
