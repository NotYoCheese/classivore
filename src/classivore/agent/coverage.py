#!/usr/bin/env python3
"""Coverage analysis — identifies taxonomy gaps and prioritizes them.

Pure analysis module with no side effects or API calls. Reads existing
labels and compares against the full taxonomy to find categories that
need more labeled pages.
"""

from datetime import datetime, timezone
from pathlib import Path

from classivore.models import CategoryGap, CoverageReport
from classivore.persistence import iter_ndjson


def analyze_coverage(
    categories: list[dict],
    labels_dir: Path,
    target_per_category: int,
    excluded_categories: set[str] | None = None,
    excluded_tier1: set[str] | None = None,
) -> CoverageReport:
    """Analyze label coverage across taxonomy categories.

    Reads labels.json (NDJSON), counts per-category occurrences, compares
    against taxonomy leaf categories, returns gaps sorted by count ascending
    (fewest labels first = highest priority).

    Args:
        categories: List of category dicts from load_taxonomy.
        labels_dir: Path to labels directory (contains labels.json).
        target_per_category: Target labeled pages per category.
        excluded_categories: Display names to exclude (e.g. niche categories).
        excluded_tier1: Tier-1 names to exclude (e.g. metadata categories).

    Returns:
        CoverageReport with gaps sorted ascending by current_count.
    """
    excluded_categories = excluded_categories or set()
    excluded_tier1 = excluded_tier1 or set()

    # Get leaf categories, excluding metadata and configured exclusions
    leaf_cats = [
        c for c in categories
        if c["is_leaf"]
        and c["path"][0] not in excluded_tier1
        and c["display_name"] not in excluded_categories
    ]

    # Build name → tier1 lookup
    tier1_lookup = {c["name"]: c["path"][0] for c in leaf_cats}

    # Count labels per category from labels.json
    labels_file = labels_dir / "labels.json"
    label_counts: dict[str, int] = {}
    total_labeled = 0

    for entry in iter_ndjson(labels_file):
        total_labeled += 1
        for cat_name in entry.get("categories", []):
            label_counts[cat_name] = label_counts.get(cat_name, 0) + 1

    # Build gap list
    gaps = []
    covered = 0
    satisfied = 0

    for cat in leaf_cats:
        name = cat["name"]
        count = label_counts.get(name, 0)

        if count > 0:
            covered += 1
        if count >= target_per_category:
            satisfied += 1

        deficit = max(0, target_per_category - count)
        if deficit > 0:
            gaps.append(CategoryGap(
                name=name,
                current_count=count,
                target_count=target_per_category,
                deficit=deficit,
                tier1_name=tier1_lookup.get(name, ""),
            ))

    # Sort by current_count ascending (emptiest first)
    gaps.sort(key=lambda g: (g.current_count, g.name))

    return CoverageReport(
        total_categories=len(leaf_cats),
        covered_categories=covered,
        satisfied_categories=satisfied,
        total_labeled_pages=total_labeled,
        gaps=gaps,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
