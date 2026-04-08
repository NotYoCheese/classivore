#!/usr/bin/env python3
"""Taxonomy onboarding — validation, config generation, and reporting."""

import csv
from collections import defaultdict
from pathlib import Path

import yaml


def validate_csv(csv_path, id_col="id", name_col="name", parent_col="parent_id"):
    """Validate a taxonomy CSV for structural integrity.

    Args:
        csv_path: Path to the CSV file.
        id_col: Column name for category ID.
        name_col: Column name for category name.
        parent_col: Column name for parent ID.

    Returns:
        Stats dict with total, leaves, max_depth, tier1_names, warnings, errors.
    """
    csv_path = Path(csv_path)
    errors = []
    warnings = []

    if not csv_path.exists():
        return {"errors": [f"File not found: {csv_path}"], "warnings": []}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Check required columns
        for col in (id_col, name_col, parent_col):
            if col not in fieldnames:
                errors.append(f"Missing required column: {col}")
        if errors:
            return {"errors": errors, "warnings": warnings}

        rows = list(reader)

    if not rows:
        errors.append("CSV is empty")
        return {"errors": errors, "warnings": warnings}

    # Check ID uniqueness
    ids = [r[id_col] for r in rows]
    id_set = set(ids)
    if len(ids) != len(id_set):
        dupes = [i for i in ids if ids.count(i) > 1]
        errors.append(f"Duplicate IDs: {sorted(set(dupes))[:10]}")

    # Check parent references
    for row in rows:
        pid = row[parent_col]
        if pid and pid not in id_set:
            errors.append(f"Invalid parent_id '{pid}' for category '{row[name_col]}'")

    # Build hierarchy for depth/cycle checks
    children_map = defaultdict(list)
    parent_map = {}
    for row in rows:
        children_map[row[parent_col]].append(row[id_col])
        parent_map[row[id_col]] = row[parent_col]

    # Check for cycles
    for row in rows:
        visited = set()
        current = row[id_col]
        while current and current in parent_map:
            if current in visited:
                errors.append(f"Cycle detected involving category '{row[name_col]}'")
                break
            visited.add(current)
            current = parent_map.get(current, "")

    # Compute stats
    roots = [r for r in rows if not r[parent_col]]
    leaves = [r for r in rows if r[id_col] not in children_map]

    # Compute depths
    depth_cache = {}

    def _depth(cat_id):
        if cat_id in depth_cache:
            return depth_cache[cat_id]
        pid = parent_map.get(cat_id, "")
        if not pid or pid not in parent_map:
            depth_cache[cat_id] = 1
        else:
            depth_cache[cat_id] = _depth(pid) + 1
        return depth_cache[cat_id]

    for row in rows:
        _depth(row[id_col])

    max_depth = max(depth_cache.values()) if depth_cache else 0

    # Tier1 names (direct children of roots, or roots themselves if flat)
    tier1_ids = set()
    for root in roots:
        tier1_ids.add(root[id_col])
    tier1_names = sorted(r[name_col] for r in rows if r[id_col] in tier1_ids)

    return {
        "total": len(rows),
        "leaves": len(leaves),
        "max_depth": max_depth,
        "tier1_names": tier1_names,
        "warnings": warnings,
        "errors": errors,
    }


def generate_default_config(slug, name, version, taxonomy_file,
                            id_col="id", name_col="name", parent_col="parent_id"):
    """Generate a default config dict for a new taxonomy.

    Args:
        slug: Short identifier (e.g. "iab-2.2").
        name: Human-readable name.
        version: Version string.
        taxonomy_file: Filename of the taxonomy CSV.
        id_col: ID column name.
        name_col: Name column name.
        parent_col: Parent column name.

    Returns:
        Config dict suitable for yaml.dump().
    """
    return {
        "name": name,
        "version": version,
        "slug": slug,
        "taxonomy_file": taxonomy_file,
        "enriched_file": "taxonomy_enriched.csv",
        "id_column": id_col,
        "name_column": name_col,
        "parent_column": parent_col,
        "description_column": None,
        "classification_type": "multi_label",
        "max_labels": 3,
        "min_confidence": 0.5,
        "model_base": "microsoft/deberta-v3-large",
        "batch_size": 8,
        "learning_rate": 2e-5,
        "max_length": 512,
        "num_epochs": 3,
        "focal_loss": {
            "alpha": 0.75,
            "gamma": 3.5,
        },
        "class_weight_cap": 7.0,
        "enrichment": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens_per_category": 300,
        },
        "labeling": {
            "model": "claude-haiku-4-5-20251001",
            "stage1_max_tokens": 300,
            "stage2_max_tokens": 500,
            "tier1_confidence_threshold": 0.3,
            "temperature": 0.0,
            "text_truncation_words": 3000,
        },
        "excluded_tier1_categories": [],
        "collection": {
            "targets_by_difficulty": {
                "easy": 30,
                "medium": 50,
                "hard": 80,
                "default": 40,
            },
            "max_queries_by_difficulty": {
                "easy": 10,
                "medium": 18,
                "hard": 30,
                "default": 12,
            },
            "llm_queries_from_iteration": {
                "easy": 3,
                "medium": 2,
                "hard": 1,
                "default": 2,
            },
            "max_per_domain_per_category": 3,
            "commoncrawl_crawl_id": None,
            "query_model": "claude-haiku-4-5-20251001",
        },
        "filter_config": {
            "defaults": {
                "min_words": 150,
                "min_path_depth": 3,
                "min_unique_word_ratio": 0.30,
                "allow_listing_pages": False,
            },
            "relaxations": {},
        },
        "domain_hints": {},
        "excluded_categories": [],
    }


CONFIG_HEADER = """\
# Classivore taxonomy configuration
# Generated by: classivore init
#
# Key sections:
#   enrichment          — LLM model for generating descriptions/aliases
#   labeling            — LLM model for classifying pages
#   collection          — per-difficulty targets and query budgets
#   filter_config       — content filters with per-tier1 relaxations
#   domain_hints        — authoritative domains per tier1 (auto-generated)
#   excluded_categories — categories to skip during collection
#   excluded_tier1_categories — metadata tier1s excluded from labeling
#
# After init, review and adjust:
#   - targets_by_difficulty (pages to collect per category)
#   - excluded_tier1_categories (metadata categories like "Content Language")
#   - excluded_categories (niche categories to skip)
#   - filter_config.relaxations (per-tier1 filter overrides)

"""


def write_config(config_dict, output_path):
    """Write config dict to YAML with header comments.

    Args:
        config_dict: Config dict to write.
        output_path: Path to write to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_body = yaml.dump(
        config_dict,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )

    with open(output_path, "w") as f:
        f.write(CONFIG_HEADER)
        f.write(yaml_body)


def print_onboarding_report(categories, output_path):
    """Print a comprehensive onboarding report to stdout.

    Args:
        categories: List of category dicts (from load_taxonomy).
        output_path: Path to the taxonomy directory (for display).
    """
    total = len(categories)
    leaves = [c for c in categories if c.get("is_leaf", True)]
    depths = [c.get("depth", 1) for c in categories]
    max_depth = max(depths) if depths else 0

    print(f"\n{'=' * 60}")
    print(f"  Taxonomy Onboarding Report")
    print(f"{'=' * 60}")
    print(f"  Total categories: {total}  |  Leaf categories: {len(leaves)}  |  Max depth: {max_depth}")

    # Difficulty distribution
    diff_counts = {"easy": 0, "medium": 0, "hard": 0, "": 0}
    for c in leaves:
        d = c.get("difficulty", "")
        if d in diff_counts:
            diff_counts[d] += 1
        else:
            diff_counts[""] += 1

    n_leaves = len(leaves) or 1
    print(f"\n  Difficulty distribution (leaf categories):")
    print(f"    Easy:    {diff_counts['easy']:>4} ({diff_counts['easy']*100//n_leaves}%)")
    print(f"    Medium:  {diff_counts['medium']:>4} ({diff_counts['medium']*100//n_leaves}%)")
    print(f"    Hard:    {diff_counts['hard']:>4} ({diff_counts['hard']*100//n_leaves}%)")
    if diff_counts[""]:
        print(f"    Unknown: {diff_counts['']:>4} (not yet enriched)")

    # Hardest categories
    hard_cats = [c for c in leaves if c.get("difficulty") == "hard"]
    if hard_cats:
        print(f"\n  Top 20 hardest categories:")
        for c in hard_cats[:20]:
            path_str = " > ".join(c.get("path", [c["name"]]))
            print(f"    {c['name']:<40} {path_str}")
        if len(hard_cats) > 20:
            print(f"    ... and {len(hard_cats) - 20} more")

    # Empty aliases
    no_aliases = [c for c in leaves if not c.get("aliases")]
    if no_aliases:
        print(f"\n  Categories with empty aliases ({len(no_aliases)} — review recommended):")
        for c in no_aliases[:15]:
            print(f"    {c['name']}")
        if len(no_aliases) > 15:
            print(f"    ... and {len(no_aliases) - 15} more")

    # Data quality issues
    issues = []
    # Non-leaf with no children
    parent_ids = {c.get("parent_id") for c in categories}
    for c in categories:
        if not c.get("is_leaf") and c["id"] not in parent_ids:
            issues.append(f"Non-leaf '{c['name']}' has no children in CSV")
    # Depth vs path mismatch
    for c in categories:
        path_len = len(c.get("path", []))
        depth = c.get("depth", 1)
        if path_len != depth:
            issues.append(f"'{c['name']}': depth={depth} but path has {path_len} segments")
    if issues:
        print(f"\n  Potential data quality issues ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    # Collection estimates
    n_easy = diff_counts["easy"]
    n_medium = diff_counts["medium"]
    n_hard = diff_counts["hard"]
    n_unknown = diff_counts[""]
    total_pages = n_easy * 30 + n_medium * 50 + n_hard * 80 + n_unknown * 40
    cost = total_pages * 0.002

    print(f"\n  Estimated collection at default targets:")
    print(f"    Easy   ({n_easy:>3} cats x 30 pages): {n_easy * 30:>6} pages")
    print(f"    Medium ({n_medium:>3} cats x 50 pages): {n_medium * 50:>6} pages")
    print(f"    Hard   ({n_hard:>3} cats x 80 pages): {n_hard * 80:>6} pages")
    if n_unknown:
        print(f"    Unknown ({n_unknown:>2} cats x 40 pages): {n_unknown * 40:>6} pages")
    print(f"    Total:                       {total_pages:>6} pages")
    print(f"    Est. labeling cost @ $0.002/page: ${cost:.2f}")

    print(f"\n  Output: {output_path}")
    print()
