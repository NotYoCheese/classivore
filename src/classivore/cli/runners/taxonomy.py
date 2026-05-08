#!/usr/bin/env python3
"""Runner for `classivore taxonomy` — print stats, enrichment coverage, and gaps."""


def run(args):
    from collections import defaultdict
    from pathlib import Path

    from classivore.agent.coverage import analyze_coverage
    from classivore.config.settings import get_data_dir, load_taxonomy_config
    from classivore.taxonomy.loader import apply_enriched_if_present, load_taxonomy

    config = load_taxonomy_config(args.taxonomy)
    apply_enriched_if_present(config)
    categories = load_taxonomy(config)

    excluded = set(config.excluded_categories)
    excluded_tier1 = set(config.excluded_tier1_categories)

    total = len(categories)
    leaves = [c for c in categories if c["is_leaf"]]
    non_leaves = total - len(leaves)
    excluded_count = sum(1 for c in categories if c["display_name"] in excluded)

    depth_counts = defaultdict(int)
    for c in categories:
        depth_counts[c["depth"]] += 1

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

    diff_counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    for c in leaves:
        d = c.get("difficulty", "")
        if d in valid_difficulties:
            diff_counts[d] += 1
        else:
            diff_counts["unknown"] += 1
    n_leaves = len(leaves) or 1

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

    sep = "─" * 55
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
    hline = "─" * (len(header) - 2)
    print(f"  {hline}")
    for name, data in tier1_sorted:
        print(f"  {name.ljust(name_width)}  {data['total']:>5}  {data['leaves']:>6}  {data['hard']:>4}  {data['excl']:>4}")

    data_dir = get_data_dir(getattr(args, "data_dir", None))
    labels_dir = Path(data_dir) / "labels" / config.slug
    target = config.target_per_category

    report = analyze_coverage(
        categories, labels_dir, target,
        excluded_categories=None,
        excluded_tier1=excluded_tier1,
    )

    excluded_names = {c["name"] for c in categories if c["display_name"] in excluded}

    if report.gaps:
        print(f"\nUnsatisfied Categories  ({len(report.gaps)} below target of {target})")
        gap_labels = []
        for g in report.gaps:
            label = f"{g.tier1_name}: {g.name}" if g.tier1_name else g.name
            gap_labels.append((label, g))
        label_width = max(len(label) for label, _ in gap_labels)
        gline = "─" * (label_width + 25)
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
