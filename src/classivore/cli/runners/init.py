#!/usr/bin/env python3
"""Runner for `classivore init` — onboard a new taxonomy from a CSV."""

import sys


def run(args):
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

    tax_dir = taxonomies_dir / slug
    tax_dir.mkdir(parents=True, exist_ok=True)

    dest_csv = tax_dir / "taxonomy.csv"
    normalize_taxonomy_csv(
        csv_path, dest_csv,
        id_col=args.id_col, name_col=args.name_col, parent_col=args.parent_col,
    )
    print(f"\nWrote normalized taxonomy CSV to {dest_csv}")

    config_dict = generate_default_config(
        slug=slug, name=args.name, version=args.version,
        taxonomy_file="taxonomy.csv",
        id_col=args.id_col, name_col=args.name_col, parent_col=args.parent_col,
    )
    config_path = tax_dir / "config.yaml"
    write_config(config_dict, config_path)
    print(f"Generated config at {config_path}")

    if not args.skip_enrichment:
        _run_enrichment(config_path, tax_dir, args)
    else:
        print("\nSkipping enrichment (--skip-enrichment)")

    if not args.skip_hints:
        _run_domain_hints(config_path, yaml)
    else:
        print("\nSkipping domain hints (--skip-hints)")

    from classivore.config.settings import TaxonomyConfig
    from classivore.taxonomy.loader import apply_enriched_if_present, load_taxonomy

    config = TaxonomyConfig(config_path)
    apply_enriched_if_present(config)
    categories = load_taxonomy(config)
    print_onboarding_report(categories, tax_dir)

    print(f"  Next steps:")
    print(f"    1. Review {config_path}")
    print(f"       - Adjust targets_by_difficulty if needed")
    print(f"       - Add to excluded_categories any categories flagged above")
    print(f"       - Add to excluded_tier1_categories any metadata tier1s")
    print(f"    2. Run: classivore collect --taxonomy {slug}")
    print(f"    3. Run: classivore label --taxonomy {slug}")
    print()


def _run_enrichment(config_path, tax_dir, args):
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

    if not requests:
        print("  All categories already enriched.")
        return

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


def _run_domain_hints(config_path, yaml):
    print(f"\nGenerating domain hints...")
    from classivore.config.settings import TaxonomyConfig
    from classivore.taxonomy.enricher import generate_domain_hints
    from classivore.taxonomy.loader import apply_enriched_if_present, load_taxonomy

    config = TaxonomyConfig(config_path)
    apply_enriched_if_present(config)
    categories = load_taxonomy(config)

    excluded_tier1 = set(config.excluded_tier1_categories)
    tier1_names = sorted({
        c["path"][0] for c in categories if c.get("path")
    } - excluded_tier1)

    new_hints = generate_domain_hints(tier1_names, config)

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["domain_hints"] = {k: v for k, v in new_hints.items() if v}
    with open(config_path, "w") as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)

    total_domains = sum(len(v) for v in new_hints.values())
    print(f"  Generated hints for {len(new_hints)} tier1 categories ({total_domains} domains)")
