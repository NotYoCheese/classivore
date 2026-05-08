#!/usr/bin/env python3
"""Runner for `classivore hints` — generate per-tier1 domain hints via LLM."""


def run(args):
    import yaml

    from classivore.config.settings import load_taxonomy_config
    from classivore.taxonomy.enricher import generate_domain_hints
    from classivore.taxonomy.loader import apply_enriched_if_present, load_taxonomy

    config = load_taxonomy_config(args.taxonomy)

    apply_enriched_if_present(config)
    categories = load_taxonomy(config)

    excluded_tier1 = set(config.excluded_tier1_categories)
    tier1_names = sorted({
        c["path"][0] for c in categories if c.get("path")
    } - excluded_tier1)

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

    new_hints = generate_domain_hints(missing, config)

    merged_hints = dict(existing)
    total_domains = 0
    for name, domains in new_hints.items():
        if domains:
            merged_hints[name] = domains
            total_domains += len(domains)

    with open(config.config_path) as f:
        raw = yaml.safe_load(f)
    raw["domain_hints"] = merged_hints
    with open(config.config_path, "w") as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated hints for {len(new_hints)} tier1 categories")
    print(f"  Total domains added: {total_domains}")
    print(f"  Config updated: {config.config_path}")
