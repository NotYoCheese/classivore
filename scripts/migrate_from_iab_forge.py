#!/usr/bin/env python3
"""
Migrate data from iab_forge into classivore's shared corpus + taxonomy labels format.

This script:
1. Creates taxonomy.csv for IAB 2.2 from the official IAB TSV
2. Converts scraped pages into the shared corpus format (deduped by URL + content)
3. Converts labeled pages into taxonomy-specific label files
4. Reports stats on deduplication and data quality

Usage:
    python scripts/migrate_from_iab_forge.py
    python scripts/migrate_from_iab_forge.py --iab-forge-dir /path/to/iab_forge
    python scripts/migrate_from_iab_forge.py --dry-run
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(data)} records to {path}")


def create_taxonomy_csv(iab_tsv_path, output_path):
    """Convert official IAB 2.2 TSV to classivore taxonomy.csv format."""
    rows = []
    with open(iab_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # Skip header row

        for row in reader:
            if not row or not row[0].strip():
                continue

            uid = row[0].strip()

            # Skip non-numeric IDs (header remnants, metadata rows)
            if not uid.isdigit():
                continue
            parent = row[1].strip() if len(row) > 1 else ""
            name = row[2].strip() if len(row) > 2 else ""
            tier1 = row[3].strip() if len(row) > 3 else ""
            tier2 = row[4].strip() if len(row) > 4 else ""
            tier3 = row[5].strip() if len(row) > 5 else ""
            tier4 = row[6].strip() if len(row) > 6 else ""

            # Build display_name: "Tier1: LowestTier"
            lowest = tier4 or tier3 or tier2 or tier1
            display_name = f"{tier1}: {lowest}" if tier1 else name

            # Build path
            path_parts = [p for p in [tier1, tier2, tier3, tier4] if p]

            # Depth = number of non-empty tier columns
            depth = len(path_parts)

            rows.append({
                "id": uid,
                "parent_id": parent,
                "name": name,
                "display_name": display_name,
                "tier1": tier1,
                "tier2": tier2,
                "tier3": tier3,
                "tier4": tier4,
                "path": " > ".join(path_parts),
                "depth": depth,
            })

    # Compute is_leaf and children_count from parent relationships
    parent_ids = set(r["parent_id"] for r in rows if r["parent_id"])
    for row in rows:
        row["is_leaf"] = row["id"] not in parent_ids
        row["children_count"] = sum(1 for r in rows if r["parent_id"] == row["id"])

    leaf_count = sum(1 for r in rows if r["is_leaf"])
    parent_count = len(rows) - leaf_count
    print(f"  Total nodes: {len(rows)} ({leaf_count} leaves, {parent_count} parents)")

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "parent_id", "name", "display_name",
            "tier1", "tier2", "tier3", "tier4", "path",
            "depth", "is_leaf", "children_count",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} categories to {output_path}")
    return rows


def build_corpus(scraped_path, labeled_path):
    """
    Build deduplicated corpus from scraped and labeled data.

    Deduplicates by URL first, then by content hash.
    Returns (corpus_pages, stats).
    """
    pages_by_url = {}
    content_hashes = {}
    stats = {
        "scraped_input": 0,
        "labeled_input": 0,
        "url_dupes": 0,
        "content_dupes": 0,
        "corpus_output": 0,
    }

    # Load scraped pages first (base content)
    if os.path.exists(scraped_path):
        scraped = load_json(scraped_path)
        stats["scraped_input"] = len(scraped)
        for page in scraped:
            url = page.get("url", "")
            if not url or not page.get("text"):
                continue
            if url in pages_by_url:
                stats["url_dupes"] += 1
                continue
            text_hash = hashlib.md5(page["text"].encode("utf-8")).hexdigest()
            if text_hash in content_hashes:
                stats["content_dupes"] += 1
                continue
            content_hashes[text_hash] = url
            pages_by_url[url] = {
                "url": url,
                "title": page.get("title", ""),
                "text": page["text"],
                "word_count": page.get("word_count", len(page["text"].split())),
                "content_hash": text_hash,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }

    # Merge any labeled-only pages (pages that exist in labeled but not scraped)
    if os.path.exists(labeled_path):
        labeled = load_json(labeled_path)
        stats["labeled_input"] = len(labeled)
        for page in labeled:
            url = page.get("url", "")
            if not url or not page.get("text"):
                continue
            if url in pages_by_url:
                continue  # Already in corpus
            text_hash = hashlib.md5(page["text"].encode("utf-8")).hexdigest()
            if text_hash in content_hashes:
                stats["content_dupes"] += 1
                continue
            content_hashes[text_hash] = url
            pages_by_url[url] = {
                "url": url,
                "title": page.get("title", ""),
                "text": page["text"],
                "word_count": page.get("word_count", len(page["text"].split())),
                "content_hash": text_hash,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }

    corpus = list(pages_by_url.values())
    stats["corpus_output"] = len(corpus)
    return corpus, stats


def build_labels(labeled_path, corpus_urls):
    """
    Extract taxonomy-specific labels from iab_forge labeled data.

    Only includes labels for pages that exist in the corpus.
    Returns (labels, stats).
    """
    stats = {
        "input": 0,
        "in_corpus": 0,
        "no_categories": 0,
        "output": 0,
    }

    labels = []
    if not os.path.exists(labeled_path):
        return labels, stats

    labeled = load_json(labeled_path)
    stats["input"] = len(labeled)

    for page in labeled:
        url = page.get("url", "")
        categories = page.get("categories", [])

        if url not in corpus_urls:
            continue
        stats["in_corpus"] += 1

        if not categories:
            stats["no_categories"] += 1
            continue

        labels.append({
            "url": url,
            "categories": categories,
            "reasoning": page.get("reasoning", ""),
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "review_status": page.get("review_action", "unreviewed"),
        })
        stats["output"] += 1

    return labels, stats


def main():
    parser = argparse.ArgumentParser(description="Migrate iab_forge data to classivore")
    parser.add_argument(
        "--iab-forge-dir",
        default=os.path.expanduser("~/Developer/iab_forge"),
        help="Path to iab_forge project (default: ~/Developer/iab_forge)",
    )
    parser.add_argument(
        "--iab-tsv",
        default=os.path.expanduser(
            "~/Developer/Taxonomies/Content Taxonomies/Content Taxonomy 2.2.tsv"
        ),
        help="Path to official IAB 2.2 TSV file",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Path to classivore project root (default: this project)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write files")
    args = parser.parse_args()

    forge_dir = Path(args.iab_forge_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Migrating iab_forge data to classivore")
    print("=" * 70)
    print(f"  Source: {forge_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Step 1: Create taxonomy.csv from official IAB TSV
    print("[1/3] Creating taxonomy.csv from official IAB 2.2 TSV...")
    iab_tsv = Path(args.iab_tsv)
    taxonomy_csv = output_dir / "taxonomies" / "iab_2.2" / "taxonomy.csv"
    if not iab_tsv.exists():
        print(f"  WARNING: IAB TSV not found at {iab_tsv}")
        print("  Skipping taxonomy creation. Download from:")
        print("  https://github.com/InteractiveAdvertisingBureau/Taxonomies")
    elif not args.dry_run:
        categories = create_taxonomy_csv(str(iab_tsv), str(taxonomy_csv))
    else:
        print(f"  [DRY RUN] Would create {taxonomy_csv}")

    # Step 2: Build shared corpus
    print()
    print("[2/3] Building shared corpus (deduplicated by URL + content)...")
    scraped_path = forge_dir / "data" / "scraped_pages.json"
    labeled_path = forge_dir / "data" / "labeled_data.json"
    corpus_path = output_dir / "data" / "corpus" / "pages.json"

    corpus, corpus_stats = build_corpus(str(scraped_path), str(labeled_path))
    print(f"  Scraped input:    {corpus_stats['scraped_input']:>7}")
    print(f"  Labeled input:    {corpus_stats['labeled_input']:>7}")
    print(f"  URL duplicates:   {corpus_stats['url_dupes']:>7}")
    print(f"  Content dupes:    {corpus_stats['content_dupes']:>7}")
    print(f"  Corpus output:    {corpus_stats['corpus_output']:>7}")

    if not args.dry_run:
        save_json(corpus, str(corpus_path))

    # Step 3: Extract IAB labels
    print()
    print("[3/3] Extracting IAB 2.2 labels...")
    labels_path = output_dir / "data" / "labels" / "iab-2.2.json"
    corpus_urls = {p["url"] for p in corpus}

    # Use reviewed data if available, otherwise labeled
    reviewed_path = forge_dir / "data" / "reviewed_data.json"
    source_path = str(reviewed_path) if reviewed_path.exists() else str(labeled_path)
    print(f"  Source: {source_path}")

    labels, label_stats = build_labels(source_path, corpus_urls)
    print(f"  Input pages:      {label_stats['input']:>7}")
    print(f"  In corpus:        {label_stats['in_corpus']:>7}")
    print(f"  No categories:    {label_stats['no_categories']:>7}")
    print(f"  Labels output:    {label_stats['output']:>7}")

    if not args.dry_run:
        save_json(labels, str(labels_path))

    # Summary
    print()
    print("=" * 70)
    print("Migration complete" if not args.dry_run else "DRY RUN complete")
    print("=" * 70)
    print(f"  Corpus:     {corpus_stats['corpus_output']} pages in data/corpus/pages.json")
    print(f"  IAB Labels: {label_stats['output']} labels in data/labels/iab-2.2.json")
    print(f"  Taxonomy:   taxonomies/iab_2.2/taxonomy.csv")
    print()
    if not args.dry_run:
        print("Next steps:")
        print("  1. classivore enrich --taxonomy iab-2.2")
        print("  2. classivore validate --taxonomy iab-2.2")
        print("  3. classivore train --taxonomy iab-2.2")
    else:
        print("Run without --dry-run to write files.")


if __name__ == "__main__":
    main()
