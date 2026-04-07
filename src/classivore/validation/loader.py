#!/usr/bin/env python3
"""Load classivore corpus + labels into a DataFrame for validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classivore.persistence import load_ndjson


def load_labeled_data(data_dir: Path, taxonomy_slug: str) -> pd.DataFrame:
    """Join corpus pages with taxonomy labels into a flat DataFrame.

    Returns a DataFrame with columns: url, text, label, confidence, review_status.
    Multi-label entries are exploded into one row per label.
    """
    corpus_path = data_dir / "corpus" / "pages.json"

    # Support both old ({slug}.json) and new ({slug}/labels.json) label paths
    labels_path = data_dir / "labels" / taxonomy_slug / "labels.json"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / f"{taxonomy_slug}.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    corpus = {page["url"]: page for page in load_ndjson(corpus_path)}
    labels = load_ndjson(labels_path)

    rows = []
    for entry in labels:
        url = entry["url"]
        page = corpus.get(url)
        if page is None:
            continue
        text = page.get("text", "")
        review_status = entry.get("review_status", "unreviewed")
        for cat in entry.get("categories", []):
            # Handle both dict format ({"category": ..., "confidence": ...})
            # and plain string format from _write_labels
            if isinstance(cat, dict):
                label = cat.get("category", cat.get("name", ""))
                confidence = cat.get("confidence", 0.0)
            else:
                label = cat
                confidence = 1.0
            rows.append({
                "url": url,
                "text": text,
                "label": label,
                "confidence": confidence,
                "review_status": review_status,
            })

    if not rows:
        raise ValueError("No labeled data found after joining corpus and labels.")

    return pd.DataFrame(rows)


def load_scraped_data(data_dir: Path) -> pd.DataFrame:
    """Load scraped corpus pages (without labels) for basic quality checks.

    Returns a DataFrame with columns: url, text, word_count, content_hash.
    """
    corpus_path = data_dir / "corpus" / "pages.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    pages = load_ndjson(corpus_path)

    if not pages:
        raise ValueError("Corpus is empty.")

    return pd.DataFrame(pages)
