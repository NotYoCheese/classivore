#!/usr/bin/env python3
"""Load classivore corpus + labels into a DataFrame for validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_labeled_data(data_dir: Path, taxonomy_slug: str) -> pd.DataFrame:
    """Join corpus pages with taxonomy labels into a flat DataFrame.

    Returns a DataFrame with columns: url, text, label, confidence, review_status.
    Multi-label entries are exploded into one row per label.
    """
    corpus_path = data_dir / "corpus" / "pages.json"
    labels_path = data_dir / "labels" / f"{taxonomy_slug}.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with open(corpus_path) as f:
        corpus = {page["url"]: page for page in json.load(f)}

    with open(labels_path) as f:
        labels = json.load(f)

    rows = []
    for entry in labels:
        url = entry["url"]
        page = corpus.get(url)
        if page is None:
            continue
        text = page.get("text", "")
        review_status = entry.get("review_status", "unreviewed")
        for cat in entry.get("categories", []):
            rows.append({
                "url": url,
                "text": text,
                "label": cat["category"],
                "confidence": cat.get("confidence", 0.0),
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

    with open(corpus_path) as f:
        pages = json.load(f)

    if not pages:
        raise ValueError("Corpus is empty.")

    return pd.DataFrame(pages)
