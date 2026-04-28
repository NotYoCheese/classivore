#!/usr/bin/env python3
"""Dataset loading and preparation for training.

Joins corpus text with label_state.json (authoritative source for labels
with confidence scores). Handles train/val/test splitting, label matrix
construction, and confidence weighting.

Key decisions:
- label_state.json is authoritative (has confidence + provenance)
- Non-leaf categories excluded from label matrix
- Legacy labels (reasoning == "seeded from existing labels") get 0.75 weight
- Category IDs sorted numerically to prevent string-sort bugs
"""

import json
from pathlib import Path

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Dataset

from classivore.logging_config import get_logger
from classivore.persistence import iter_ndjson, load_ndjson

logger = get_logger(__name__)

LEGACY_CONFIDENCE_WEIGHT = 0.75
LEGACY_REASONING = "seeded from existing labels"


def load_training_data(config, data_dir):
    """Load and join corpus text with labels for training.

    Args:
        config: TaxonomyConfig instance.
        data_dir: Path to data directory.

    Returns:
        Dict with keys:
            texts: list of strings
            label_matrix: numpy array (num_samples, num_classes) binary
            confidence_matrix: numpy array (num_samples, num_classes) weights
            label_names: list of leaf category names (ordered by index)
            label_to_index: dict mapping category name → index
            stats: dict with data statistics
    """
    data_dir = Path(data_dir)
    corpus_file = data_dir / "corpus" / "pages.json"
    state_file = data_dir / "labels" / config.slug / "label_state.json"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_file}")
    if not state_file.exists():
        raise FileNotFoundError(f"Label state not found: {state_file}")

    # Load taxonomy to identify leaf categories, excluding metadata tier-1s
    from classivore.taxonomy.loader import load_taxonomy
    categories = load_taxonomy(config)
    excluded_tier1 = set(getattr(config, "excluded_tier1_categories", []))
    excluded_cats = set(getattr(config, "excluded_categories", []))
    leaf_names = sorted(
        [
            c["name"] for c in categories
            if c["is_leaf"]
            and c["path"][0] not in excluded_tier1
            and c["display_name"] not in excluded_cats
        ],
        key=lambda n: next(int(c["id"]) for c in categories if c["name"] == n),
    )
    label_to_index = {name: i for i, name in enumerate(leaf_names)}
    num_classes = len(leaf_names)

    # Load corpus text keyed by content_hash
    corpus = {}
    for page in iter_ndjson(corpus_file):
        h = page.get("content_hash", "")
        if h:
            corpus[h] = page.get("text", "")

    # Load label state
    state = json.loads(state_file.read_text())
    pages = state.get("pages", {})

    # Join: build parallel lists
    texts = []
    label_matrix = []
    confidence_matrix = []
    skipped_no_text = 0
    skipped_no_labels = 0
    skipped_not_complete = 0
    legacy_count = 0
    pipeline_count = 0

    for content_hash, page in pages.items():
        if page.get("status") != "stage2_complete":
            skipped_not_complete += 1
            continue

        labels = page.get("labels") or []
        if not labels:
            skipped_no_labels += 1
            continue

        text = corpus.get(content_hash)
        if not text:
            skipped_no_text += 1
            continue

        # Build label and confidence vectors
        label_vec = np.zeros(num_classes, dtype=np.float32)
        conf_vec = np.ones(num_classes, dtype=np.float32)

        is_legacy = page.get("reasoning") == LEGACY_REASONING

        if is_legacy:
            legacy_count += 1
        else:
            pipeline_count += 1

        for lbl in labels:
            name = lbl.get("name", "")
            confidence = lbl.get("confidence", 1.0)

            idx = label_to_index.get(name)
            if idx is None:
                # Non-leaf or unknown category — skip silently
                continue

            label_vec[idx] = 1.0

            if is_legacy:
                conf_vec[idx] = LEGACY_CONFIDENCE_WEIGHT
            else:
                conf_vec[idx] = confidence

        # Only include if at least one leaf label
        if label_vec.sum() == 0:
            skipped_no_labels += 1
            continue

        texts.append(text)
        label_matrix.append(label_vec)
        confidence_matrix.append(conf_vec)

    label_matrix = np.array(label_matrix)
    confidence_matrix = np.array(confidence_matrix)

    # Compute per-class sample counts
    label_counts = label_matrix.sum(axis=0)

    stats = {
        "total_pages": len(texts),
        "num_classes": num_classes,
        "legacy_pages": legacy_count,
        "pipeline_pages": pipeline_count,
        "skipped_no_text": skipped_no_text,
        "skipped_no_labels": skipped_no_labels,
        "skipped_not_complete": skipped_not_complete,
        "categories_with_labels": int((label_counts > 0).sum()),
        "categories_below_20": int((label_counts < 20).sum()),
        "min_samples": int(label_counts.min()) if label_counts.size > 0 else 0,
        "max_samples": int(label_counts.max()) if label_counts.size > 0 else 0,
        "median_samples": int(np.median(label_counts)) if label_counts.size > 0 else 0,
        "label_counts": label_counts,
    }

    logger.info(
        "training_data_loaded",
        total=len(texts),
        classes=num_classes,
        legacy=legacy_count,
        pipeline=pipeline_count,
        skipped_no_text=skipped_no_text,
    )

    return {
        "texts": texts,
        "label_matrix": label_matrix,
        "confidence_matrix": confidence_matrix,
        "label_names": leaf_names,
        "label_to_index": label_to_index,
        "stats": stats,
    }


def split_data(data, train_ratio=0.7, val_ratio=0.2, seed=42):
    """Multi-label stratified split into train/val/test.

    Uses iterative stratification (Sechidis et al. 2011) to keep each
    label's representation proportional across all three splits. A plain
    random split lands thin-tail categories with 0–2 test samples by
    chance, which makes per-category F1 statistically meaningless.

    Args:
        data: Dict from load_training_data.
        train_ratio: Fraction for training. Default 0.7.
        val_ratio: Fraction for validation. Default 0.2.
        seed: Random seed for reproducibility.

    Returns:
        Dict with train/val/test splits, each containing:
            texts, label_matrix, confidence_matrix, indices
    """
    n = len(data["texts"])
    test_ratio = 1.0 - train_ratio - val_ratio
    Y = data["label_matrix"]
    X = np.arange(n).reshape(-1, 1)

    s1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_idx, test_idx = next(s1.split(X, Y))

    val_within = val_ratio / (train_ratio + val_ratio)
    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_within, random_state=seed
    )
    inner_train, inner_val = next(s2.split(X[trainval_idx], Y[trainval_idx]))
    train_idx = trainval_idx[inner_train]
    val_idx = trainval_idx[inner_val]

    def _subset(idx):
        return {
            "texts": [data["texts"][i] for i in idx],
            "label_matrix": data["label_matrix"][idx],
            "confidence_matrix": data["confidence_matrix"][idx],
            "indices": idx,
        }

    return {
        "train": _subset(train_idx),
        "val": _subset(val_idx),
        "test": _subset(test_idx),
    }


class ClassificationDataset(Dataset):
    """PyTorch Dataset for multi-label text classification."""

    def __init__(self, texts, label_matrix, confidence_matrix, tokenizer,
                 max_length=512):
        self.texts = texts
        self.label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
        self.confidence_matrix = torch.tensor(confidence_matrix, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.label_matrix[idx],
            "confidence_weights": self.confidence_matrix[idx],
        }
