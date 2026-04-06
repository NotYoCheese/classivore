#!/usr/bin/env python3
"""Tests for training dataset loading and preparation."""

import json

import numpy as np
import pytest

from classivore.training.dataset import (
    LEGACY_CONFIDENCE_WEIGHT,
    LEGACY_REASONING,
    ClassificationDataset,
    load_training_data,
    split_data,
)


def _make_config(tmp_path):
    """Create a minimal TaxonomyConfig-like object."""
    from unittest.mock import MagicMock

    # Create a minimal taxonomy CSV
    tax_dir = tmp_path / "taxonomies" / "test"
    tax_dir.mkdir(parents=True)
    csv_path = tax_dir / "taxonomy.csv"
    csv_path.write_text(
        "id,name,parent_id,is_leaf\n"
        "1,Automotive,,false\n"
        "2,Sedan,1,true\n"
        "3,SUV,1,true\n"
        "4,Technology,,false\n"
        "5,Python,4,true\n"
    )

    config = MagicMock()
    config.slug = "test-tax"
    config.taxonomy_file = csv_path
    config.id_column = "id"
    config.name_column = "name"
    config.parent_column = "parent_id"
    config.description_column = None
    config.excluded_categories = []
    return config


def _write_corpus(data_dir, pages):
    """Write corpus NDJSON."""
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    with open(corpus_dir / "pages.json", "w") as f:
        for p in pages:
            f.write(json.dumps(p) + "\n")


def _write_label_state(data_dir, slug, pages_dict):
    """Write label_state.json."""
    labels_dir = data_dir / "labels" / slug
    labels_dir.mkdir(parents=True)
    state = {
        "started_at": "2026-04-01T00:00:00Z",
        "last_checkpoint_at": "2026-04-01T00:00:00Z",
        "stage1_batch_ids": [],
        "stage2_batch_ids": [],
        "stats": {},
        "pages": pages_dict,
    }
    with open(labels_dir / "label_state.json", "w") as f:
        json.dump(state, f)


class TestLoadTrainingData:
    def test_basic_loading(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [
            {"content_hash": "h1", "text": "A great sedan review article."},
            {"content_hash": "h2", "text": "Python programming tutorial."},
        ])

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "analyzed",
                "labels": [{"name": "Sedan", "confidence": 0.92}],
                "error": None,
            },
            "h2": {
                "url": "http://b.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "analyzed",
                "labels": [{"name": "Python", "confidence": 0.85}],
                "error": None,
            },
        })

        data = load_training_data(config, data_dir)

        assert len(data["texts"]) == 2
        assert data["label_matrix"].shape == (2, 3)  # 3 leaf categories
        assert data["stats"]["total_pages"] == 2
        assert "Sedan" in data["label_names"]
        assert "Python" in data["label_names"]
        # Non-leaf categories should NOT be in label_names
        assert "Automotive" not in data["label_names"]
        assert "Technology" not in data["label_names"]

    def test_non_leaf_labels_excluded(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [
            {"content_hash": "h1", "text": "General automotive content."},
        ])

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "analyzed",
                "labels": [
                    {"name": "Automotive", "confidence": 0.9},  # Non-leaf
                    {"name": "Sedan", "confidence": 0.8},       # Leaf
                ],
                "error": None,
            },
        })

        data = load_training_data(config, data_dir)

        # Should include the page (has leaf label)
        assert len(data["texts"]) == 1
        # Only leaf label should be in matrix
        sedan_idx = data["label_to_index"]["Sedan"]
        assert data["label_matrix"][0, sedan_idx] == 1.0

    def test_legacy_labels_get_discount(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [
            {"content_hash": "h1", "text": "Legacy labeled content."},
            {"content_hash": "h2", "text": "New pipeline content."},
        ])

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None,
                "reasoning": LEGACY_REASONING,  # Legacy
                "labels": [{"name": "Sedan", "confidence": 1.0}],
                "error": None,
            },
            "h2": {
                "url": "http://b.com", "status": "stage2_complete",
                "tier1_categories": None,
                "reasoning": "analyzed by stage 2",  # Current pipeline
                "labels": [{"name": "SUV", "confidence": 0.95}],
                "error": None,
            },
        })

        data = load_training_data(config, data_dir)

        sedan_idx = data["label_to_index"]["Sedan"]
        suv_idx = data["label_to_index"]["SUV"]

        # Legacy page should have discounted confidence
        assert data["confidence_matrix"][0, sedan_idx] == LEGACY_CONFIDENCE_WEIGHT
        # Current pipeline page should have real confidence
        assert data["confidence_matrix"][1, suv_idx] == 0.95
        # Stats should track legacy vs pipeline
        assert data["stats"]["legacy_pages"] == 1
        assert data["stats"]["pipeline_pages"] == 1

    def test_skips_incomplete_pages(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [
            {"content_hash": "h1", "text": "Complete page."},
            {"content_hash": "h2", "text": "Incomplete page."},
        ])

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "ok",
                "labels": [{"name": "Sedan", "confidence": 0.9}],
                "error": None,
            },
            "h2": {
                "url": "http://b.com", "status": "stage1_complete",
                "tier1_categories": [{"name": "Automotive", "confidence": 0.8}],
                "labels": None, "reasoning": None, "error": None,
            },
        })

        data = load_training_data(config, data_dir)
        assert len(data["texts"]) == 1

    def test_skips_pages_without_corpus_text(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [])  # Empty corpus

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "ok",
                "labels": [{"name": "Sedan", "confidence": 0.9}],
                "error": None,
            },
        })

        data = load_training_data(config, data_dir)
        assert len(data["texts"]) == 0
        assert data["stats"]["skipped_no_text"] == 1

    def test_multi_label_page(self, tmp_path):
        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"

        _write_corpus(data_dir, [
            {"content_hash": "h1", "text": "Self-driving Python cars."},
        ])

        _write_label_state(data_dir, "test-tax", {
            "h1": {
                "url": "http://a.com", "status": "stage2_complete",
                "tier1_categories": None, "reasoning": "analyzed",
                "labels": [
                    {"name": "Sedan", "confidence": 0.9},
                    {"name": "Python", "confidence": 0.7},
                ],
                "error": None,
            },
        })

        data = load_training_data(config, data_dir)

        sedan_idx = data["label_to_index"]["Sedan"]
        python_idx = data["label_to_index"]["Python"]

        assert data["label_matrix"][0, sedan_idx] == 1.0
        assert data["label_matrix"][0, python_idx] == 1.0
        assert data["confidence_matrix"][0, sedan_idx] == 0.9
        assert data["confidence_matrix"][0, python_idx] == 0.7


class TestSplitData:
    def _make_data(self, n=100):
        return {
            "texts": [f"text {i}" for i in range(n)],
            "label_matrix": np.eye(10, dtype=np.float32)[np.arange(n) % 10],
            "confidence_matrix": np.ones((n, 10), dtype=np.float32),
        }

    def test_split_ratios(self):
        data = self._make_data(100)
        splits = split_data(data, train_ratio=0.7, val_ratio=0.2)

        # Allow ±1 for integer rounding
        assert 69 <= len(splits["train"]["texts"]) <= 71
        assert 19 <= len(splits["val"]["texts"]) <= 21
        assert 9 <= len(splits["test"]["texts"]) <= 11

    def test_no_overlap(self):
        data = self._make_data(100)
        splits = split_data(data)

        train_set = set(splits["train"]["indices"])
        val_set = set(splits["val"]["indices"])
        test_set = set(splits["test"]["indices"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_covers_all_data(self):
        data = self._make_data(100)
        splits = split_data(data)

        all_indices = set(splits["train"]["indices"]) | set(splits["val"]["indices"]) | set(splits["test"]["indices"])
        assert all_indices == set(range(100))

    def test_reproducible(self):
        data = self._make_data(100)
        s1 = split_data(data, seed=42)
        s2 = split_data(data, seed=42)

        assert list(s1["train"]["indices"]) == list(s2["train"]["indices"])

    def test_different_seed_different_split(self):
        data = self._make_data(100)
        s1 = split_data(data, seed=42)
        s2 = split_data(data, seed=99)

        assert list(s1["train"]["indices"]) != list(s2["train"]["indices"])


class TestClassificationDataset:
    def test_len(self):
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        ds = ClassificationDataset(
            texts=["a", "b", "c"],
            label_matrix=np.zeros((3, 5)),
            confidence_matrix=np.ones((3, 5)),
            tokenizer=tokenizer,
        )
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        ds = ClassificationDataset(
            texts=["Hello world this is a test."],
            label_matrix=np.array([[1.0, 0.0, 0.0]]),
            confidence_matrix=np.array([[0.9, 1.0, 1.0]]),
            tokenizer=tokenizer,
            max_length=32,
        )

        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "confidence_weights" in item
        assert item["input_ids"].shape[0] == 32
        assert item["labels"].shape[0] == 3
