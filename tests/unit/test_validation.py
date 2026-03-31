#!/usr/bin/env python3
"""Tests for the validation module."""

import json
from pathlib import Path

import pytest

from classivore.validation.formatter import format_report
from classivore.validation.loader import load_labeled_data, load_scraped_data
from classivore.validation.runner import (
    _check_confidence,
    _check_taxonomy,
    validate_labeled,
    validate_scraped,
)


@pytest.fixture
def corpus_pages():
    """Sample corpus pages."""
    return [
        {
            "url": f"https://example.com/page{i}",
            "title": f"Page {i}",
            "text": f"This is the content of page {i} about topic {i % 5}. " * 20,
            "word_count": 200,
            "content_hash": f"hash{i}",
            "collected_at": "2026-03-31T12:00:00Z",
        }
        for i in range(100)
    ]


@pytest.fixture
def label_entries():
    """Sample label entries with multi-label categories."""
    categories = [
        "Automotive",
        "Technology",
        "Science",
        "Sports",
        "Entertainment",
    ]
    entries = []
    for i in range(100):
        cat = categories[i % len(categories)]
        entries.append({
            "url": f"https://example.com/page{i}",
            "categories": [
                {"category": cat, "confidence": 0.85 + (i % 10) * 0.01},
            ],
            "reasoning": f"This page is about {cat}.",
            "labeled_at": "2026-03-31T14:00:00Z",
            "review_status": "unreviewed",
        })
    # Add a few multi-label entries
    entries[0]["categories"].append(
        {"category": "Science", "confidence": 0.72}
    )
    entries[1]["categories"].append(
        {"category": "Sports", "confidence": 0.65}
    )
    return entries


@pytest.fixture
def data_dir(tmp_path, corpus_pages, label_entries):
    """Create a temporary data directory with corpus and labels."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    with open(corpus_dir / "pages.json", "w") as f:
        json.dump(corpus_pages, f)

    with open(labels_dir / "test-taxonomy.json", "w") as f:
        json.dump(label_entries, f)

    return tmp_path


class TestLoader:
    def test_load_labeled_data(self, data_dir):
        df = load_labeled_data(data_dir, "test-taxonomy")
        # 100 single-label + 2 extra labels from multi-label entries
        assert len(df) == 102
        assert set(df.columns) == {"url", "text", "label", "confidence", "review_status"}

    def test_load_labeled_data_missing_corpus(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Corpus not found"):
            load_labeled_data(tmp_path, "test-taxonomy")

    def test_load_labeled_data_missing_labels(self, tmp_path):
        (tmp_path / "corpus").mkdir()
        with open(tmp_path / "corpus" / "pages.json", "w") as f:
            json.dump([], f)
        with pytest.raises(FileNotFoundError, match="Labels not found"):
            load_labeled_data(tmp_path, "test-taxonomy")

    def test_load_scraped_data(self, data_dir):
        df = load_scraped_data(data_dir)
        assert len(df) == 100
        assert "url" in df.columns
        assert "text" in df.columns

    def test_load_scraped_data_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scraped_data(tmp_path)


class TestTaxonomyChecks:
    def test_unknown_labels(self):
        import pandas as pd

        df = pd.DataFrame({
            "label": ["A", "B", "C", "UNKNOWN"],
            "confidence": [0.9, 0.8, 0.7, 0.6],
        })
        result = _check_taxonomy(df, taxonomy_categories=["A", "B", "C"])
        findings = result["findings"]
        assert any("not in taxonomy" in f["message"] for f in findings)

    def test_missing_coverage(self):
        import pandas as pd

        df = pd.DataFrame({
            "label": ["A", "A", "B"],
            "confidence": [0.9, 0.8, 0.7],
        })
        result = _check_taxonomy(df, taxonomy_categories=["A", "B", "C", "D", "E"])
        findings = result["findings"]
        assert any("no training data" in f["message"] for f in findings)

    def test_no_taxonomy_categories(self):
        import pandas as pd

        df = pd.DataFrame({"label": ["A"], "confidence": [0.9]})
        result = _check_taxonomy(df, taxonomy_categories=None)
        assert result["findings"] == []

    def test_thin_categories(self):
        import pandas as pd

        labels = ["A"] * 50 + ["B"] * 3
        df = pd.DataFrame({
            "label": labels,
            "confidence": [0.9] * len(labels),
        })
        result = _check_taxonomy(df, taxonomy_categories=["A", "B"])
        findings = result["findings"]
        assert any("fewer than 10" in f["message"] for f in findings)


class TestConfidenceChecks:
    def test_low_confidence_warning(self):
        import pandas as pd

        df = pd.DataFrame({
            "confidence": [0.3] * 30 + [0.9] * 70,
        })
        result = _check_confidence(df)
        findings = result["findings"]
        assert any("below 0.5" in f["message"] for f in findings)
        assert findings[0]["severity"] == "Warning"

    def test_healthy_confidence(self):
        import pandas as pd

        df = pd.DataFrame({
            "confidence": [0.85] * 100,
        })
        result = _check_confidence(df)
        findings = result["findings"]
        assert findings[0]["severity"] == "Info"


class TestRunner:
    def test_validate_labeled(self, data_dir):
        report = validate_labeled(
            data_dir, "test-taxonomy", skip_noise=True,
        )
        assert report["overall_severity"] in ("Critical", "Warning", "Info")
        assert "findings" in report
        assert "recommendations" in report
        assert report["total_pages"] == 100
        assert report["total_labels"] == 102

    def test_validate_labeled_with_taxonomy(self, data_dir):
        report = validate_labeled(
            data_dir,
            "test-taxonomy",
            taxonomy_categories=["Automotive", "Technology", "Science", "Sports", "Entertainment"],
            skip_noise=True,
        )
        assert report["overall_severity"] in ("Critical", "Warning", "Info")
        # All categories covered, no unknowns
        taxonomy_findings = [
            f for f in report["findings"] if f["category"] == "Taxonomy"
        ]
        assert not any("not in taxonomy" in f["message"] for f in taxonomy_findings)

    def test_validate_scraped(self, data_dir):
        report = validate_scraped(data_dir)
        assert report["total_pages"] == 100
        assert "findings" in report


class TestFormatter:
    def test_format_report_with_color(self):
        report = {
            "overall_severity": "Warning",
            "taxonomy_slug": "test",
            "total_pages": 100,
            "total_labels": 200,
            "summary": {
                "num_classes": 5,
                "imbalance_ratio": 3.5,
                "exact_duplicates": 2,
                "near_duplicates": 5,
                "suspect_mislabels": 3,
            },
            "findings": [
                {"category": "Distribution", "severity": "Info", "message": "OK"},
                {"category": "Duplicates", "severity": "Warning", "message": "Found dups"},
            ],
            "recommendations": ["Fix the dups"],
        }
        output = format_report(report, use_color=True)
        assert "Data Quality Report" in output
        assert "Warning" in output
        assert "Fix the dups" in output

    def test_format_report_no_color(self):
        report = {
            "overall_severity": "Info",
            "total_pages": 50,
            "findings": [
                {"category": "Corpus", "severity": "Info", "message": "All good"},
            ],
            "recommendations": [],
        }
        output = format_report(report, use_color=False)
        assert "\033[" not in output
        assert "All good" in output
