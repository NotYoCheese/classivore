#!/usr/bin/env python3
"""Tests for agent coverage analysis."""

import json

import pytest

from classivore.agent.coverage import analyze_coverage


def _make_cat(name, tier1, is_leaf=True, depth=2):
    """Helper to build category dicts matching taxonomy loader format."""
    return {
        "id": str(hash(name) % 1000),
        "name": name,
        "display_name": f"{tier1}: {name}" if depth > 1 else name,
        "parent_id": str(hash(tier1) % 1000) if depth > 1 else "",
        "path": [tier1] if depth == 1 else [tier1, name],
        "depth": depth,
        "is_leaf": is_leaf,
        "children_count": 0 if is_leaf else 3,
        "description": "",
        "boundaries": "",
    }


def _write_labels(labels_dir, entries):
    """Write label entries as NDJSON."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_file = labels_dir / "labels.json"
    with open(labels_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestAnalyzeCoverage:
    """Test coverage analysis."""

    def test_empty_labels_all_gaps(self, tmp_path):
        """All categories are gaps when no labels exist."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("SUV", "Automotive"),
            _make_cat("Python", "Technology"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert report.total_categories == 3
        assert report.covered_categories == 0
        assert report.satisfied_categories == 0
        assert len(report.gaps) == 3
        assert report.coverage_pct == 0.0

    def test_partial_coverage(self, tmp_path):
        """Categories with some labels are partially covered."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("SUV", "Automotive"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        _write_labels(labels_dir, [
            {"url": "http://a.com", "content_hash": "h1", "categories": ["Sedan"]},
            {"url": "http://b.com", "content_hash": "h2", "categories": ["Sedan"]},
            {"url": "http://c.com", "content_hash": "h3", "categories": ["Sedan"]},
        ])

        report = analyze_coverage(cats, labels_dir, target_per_category=5)

        assert report.covered_categories == 1
        assert report.satisfied_categories == 0
        assert report.total_labeled_pages == 3
        # SUV has 0 labels, should be first gap
        assert report.gaps[0].name == "SUV"
        assert report.gaps[0].current_count == 0
        assert report.gaps[1].name == "Sedan"
        assert report.gaps[1].current_count == 3
        assert report.gaps[1].deficit == 2

    def test_satisfied_categories_excluded_from_gaps(self, tmp_path):
        """Categories meeting target are not in gaps list."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("SUV", "Automotive"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        _write_labels(labels_dir, [
            {"url": f"http://a{i}.com", "content_hash": f"h{i}", "categories": ["Sedan"]}
            for i in range(10)
        ])

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert report.satisfied_categories == 1
        assert len(report.gaps) == 1  # Only SUV
        assert report.gaps[0].name == "SUV"

    def test_excluded_categories_omitted(self, tmp_path):
        """Excluded categories don't appear in report."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("SUV", "Automotive"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        report = analyze_coverage(
            cats, labels_dir,
            target_per_category=10,
            excluded_categories={"Automotive: SUV"},
        )

        assert report.total_categories == 1
        assert len(report.gaps) == 1
        assert report.gaps[0].name == "Sedan"

    def test_excluded_tier1_omitted(self, tmp_path):
        """Categories under excluded tier-1 don't appear."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("Python", "Technology"),
            _make_cat("English", "Content Language"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        report = analyze_coverage(
            cats, labels_dir,
            target_per_category=10,
            excluded_tier1={"Content Language"},
        )

        assert report.total_categories == 2
        assert all(g.name != "English" for g in report.gaps)

    def test_sorted_by_count_ascending(self, tmp_path):
        """Gaps are sorted with fewest labels first."""
        cats = [
            _make_cat("A", "Tier1"),
            _make_cat("B", "Tier1"),
            _make_cat("C", "Tier1"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        _write_labels(labels_dir, [
            {"url": "http://1.com", "content_hash": "h1", "categories": ["C"]},
            {"url": "http://2.com", "content_hash": "h2", "categories": ["C"]},
            {"url": "http://3.com", "content_hash": "h3", "categories": ["C"]},
            {"url": "http://4.com", "content_hash": "h4", "categories": ["A"]},
        ])

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert report.gaps[0].name == "B"    # 0 labels
        assert report.gaps[1].name == "A"    # 1 label
        assert report.gaps[2].name == "C"    # 3 labels

    def test_non_leaf_excluded(self, tmp_path):
        """Non-leaf (parent) categories are not counted as gaps."""
        cats = [
            _make_cat("Automotive", "Automotive", is_leaf=False, depth=1),
            _make_cat("Sedan", "Automotive"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert report.total_categories == 1
        assert report.gaps[0].name == "Sedan"

    def test_multi_label_pages_counted(self, tmp_path):
        """Pages with multiple categories increment all of them."""
        cats = [
            _make_cat("Sedan", "Automotive"),
            _make_cat("Python", "Technology"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        _write_labels(labels_dir, [
            {"url": "http://a.com", "content_hash": "h1", "categories": ["Sedan", "Python"]},
        ])

        report = analyze_coverage(cats, labels_dir, target_per_category=5)

        assert report.gaps[0].current_count == 1
        assert report.gaps[1].current_count == 1
        assert report.total_labeled_pages == 1

    def test_worst_gaps_property(self, tmp_path):
        """worst_gaps returns top 50."""
        cats = [_make_cat(f"Cat{i}", "Tier1") for i in range(100)]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert len(report.worst_gaps) == 50
        assert len(report.gaps) == 100

    def test_coverage_pct(self, tmp_path):
        """coverage_pct correctly computes percentage."""
        cats = [
            _make_cat("A", "Tier1"),
            _make_cat("B", "Tier1"),
            _make_cat("C", "Tier1"),
            _make_cat("D", "Tier1"),
        ]
        labels_dir = tmp_path / "labels" / "iab-2.2"
        _write_labels(labels_dir, [
            {"url": f"http://{i}.com", "content_hash": f"h{i}", "categories": ["A"]}
            for i in range(10)
        ])

        report = analyze_coverage(cats, labels_dir, target_per_category=10)

        assert report.coverage_pct == 25.0  # 1 of 4 satisfied
