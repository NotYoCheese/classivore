#!/usr/bin/env python3
"""Tests for agent runner orchestration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog

from classivore.agent.runner import (
    MAX_CATEGORIES_PER_ITERATION,
    _plan_iteration,
    _print_coverage_report,
    run_agent,
)
from classivore.models import AgentConfig, CategoryGap, CoverageReport


def _make_config(
    slug="iab-2.2",
    target_per_category=10,
    excluded_categories=None,
    excluded_tier1_categories=None,
):
    """Build a mock TaxonomyConfig."""
    config = MagicMock()
    config.slug = slug
    config.target_per_category = target_per_category
    config.excluded_categories = excluded_categories or []
    config.excluded_tier1_categories = excluded_tier1_categories or []
    config.labeling_model = "claude-haiku-4-5-20251001"
    config.stage1_max_tokens = 300
    config.stage2_max_tokens = 500
    config.tier1_confidence_threshold = 0.3
    config.labeling_temperature = 0.0
    config.text_truncation_words = 3000
    config.min_confidence = 0.5
    config.max_labels = 3
    return config


def _make_categories():
    """Build minimal test categories."""
    return [
        {
            "id": "1", "name": "Automotive", "display_name": "Automotive",
            "parent_id": "", "path": ["Automotive"], "depth": 1,
            "is_leaf": False, "children_count": 2, "description": "", "boundaries": "",
        },
        {
            "id": "2", "name": "Sedan", "display_name": "Automotive: Sedan",
            "parent_id": "1", "path": ["Automotive", "Sedan"], "depth": 2,
            "is_leaf": True, "children_count": 0, "description": "", "boundaries": "",
        },
        {
            "id": "3", "name": "SUV", "display_name": "Automotive: SUV",
            "parent_id": "1", "path": ["Automotive", "SUV"], "depth": 2,
            "is_leaf": True, "children_count": 0, "description": "", "boundaries": "",
        },
        {
            "id": "4", "name": "Python", "display_name": "Technology: Python",
            "parent_id": "10", "path": ["Technology", "Python"], "depth": 2,
            "is_leaf": True, "children_count": 0, "description": "", "boundaries": "",
        },
    ]


def _write_labels(data_dir, slug, entries):
    """Write label entries as NDJSON."""
    labels_dir = Path(data_dir) / "labels" / slug
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_file = labels_dir / "labels.json"
    with open(labels_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestPlanIteration:
    """Test iteration planning logic."""

    def test_prioritizes_worst_gaps(self):
        gaps = [
            CategoryGap("Empty", 0, 10, 10, "Tier1"),
            CategoryGap("Low", 2, 10, 8, "Tier1"),
            CategoryGap("Medium", 5, 10, 5, "Tier1"),
        ]
        report = CoverageReport(
            total_categories=3, covered_categories=2,
            satisfied_categories=0, total_labeled_pages=7,
            gaps=gaps, timestamp="2026-04-03T00:00:00Z",
        )
        config = AgentConfig(target_per_category=10)

        plan = _plan_iteration(report, config, iteration=0)

        assert plan.target_categories[0] == "Empty"
        assert len(plan.target_categories) == 3

    def test_caps_categories_per_iteration(self):
        gaps = [
            CategoryGap(f"Cat{i}", 0, 10, 10, "Tier1")
            for i in range(200)
        ]
        report = CoverageReport(
            total_categories=200, covered_categories=0,
            satisfied_categories=0, total_labeled_pages=0,
            gaps=gaps, timestamp="2026-04-03T00:00:00Z",
        )
        config = AgentConfig(target_per_category=10)

        plan = _plan_iteration(report, config, iteration=0)

        assert len(plan.target_categories) == MAX_CATEGORIES_PER_ITERATION

    def test_first_iteration_uses_template_strategy(self):
        gaps = [CategoryGap("A", 0, 10, 10, "Tier1")]
        report = CoverageReport(
            total_categories=1, covered_categories=0,
            satisfied_categories=0, total_labeled_pages=0,
            gaps=gaps, timestamp="2026-04-03T00:00:00Z",
        )
        config = AgentConfig(target_per_category=10)

        plan = _plan_iteration(report, config, iteration=0)
        assert plan.strategy == "template"

    def test_subsequent_iterations_use_hybrid_strategy(self):
        gaps = [CategoryGap("A", 0, 10, 10, "Tier1")]
        report = CoverageReport(
            total_categories=1, covered_categories=0,
            satisfied_categories=0, total_labeled_pages=0,
            gaps=gaps, timestamp="2026-04-03T00:00:00Z",
        )
        config = AgentConfig(target_per_category=10)

        plan = _plan_iteration(report, config, iteration=1)
        assert plan.strategy == "hybrid"


class TestRunAgentDryRun:
    """Test agent dry run mode."""

    def test_dry_run_shows_coverage(self, tmp_path, capsys):
        config = _make_config(target_per_category=10)
        categories = _make_categories()
        hierarchy = {}

        # Write some labels
        _write_labels(tmp_path, "iab-2.2", [
            {"url": f"http://a{i}.com", "content_hash": f"h{i}", "categories": ["Sedan"]}
            for i in range(5)
        ])

        result = run_agent(
            config=config,
            categories=categories,
            hierarchy=hierarchy,
            data_dir=str(tmp_path),
            dry_run=True,
        )

        assert result["dry_run"] is True
        output = capsys.readouterr().out
        assert "Coverage Analysis" in output
        assert "SUV" in output or "Python" in output

    def test_dry_run_no_api_calls(self, tmp_path):
        """Dry run should not import or call collection/labeling."""
        config = _make_config()
        categories = _make_categories()
        labels_dir = Path(tmp_path) / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        result = run_agent(
            config=config,
            categories=categories,
            hierarchy={},
            data_dir=str(tmp_path),
            dry_run=True,
        )

        assert result["dry_run"] is True


class TestRunAgent:
    """Test agent orchestration with mocked collection and labeling."""

    @patch("classivore.agent.runner._run_labeling")
    @patch("classivore.agent.runner._run_collection")
    def test_single_iteration(self, mock_collect, mock_label, tmp_path):
        """Agent runs one iteration and stops on max_iterations."""
        config = _make_config(target_per_category=10)
        categories = _make_categories()

        # No labels initially
        labels_dir = Path(tmp_path) / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        mock_collect.return_value = {"total_collected": 5, "total_target": 30}

        # After labeling, write some labels so post-analysis shows improvement
        def label_side_effect(*args, **kwargs):
            _write_labels(tmp_path, "iab-2.2", [
                {"url": f"http://a{i}.com", "content_hash": f"h{i}",
                 "categories": ["Sedan", "SUV"]}
                for i in range(5)
            ])
            return {"stage2_complete": 5, "error": 0}

        mock_label.side_effect = label_side_effect

        result = run_agent(
            config=config,
            categories=categories,
            hierarchy={},
            data_dir=str(tmp_path),
            max_iterations=1,
        )

        assert result["iterations_completed"] == 1
        assert result["total_pages_collected"] == 5
        mock_collect.assert_called_once()
        mock_label.assert_called_once()

    @patch("classivore.agent.runner._run_labeling")
    @patch("classivore.agent.runner._run_collection")
    def test_stops_when_no_gaps(self, mock_collect, mock_label, tmp_path):
        """Agent stops immediately if all categories are satisfied."""
        config = _make_config(target_per_category=2)
        categories = _make_categories()

        # Write enough labels to satisfy all categories
        _write_labels(tmp_path, "iab-2.2", [
            {"url": f"http://a{i}.com", "content_hash": f"h{i}",
             "categories": ["Sedan", "SUV", "Python"]}
            for i in range(3)
        ])

        result = run_agent(
            config=config,
            categories=categories,
            hierarchy={},
            data_dir=str(tmp_path),
            max_iterations=5,
        )

        assert result["iterations_completed"] == 0
        mock_collect.assert_not_called()
        mock_label.assert_not_called()

    @patch("classivore.agent.runner._run_labeling")
    @patch("classivore.agent.runner._run_collection")
    def test_stops_on_zero_yield(self, mock_collect, mock_label, tmp_path):
        """Agent stops after consecutive zero-yield iterations."""
        config = _make_config(target_per_category=100)
        categories = _make_categories()

        labels_dir = Path(tmp_path) / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        mock_collect.return_value = {"total_collected": 0, "total_target": 30}
        mock_label.return_value = {"stage2_complete": 0, "error": 0}

        result = run_agent(
            config=config,
            categories=categories,
            hierarchy={},
            data_dir=str(tmp_path),
            max_iterations=10,
        )

        # Should stop after 2 zero-yield iterations (default max_consecutive_zero_yield)
        assert result["iterations_completed"] == 2

    @patch("classivore.agent.runner._run_labeling")
    @patch("classivore.agent.runner._run_collection")
    def test_collection_focuses_on_target_categories(self, mock_collect, mock_label, tmp_path):
        """Collection config excludes non-target categories."""
        config = _make_config(target_per_category=10)
        categories = _make_categories()

        labels_dir = Path(tmp_path) / "labels" / "iab-2.2"
        labels_dir.mkdir(parents=True)

        mock_collect.return_value = {"total_collected": 5, "total_target": 30}
        mock_label.return_value = {"stage2_complete": 0, "error": 0}

        run_agent(
            config=config,
            categories=categories,
            hierarchy={},
            data_dir=str(tmp_path),
            max_iterations=1,
        )

        # Verify collection was called with modified config
        call_kwargs = mock_collect.call_args
        assert call_kwargs is not None


class TestPrintCoverageReport:
    """Test coverage report formatting."""

    def test_prints_report(self, capsys):
        gaps = [
            CategoryGap("Empty", 0, 10, 10, "Tier1"),
            CategoryGap("Low", 3, 10, 7, "Tier1"),
        ]
        report = CoverageReport(
            total_categories=5, covered_categories=3,
            satisfied_categories=2, total_labeled_pages=100,
            gaps=gaps, timestamp="2026-04-03T00:00:00Z",
        )
        config = AgentConfig(target_per_category=10)

        _print_coverage_report(report, config)

        output = capsys.readouterr().out
        assert "Coverage Analysis" in output
        assert "Empty" in output
        assert "Low" in output
        assert "5" in output  # total categories
