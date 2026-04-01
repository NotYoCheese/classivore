#!/usr/bin/env python3
"""Tests for collection orchestrator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from classivore.collection import run_collection


def _make_config():
    config = MagicMock()
    config.target_per_category = 2
    config.max_queries_per_category = 6
    config.max_per_domain_per_category = 50
    config.commoncrawl_crawl_id = None  # Skip CC for test speed
    config.query_model = "claude-haiku-4-5-20251001"
    config.excluded_categories = []
    return config


def _make_categories():
    return [
        {
            "id": "1", "name": "Automotive", "display_name": "Automotive: Automotive",
            "description": "Vehicles and cars.", "boundaries": "", "path": ["Automotive"],
            "depth": 1, "is_leaf": False, "children_count": 1,
        },
        {
            "id": "2", "name": "Sedan", "display_name": "Automotive: Sedan",
            "description": "Four-door cars.", "boundaries": "", "path": ["Automotive", "Sedan"],
            "depth": 2, "is_leaf": True, "children_count": 0,
        },
    ]


class TestRunCollection:
    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.search_brave")
    def test_collects_pages(self, mock_search, mock_fetch, tmp_path):
        """End-to-end: search → fetch → filter → save."""
        mock_search.return_value = [
            {"url": "https://example.com/sedan-article-one", "title": "A", "description": ""},
            {"url": "https://example.com/sedan-article-two", "title": "B", "description": ""},
        ]
        # Return enough unique words to pass filters
        words1 = " ".join(f"word{i}" for i in range(200))
        words2 = " ".join(f"other{i}" for i in range(200))
        mock_fetch.side_effect = [
            f"<html><body><p>{words1}</p></body></html>",
            f"<html><body><p>{words2}</p></body></html>",
        ]

        config = _make_config()
        categories = _make_categories()

        with patch("classivore.collection.extract_text") as mock_extract:
            mock_extract.side_effect = [words1, words2]
            summary = run_collection(
                config=config,
                categories=categories,
                data_dir=tmp_path,
                pages=2,
            )

        assert summary["total_collected"] == 2
        assert summary["satisfied_categories"] == 1

        # Verify corpus file was written
        corpus_file = tmp_path / "corpus" / "pages.json"
        assert corpus_file.exists()
        lines = corpus_file.read_text().strip().split("\n")
        assert len(lines) == 2
        page = json.loads(lines[0])
        assert "url" in page
        assert "text" in page
        assert "collected_at" in page

    @patch("classivore.collection.search_brave")
    def test_queries_only_mode(self, mock_search, tmp_path):
        """queries_only generates queries without fetching."""
        mock_search.return_value = []

        config = _make_config()
        categories = _make_categories()

        summary = run_collection(
            config=config,
            categories=categories,
            data_dir=tmp_path,
            queries_only=True,
        )

        # Should not have collected anything
        assert summary["total_collected"] == 0
        # Search should not have been called
        mock_search.assert_not_called()

    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.search_brave")
    def test_skips_blocked_urls(self, mock_search, mock_fetch, tmp_path):
        """URLs matching blocklist are skipped."""
        mock_search.return_value = [
            {"url": "https://example.com/shop/item", "title": "Shop", "description": ""},
            {"url": "https://example.com/great-sedan-article-here", "title": "Article", "description": ""},
        ]
        words = " ".join(f"word{i}" for i in range(200))
        mock_fetch.return_value = f"<html><body><p>{words}</p></body></html>"

        config = _make_config()
        config.target_per_category = 1
        categories = _make_categories()

        with patch("classivore.collection.extract_text") as mock_extract:
            mock_extract.return_value = words
            summary = run_collection(
                config=config,
                categories=categories,
                data_dir=tmp_path,
                pages=1,
            )

        # Shop URL should be filtered, article URL collected
        assert summary["total_collected"] == 1

    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.search_brave")
    def test_deduplicates_content(self, mock_search, mock_fetch, tmp_path):
        """Identical content from different URLs is deduplicated."""
        mock_search.return_value = [
            {"url": "https://a.com/sedan-article-here", "title": "A", "description": ""},
            {"url": "https://b.com/sedan-article-here", "title": "B", "description": ""},
        ]
        words = " ".join(f"word{i}" for i in range(200))
        mock_fetch.return_value = f"<html><body><p>{words}</p></body></html>"

        config = _make_config()
        categories = _make_categories()

        with patch("classivore.collection.extract_text") as mock_extract:
            mock_extract.return_value = words
            summary = run_collection(
                config=config,
                categories=categories,
                data_dir=tmp_path,
                pages=2,
            )

        # Same content = only 1 collected
        assert summary["total_collected"] == 1

    def test_resume_preserves_state(self, tmp_path):
        """Resume mode preserves collected count from previous runs."""
        from classivore.collection.state import CollectionState

        # Simulate a previous run by writing state
        state = CollectionState(tmp_path / "collection")
        state.init_category("Sedan", target=5)
        state.record_url("https://example.com/old", "Sedan", "collected", "live_scrape")
        state.record_query("Sedan", "old query")
        state.save()

        config = _make_config()
        config.target_per_category = 5
        categories = _make_categories()

        with patch("classivore.collection.search_brave") as mock_search, \
             patch("classivore.collection.fetch_page") as mock_fetch, \
             patch("classivore.collection.extract_text") as mock_extract:
            words = " ".join(f"word{i}" for i in range(200))
            mock_search.return_value = [
                {"url": "https://example.com/new-sedan-article", "title": "New", "description": ""},
            ]
            mock_fetch.return_value = f"<html><body><p>{words}</p></body></html>"
            mock_extract.return_value = words

            summary = run_collection(config=config, categories=categories, data_dir=tmp_path, pages=5)

        # 1 from previous state + 1 new = 2
        assert summary["total_collected"] == 2

    def test_excludes_configured_categories(self, tmp_path):
        """Categories in excluded_categories are skipped."""
        config = _make_config()
        config.excluded_categories = ["Automotive: Sedan"]

        categories = _make_categories()

        with patch("classivore.collection.search_brave") as mock_search:
            mock_search.return_value = []
            summary = run_collection(
                config=config,
                categories=categories,
                data_dir=tmp_path,
                queries_only=True,
            )

        # Sedan is excluded, only non-leaf Automotive remains (which is also excluded as non-leaf)
        assert summary["total_categories"] == 0


class TestAuditDomains:
    def test_audit_returns_report(self, tmp_path):
        from classivore.collection import audit_domains
        report = audit_domains(tmp_path)
        assert isinstance(report, str)
