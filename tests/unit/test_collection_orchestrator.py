#!/usr/bin/env python3
"""Tests for collection orchestrator."""

import json
import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from classivore.collection import run_collection


def _make_config(slug="iab-2.2"):
    config = MagicMock()
    config.slug = slug
    config.target_per_category = 2
    config.max_queries_per_category = 6
    config.max_per_domain_per_category = 50
    config.commoncrawl_crawl_id = None
    config.query_model = "claude-haiku-4-5-20251001"
    config.excluded_categories = []
    config.search_providers = None
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


def _words(n, prefix="word"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def _mock_search_client(results):
    """Create a mock SearchClient that returns given results."""
    client = MagicMock()
    client.search.return_value = results
    client.active_provider_count = 1
    client.reset_exhausted = MagicMock()
    return client


class TestRunCollection:
    @patch("classivore.collection.extract_text")
    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.SearchClient")
    def test_collects_pages(self, mock_client_cls, mock_fetch, mock_extract, tmp_path):
        """End-to-end: search → fetch → filter → save."""
        words1 = _words(200)
        words2 = _words(200, "other")

        client = _mock_search_client([
            {"url": "https://example.com/sedan-article-one", "title": "A", "description": ""},
            {"url": "https://example.com/sedan-article-two", "title": "B", "description": ""},
        ])
        mock_client_cls.from_config.return_value = client
        mock_fetch.side_effect = [f"<html><body><p>{words1}</p></body></html>", f"<html><body><p>{words2}</p></body></html>"]
        mock_extract.side_effect = [words1, words2]

        summary = run_collection(config=_make_config(), categories=_make_categories(), data_dir=tmp_path)

        assert summary["total_collected"] == 2
        assert summary["satisfied_categories"] == 1

        corpus_file = tmp_path / "corpus" / "pages.json"
        assert corpus_file.exists()
        lines = corpus_file.read_text().strip().split("\n")
        assert len(lines) == 2

    @patch("classivore.collection.SearchClient")
    def test_queries_only_mode(self, mock_client_cls, tmp_path):
        """queries_only generates queries without searching."""
        client = _mock_search_client([])
        mock_client_cls.from_config.return_value = client

        summary = run_collection(
            config=_make_config(), categories=_make_categories(),
            data_dir=tmp_path, queries_only=True,
        )

        assert summary["total_collected"] == 0
        client.search.assert_not_called()

    @patch("classivore.collection.extract_text")
    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.SearchClient")
    def test_skips_blocked_urls(self, mock_client_cls, mock_fetch, mock_extract, tmp_path):
        client = _mock_search_client([
            {"url": "https://example.com/cart/item", "title": "Shop", "description": ""},
            {"url": "https://example.com/great-sedan-article-here", "title": "Article", "description": ""},
        ])
        mock_client_cls.from_config.return_value = client
        words = _words(200)
        mock_fetch.return_value = f"<html><body><p>{words}</p></body></html>"
        mock_extract.return_value = words

        config = _make_config()
        config.target_per_category = 1

        summary = run_collection(config=config, categories=_make_categories(), data_dir=tmp_path)
        assert summary["total_collected"] == 1

    @patch("classivore.collection.extract_text")
    @patch("classivore.collection.fetch_page")
    @patch("classivore.collection.SearchClient")
    def test_deduplicates_content(self, mock_client_cls, mock_fetch, mock_extract, tmp_path):
        client = _mock_search_client([
            {"url": "https://a.com/sedan-article-here", "title": "A", "description": ""},
            {"url": "https://b.com/sedan-article-here", "title": "B", "description": ""},
        ])
        mock_client_cls.from_config.return_value = client
        words = _words(200)
        mock_fetch.return_value = f"<html><body><p>{words}</p></body></html>"
        mock_extract.return_value = words

        summary = run_collection(config=_make_config(), categories=_make_categories(), data_dir=tmp_path)
        assert summary["total_collected"] == 1

    def test_excludes_configured_categories(self, tmp_path):
        config = _make_config()
        config.excluded_categories = ["Automotive: Sedan"]

        with patch("classivore.collection.SearchClient") as mock_client_cls:
            client = _mock_search_client([])
            mock_client_cls.from_config.return_value = client
            summary = run_collection(
                config=config, categories=_make_categories(),
                data_dir=tmp_path, queries_only=True,
            )

        assert summary["total_categories"] == 0

    def test_per_taxonomy_state_dir(self, tmp_path):
        """State is stored in per-taxonomy directory."""
        with patch("classivore.collection.SearchClient") as mock_client_cls:
            client = _mock_search_client([])
            mock_client_cls.from_config.return_value = client
            run_collection(
                config=_make_config(slug="iab-2.2"), categories=_make_categories(),
                data_dir=tmp_path, queries_only=True,
            )

        state_file = tmp_path / "collection" / "iab-2.2" / "state.json"
        assert state_file.exists()


class TestCircuitBreaker:
    @patch("classivore.collection.time.sleep")
    @patch("classivore.collection.SearchClient")
    def test_pauses_after_consecutive_failures(self, mock_client_cls, mock_sleep, tmp_path):
        """Circuit breaker pauses after CIRCUIT_BREAKER_THRESHOLD consecutive None results."""
        client = MagicMock()
        # Return None enough times to trigger circuit breaker, then empty lists for remaining
        # With expanded templates, each category generates 30+ queries
        client.search.side_effect = [None] * 6 + [[]] * 500
        client.active_provider_count = 1
        client.reset_exhausted = MagicMock()
        mock_client_cls.from_config.return_value = client

        # Use multiple leaf categories to generate enough queries
        categories = _make_categories() + [
            {"id": "3", "name": "SUV", "display_name": "Automotive: SUV",
             "description": "Sport utility.", "boundaries": "", "path": ["Automotive", "SUV"],
             "depth": 2, "is_leaf": True, "children_count": 0},
            {"id": "4", "name": "Coupe", "display_name": "Automotive: Coupe",
             "description": "Two-door sports.", "boundaries": "", "path": ["Automotive", "Coupe"],
             "depth": 2, "is_leaf": True, "children_count": 0},
        ]

        run_collection(config=_make_config(), categories=categories, data_dir=tmp_path)

        from classivore.collection import CIRCUIT_BREAKER_PAUSE
        mock_sleep.assert_any_call(CIRCUIT_BREAKER_PAUSE)

    @patch("classivore.collection.time.sleep")
    @patch("classivore.collection.SearchClient")
    def test_resets_on_success(self, mock_client_cls, mock_sleep, tmp_path):
        """Consecutive failure count resets when search succeeds."""
        client = MagicMock()
        # 3 failures, then success, then 3 failures — should NOT trigger circuit breaker
        # Pad with empty results for the many template queries
        client.search.side_effect = [None, None, None, [], None, None, None] + [[]] * 500
        client.active_provider_count = 1
        client.reset_exhausted = MagicMock()
        mock_client_cls.from_config.return_value = client

        run_collection(config=_make_config(), categories=_make_categories(), data_dir=tmp_path)

        from classivore.collection import CIRCUIT_BREAKER_PAUSE
        # Circuit breaker should NOT have fired (never hit 5 consecutive)
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert CIRCUIT_BREAKER_PAUSE not in sleep_args


class TestQuerySlotProtection:
    @patch("classivore.collection.time.sleep")
    @patch("classivore.collection.SearchClient")
    def test_query_not_recorded_on_transient_failure(self, mock_client_cls, mock_sleep, tmp_path):
        """Query is NOT recorded when search returns None (transient failure)."""
        import classivore.collection as coll

        client = MagicMock()
        call_count = 0

        def search_then_interrupt(query, count=10):
            nonlocal call_count
            call_count += 1
            # Let a few queries fail, then interrupt to stop the loop
            if call_count >= 6:
                coll._interrupted = True
            return None

        client.search.side_effect = search_then_interrupt
        client.active_provider_count = 1
        client.reset_exhausted = MagicMock()
        mock_client_cls.from_config.return_value = client

        config = _make_config()
        config.target_per_category = 1
        run_collection(config=config, categories=_make_categories(), data_dir=tmp_path)

        # Load state and check queries_tried is empty
        state_file = tmp_path / "collection" / "iab-2.2" / "state.json"
        state = json.loads(state_file.read_text())
        sedan = state["categories"].get("Sedan", {})
        assert sedan.get("queries_tried", []) == []

    @patch("classivore.collection.SearchClient")
    def test_query_recorded_on_empty_results(self, mock_client_cls, tmp_path):
        """Query IS recorded when search returns [] (successful but empty)."""
        client = MagicMock()
        client.search.return_value = []
        client.active_provider_count = 1
        mock_client_cls.from_config.return_value = client

        config = _make_config()
        config.target_per_category = 1
        run_collection(config=config, categories=_make_categories(), data_dir=tmp_path)

        state_file = tmp_path / "collection" / "iab-2.2" / "state.json"
        state = json.loads(state_file.read_text())
        sedan = state["categories"].get("Sedan", {})
        assert len(sedan.get("queries_tried", [])) > 0


class TestSIGINT:
    @patch("classivore.collection.SearchClient")
    def test_sigint_saves_state(self, mock_client_cls, tmp_path):
        """SIGINT triggers state save and clean exit."""
        import classivore.collection as coll

        client = MagicMock()

        def search_and_interrupt(query, count=10):
            coll._interrupted = True
            return []

        client.search.side_effect = search_and_interrupt
        client.active_provider_count = 1
        mock_client_cls.from_config.return_value = client

        summary = run_collection(config=_make_config(), categories=_make_categories(), data_dir=tmp_path)

        state_file = tmp_path / "collection" / "iab-2.2" / "state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert "categories" in state



class TestAuditDomains:
    def test_audit_returns_report(self, tmp_path):
        from classivore.collection import audit_domains
        report = audit_domains(tmp_path)
        assert isinstance(report, str)
