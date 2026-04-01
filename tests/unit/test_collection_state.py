#!/usr/bin/env python3
"""Tests for collection state manager."""

import json

import pytest

from classivore.collection.state import CollectionState


@pytest.fixture
def state_dir(tmp_path):
    """Return a tmp directory for state files."""
    return tmp_path / "collection"


@pytest.fixture
def state(state_dir):
    """Create a fresh CollectionState."""
    return CollectionState(state_dir)


class TestInit:
    def test_creates_state_dir(self, state_dir):
        CollectionState(state_dir)
        assert state_dir.exists()

    def test_empty_initial_state(self, state):
        assert state.categories == {}
        assert state.urls == {}


class TestSaveAndLoad:
    def test_roundtrip(self, state_dir):
        state = CollectionState(state_dir)
        state.init_category("Automotive", target=10)
        state.record_query("Automotive", "automotive articles")
        state.save()

        loaded = CollectionState(state_dir)
        assert "Automotive" in loaded.categories
        assert loaded.categories["Automotive"]["target"] == 10
        assert "automotive articles" in loaded.categories["Automotive"]["queries_tried"]

    def test_atomic_save_leaves_no_temp(self, state_dir):
        state = CollectionState(state_dir)
        state.save()

        files = list(state_dir.iterdir())
        names = [f.name for f in files]
        assert "state.json" in names
        assert not any(n.startswith(".state") for n in names)

    def test_save_creates_valid_json(self, state_dir):
        state = CollectionState(state_dir)
        state.init_category("Tech", target=5)
        state.save()

        raw = (state_dir / "state.json").read_text()
        data = json.loads(raw)
        assert "categories" in data
        assert "urls" in data


class TestCategoryTracking:
    def test_init_category(self, state):
        state.init_category("Automotive", target=10)
        cat = state.categories["Automotive"]
        assert cat["target"] == 10
        assert cat["collected"] == 0
        assert cat["queries_tried"] == []
        assert cat["source_domains"] == {}

    def test_init_preserves_existing(self, state):
        state.init_category("Automotive", target=10)
        state.categories["Automotive"]["collected"] = 5
        state.init_category("Automotive", target=10)
        assert state.categories["Automotive"]["collected"] == 5

    def test_is_satisfied(self, state):
        state.init_category("Automotive", target=3)
        assert not state.is_satisfied("Automotive")
        state.categories["Automotive"]["collected"] = 3
        assert state.is_satisfied("Automotive")

    def test_is_satisfied_unknown_category(self, state):
        assert not state.is_satisfied("Unknown")


class TestQueryDedup:
    def test_record_query(self, state):
        state.init_category("Tech", target=5)
        state.record_query("Tech", "tech articles 2026")
        assert "tech articles 2026" in state.categories["Tech"]["queries_tried"]

    def test_has_query(self, state):
        state.init_category("Tech", target=5)
        assert not state.has_query("Tech", "tech articles")
        state.record_query("Tech", "tech articles")
        assert state.has_query("Tech", "tech articles")

    def test_has_query_unknown_category(self, state):
        assert not state.has_query("Unknown", "anything")


class TestUrlTracking:
    def test_record_url(self, state):
        state.init_category("Automotive", target=10)
        state.record_url(
            "https://example.com/article",
            category="Automotive",
            status="collected",
            source="commoncrawl",
        )
        assert "https://example.com/article" in state.urls
        assert state.urls["https://example.com/article"]["status"] == "collected"

    def test_record_url_increments_collected(self, state):
        state.init_category("Automotive", target=10)
        state.record_url(
            "https://example.com/a1", category="Automotive",
            status="collected", source="live_scrape",
        )
        state.record_url(
            "https://example.com/a2", category="Automotive",
            status="collected", source="commoncrawl",
        )
        assert state.categories["Automotive"]["collected"] == 2

    def test_record_url_tracks_domains(self, state):
        state.init_category("Automotive", target=10)
        state.record_url(
            "https://example.com/a1", category="Automotive",
            status="collected", source="live_scrape",
        )
        state.record_url(
            "https://example.com/a2", category="Automotive",
            status="collected", source="live_scrape",
        )
        assert state.categories["Automotive"]["source_domains"]["example.com"] == 2

    def test_failed_url_does_not_increment(self, state):
        state.init_category("Automotive", target=10)
        state.record_url(
            "https://example.com/bad", category="Automotive",
            status="failed", source="live_scrape",
        )
        assert state.categories["Automotive"]["collected"] == 0

    def test_filtered_url_does_not_increment(self, state):
        state.init_category("Automotive", target=10)
        state.record_url(
            "https://example.com/junk", category="Automotive",
            status="filtered", source="live_scrape",
        )
        assert state.categories["Automotive"]["collected"] == 0

    def test_is_url_known(self, state):
        assert not state.is_url_known("https://example.com/article")
        state.init_category("Tech", target=5)
        state.record_url(
            "https://example.com/article", category="Tech",
            status="collected", source="commoncrawl",
        )
        assert state.is_url_known("https://example.com/article")

    def test_url_status(self, state):
        state.init_category("Tech", target=5)
        state.record_url(
            "https://example.com/dup", category="Tech",
            status="duplicate", source="commoncrawl",
        )
        assert state.urls["https://example.com/dup"]["status"] == "duplicate"


class TestDomainCount:
    def test_domain_count_per_category(self, state):
        state.init_category("Tech", target=10)
        state.record_url("https://blog.example.com/a", category="Tech", status="collected", source="live_scrape")
        state.record_url("https://blog.example.com/b", category="Tech", status="collected", source="live_scrape")
        state.record_url("https://other.com/c", category="Tech", status="collected", source="live_scrape")
        assert state.get_domain_count("Tech", "blog.example.com") == 2
        assert state.get_domain_count("Tech", "other.com") == 1

    def test_domain_count_unknown(self, state):
        assert state.get_domain_count("Unknown", "example.com") == 0


class TestSummary:
    def test_summary(self, state):
        state.init_category("Automotive", target=10)
        state.init_category("Tech", target=5)
        state.categories["Automotive"]["collected"] = 7
        state.categories["Tech"]["collected"] = 5

        summary = state.summary()
        assert summary["total_categories"] == 2
        assert summary["satisfied_categories"] == 1
        assert summary["total_collected"] == 12
        assert summary["total_target"] == 15
