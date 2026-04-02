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


class TestTimestamps:
    def test_started_at_set_on_first_save(self, state_dir):
        state = CollectionState(state_dir)
        assert state.started_at is None
        state.save()
        assert state.started_at is not None

    def test_started_at_preserved_on_subsequent_saves(self, state_dir):
        state = CollectionState(state_dir)
        state.save()
        first_started = state.started_at
        state.save()
        assert state.started_at == first_started

    def test_last_checkpoint_updates_on_save(self, state_dir):
        state = CollectionState(state_dir)
        state.save()
        first_checkpoint = state.last_checkpoint_at
        state.save()
        assert state.last_checkpoint_at >= first_checkpoint

    def test_timestamps_persist_through_load(self, state_dir):
        state = CollectionState(state_dir)
        state.save()
        started = state.started_at

        loaded = CollectionState(state_dir)
        assert loaded.started_at == started
        assert loaded.last_checkpoint_at is not None

    def test_url_gets_timestamp(self, state):
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/a", "Tech", "collected", "live_scrape")
        assert "timestamp" in state.urls["https://example.com/a"]


class TestErrorCounts:
    def test_initial_error_counts(self, state):
        assert state.error_counts["search_errors"] == 0
        assert state.error_counts["fetch_errors"] == 0
        assert state.error_counts["filtered"] == 0
        assert state.error_counts["duplicates"] == 0

    def test_failed_increments_fetch_errors(self, state):
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/bad", "Tech", "failed", "live_scrape")
        assert state.error_counts["fetch_errors"] == 1

    def test_filtered_increments_filtered(self, state):
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/junk", "Tech", "filtered", "search")
        assert state.error_counts["filtered"] == 1

    def test_duplicate_increments_duplicates(self, state):
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/dup", "Tech", "duplicate", "live_scrape")
        assert state.error_counts["duplicates"] == 1

    def test_collected_does_not_increment_errors(self, state):
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/good", "Tech", "collected", "live_scrape")
        assert state.error_counts == {
            "search_errors": 0, "fetch_errors": 0, "filtered": 0, "duplicates": 0,
        }

    def test_search_error_counter(self, state):
        state.record_search_error()
        state.record_search_error()
        assert state.error_counts["search_errors"] == 2

    def test_error_counts_persist(self, state_dir):
        state = CollectionState(state_dir)
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/bad", "Tech", "failed", "live_scrape")
        state.record_search_error()
        state.save()

        loaded = CollectionState(state_dir)
        assert loaded.error_counts["fetch_errors"] == 1
        assert loaded.error_counts["search_errors"] == 1


class TestBackwardCompat:
    def test_loads_old_state_without_timestamps(self, state_dir):
        """State files from before timestamps were added should load fine."""
        state_dir.mkdir(parents=True, exist_ok=True)
        old_data = {
            "categories": {"Tech": {"target": 5, "collected": 2, "queries_tried": [], "source_domains": {}}},
            "urls": {"https://example.com/a": {"category": "Tech", "status": "collected", "source": "live_scrape"}},
        }
        (state_dir / "state.json").write_text(json.dumps(old_data))

        state = CollectionState(state_dir)
        assert state.started_at is None
        assert state.last_checkpoint_at is None
        assert state.error_counts["search_errors"] == 0
        assert state.categories["Tech"]["collected"] == 2


class TestRecentUrls:
    def test_recent_urls_within_window(self, state):
        from datetime import datetime, timezone
        state.init_category("Tech", target=5)
        # Record URLs — they get current timestamps
        state.record_url("https://example.com/a", "Tech", "collected", "live_scrape")
        state.record_url("https://example.com/b", "Tech", "collected", "live_scrape")
        assert state.recent_urls(minutes=10) == 2

    def test_recent_urls_outside_window(self, state):
        from datetime import datetime, timedelta, timezone
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/old", "Tech", "collected", "live_scrape")
        # Manually backdate the timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        state.urls["https://example.com/old"]["timestamp"] = old_time
        assert state.recent_urls(minutes=10) == 0

    def test_recent_urls_mixed(self, state):
        from datetime import datetime, timedelta, timezone
        state.init_category("Tech", target=5)
        state.record_url("https://example.com/new", "Tech", "collected", "live_scrape")
        state.record_url("https://example.com/old", "Tech", "collected", "live_scrape")
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        state.urls["https://example.com/old"]["timestamp"] = old_time
        assert state.recent_urls(minutes=10) == 1

    def test_recent_urls_no_timestamp(self, state):
        """URLs without timestamps (from old state) are excluded."""
        state.urls["https://example.com/legacy"] = {"status": "collected", "category": "Tech", "source": "live_scrape"}
        assert state.recent_urls(minutes=10) == 0


class TestCoverageHistogram:
    def test_empty_state(self, state):
        assert state.coverage_histogram() == {"0": 0, "1-5": 0, "6-10": 0, "11-50": 0, "50+": 0}

    def test_mixed_buckets(self, state):
        state.categories = {
            "a": {"collected": 0, "target": 50},
            "b": {"collected": 1, "target": 50},
            "c": {"collected": 5, "target": 50},
            "d": {"collected": 6, "target": 50},
            "e": {"collected": 10, "target": 50},
            "f": {"collected": 25, "target": 50},
            "g": {"collected": 51, "target": 50},
        }
        hist = state.coverage_histogram()
        assert hist["0"] == 1
        assert hist["1-5"] == 2
        assert hist["6-10"] == 2
        assert hist["11-50"] == 1
        assert hist["50+"] == 1


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

    def test_summary_includes_timestamps(self, state_dir):
        state = CollectionState(state_dir)
        state.save()
        summary = state.summary()
        assert summary["started_at"] is not None
        assert summary["last_checkpoint_at"] is not None

    def test_summary_includes_error_counts(self, state):
        state.record_search_error()
        summary = state.summary()
        assert summary["error_counts"]["search_errors"] == 1
