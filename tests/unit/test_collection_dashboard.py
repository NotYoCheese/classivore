#!/usr/bin/env python3
"""Tests for collection status dashboard."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from classivore.collection.dashboard import format_status_dashboard
from classivore.collection.domains import DomainTracker
from classivore.collection.state import CollectionState


@pytest.fixture
def state(tmp_path):
    return CollectionState(tmp_path / "state")


@pytest.fixture
def domains(tmp_path):
    return DomainTracker(tmp_path / "domains")


class TestEmptyDashboard:
    def test_renders_without_crash(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Collection Status" in output

    def test_shows_not_started(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Not yet started" in output

    def test_shows_no_categories(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "No categories" in output


class TestWithData:
    def test_shows_taxonomy_slug(self, state, domains):
        output = format_status_dashboard(state, domains, taxonomy_slug="iab-2.2")
        assert "iab-2.2" in output

    def test_shows_progress(self, state, domains):
        state.init_category("Sedan", target=50)
        state.init_category("SUV", target=50)
        state.categories["Sedan"]["collected"] = 50
        state.categories["SUV"]["collected"] = 10

        output = format_status_dashboard(state, domains)
        assert "1/2 categories satisfied" in output
        assert "60/100 pages collected" in output

    def test_shows_started_time(self, state, domains):
        state.started_at = datetime.now(timezone.utc).isoformat()
        output = format_status_dashboard(state, domains)
        assert "0h 0m ago" in output

    def test_shows_last_update(self, state, domains):
        state.last_checkpoint_at = "2026-04-02T12:00:00+00:00"
        output = format_status_dashboard(state, domains)
        assert "Last update:" in output
        assert "2026-04-02" in output


class TestCoverageHistogram:
    def test_histogram_buckets(self, state, domains):
        state.categories = {
            "a": {"collected": 0, "target": 50},
            "b": {"collected": 3, "target": 50},
            "c": {"collected": 8, "target": 50},
            "d": {"collected": 25, "target": 50},
            "e": {"collected": 60, "target": 50},
        }
        output = format_status_dashboard(state, domains)
        assert "Coverage" in output
        assert "0 pages:" in output
        assert "1-5" in output
        assert "50+" in output

    def test_no_histogram_when_empty(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Coverage" not in output


class TestVelocity:
    def test_velocity_with_recent_urls(self, state, domains):
        state.init_category("Sedan", target=50)
        # Add URLs with recent timestamps
        for i in range(5):
            state.record_url(f"https://example.com/{i}", "Sedan", "collected", "live_scrape")

        output = format_status_dashboard(state, domains)
        assert "Pages collected: 5" in output
        assert "pages/min" in output

    def test_velocity_no_recent(self, state, domains):
        state.init_category("Sedan", target=50)
        output = format_status_dashboard(state, domains)
        assert "no recent activity" in output
        assert "unknown" in output

    def test_eta_shown(self, state, domains):
        state.init_category("Sedan", target=100)
        for i in range(10):
            state.record_url(f"https://example.com/{i}", "Sedan", "collected", "live_scrape")

        output = format_status_dashboard(state, domains)
        assert "Est. remaining:" in output
        assert "unknown" not in output

    def test_eta_complete(self, state, domains):
        state.init_category("Sedan", target=5)
        for i in range(5):
            state.record_url(f"https://example.com/{i}", "Sedan", "collected", "live_scrape")

        output = format_status_dashboard(state, domains)
        assert "complete" in output


class TestErrors:
    def test_errors_shown(self, state, domains):
        state.error_counts["search_errors"] = 12
        state.error_counts["fetch_errors"] = 45
        state.error_counts["filtered"] = 89
        state.error_counts["duplicates"] = 31

        output = format_status_dashboard(state, domains)
        assert "Errors" in output
        assert "Search errors:  12" in output
        assert "Fetch failures: 45" in output
        assert "Filtered:       89" in output
        assert "Duplicates:     31" in output

    def test_no_errors_section_when_clean(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Errors" not in output


class TestDomainSummary:
    def test_shows_top_domains(self, state, domains):
        domains.record_result("example.com", success=True)
        domains.record_result("example.com", success=True)
        domains.record_result("other.com", success=False)

        output = format_status_dashboard(state, domains)
        assert "Top Domains" in output
        assert "example.com" in output

    def test_shows_blocked_count(self, state, domains):
        domains.add_to_blocklist("spam.com")
        # Auto-block a domain
        for _ in range(6):
            domains.record_result("bad.com", success=False)

        output = format_status_dashboard(state, domains)
        assert "Blocked:" in output
        assert "1 manual" in output

    def test_no_domain_section_when_empty(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Top Domains" not in output


class TestCorpusCount:
    def test_shows_corpus_total(self, state, domains, tmp_path):
        corpus_file = tmp_path / "pages.json"
        with open(corpus_file, "w") as f:
            for i in range(100):
                f.write(json.dumps({"url": f"https://example.com/{i}"}) + "\n")

        output = format_status_dashboard(state, domains, corpus_file=corpus_file)
        assert "100 total corpus pages" in output
