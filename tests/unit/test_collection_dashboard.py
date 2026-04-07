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


def _write_labels(tmp_path, entries):
    """Write label entries as NDJSON."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_dir / "labels.json", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return labels_dir


class TestEmptyDashboard:
    def test_renders_without_crash(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Collection Status" in output

    def test_shows_not_started(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Not yet started" in output

    def test_shows_no_labels_yet(self, state, domains, tmp_path):
        labels_dir = tmp_path / "labels"
        output = format_status_dashboard(
            state, domains, labels_dir=labels_dir, target_per_category=50,
        )
        assert "No labels yet" in output


class TestWithData:
    def test_shows_taxonomy_slug(self, state, domains):
        output = format_status_dashboard(state, domains, taxonomy_slug="iab-2.2")
        assert "iab-2.2" in output

    def test_shows_progress_from_labels(self, state, domains, tmp_path):
        labels_dir = _write_labels(tmp_path, [
            {"url": f"http://a{i}.com", "content_hash": f"h{i}", "categories": ["Sedan"]}
            for i in range(50)
        ] + [
            {"url": f"http://b{i}.com", "content_hash": f"g{i}", "categories": ["SUV"]}
            for i in range(10)
        ])

        output = format_status_dashboard(
            state, domains, labels_dir=labels_dir, target_per_category=50,
        )
        assert "1/2 categories at target" in output
        assert "60 labeled pages" in output

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
    def test_histogram_buckets(self, state, domains, tmp_path):
        entries = []
        # Category with 3 labels
        for i in range(3):
            entries.append({"url": f"http://a{i}.com", "content_hash": f"a{i}", "categories": ["Low"]})
        # Category with 25 labels
        for i in range(25):
            entries.append({"url": f"http://b{i}.com", "content_hash": f"b{i}", "categories": ["Medium"]})
        # Category with 60 labels
        for i in range(60):
            entries.append({"url": f"http://c{i}.com", "content_hash": f"c{i}", "categories": ["High"]})
        labels_dir = _write_labels(tmp_path, entries)

        output = format_status_dashboard(
            state, domains, labels_dir=labels_dir, target_per_category=50,
        )
        assert "Label Coverage" in output
        assert "1-10" in output
        assert "50+" in output

    def test_no_histogram_without_labels_dir(self, state, domains):
        output = format_status_dashboard(state, domains)
        assert "Label Coverage" not in output


class TestVelocity:
    def test_velocity_with_recent_urls(self, state, domains):
        state.init_category("Sedan", target=50)
        for i in range(5):
            state.record_url(f"https://example.com/{i}", "Sedan", "collected", "live_scrape")

        output = format_status_dashboard(state, domains)
        assert "Pages collected: 5" in output
        assert "pages/min" in output

    def test_velocity_no_recent(self, state, domains):
        state.init_category("Sedan", target=50)
        output = format_status_dashboard(state, domains)
        assert "no recent activity" in output


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
