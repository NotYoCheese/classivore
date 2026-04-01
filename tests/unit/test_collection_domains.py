#!/usr/bin/env python3
"""Tests for domain quality tracker."""

import json

import pytest

from classivore.collection.domains import DomainTracker


@pytest.fixture
def tracker_dir(tmp_path):
    return tmp_path / "collection"


@pytest.fixture
def tracker(tracker_dir):
    return DomainTracker(tracker_dir)


class TestInit:
    def test_creates_dir(self, tracker_dir):
        DomainTracker(tracker_dir)
        assert tracker_dir.exists()

    def test_empty_initial_state(self, tracker):
        assert tracker.scores == {}
        assert tracker.blocklist == set()


class TestRecordResult:
    def test_record_success(self, tracker):
        tracker.record_result("example.com", success=True)
        assert tracker.scores["example.com"]["successes"] == 1
        assert tracker.scores["example.com"]["attempts"] == 1

    def test_record_failure(self, tracker):
        tracker.record_result("example.com", success=False)
        assert tracker.scores["example.com"]["successes"] == 0
        assert tracker.scores["example.com"]["attempts"] == 1

    def test_accumulates(self, tracker):
        tracker.record_result("example.com", success=True)
        tracker.record_result("example.com", success=True)
        tracker.record_result("example.com", success=False)
        assert tracker.scores["example.com"]["successes"] == 2
        assert tracker.scores["example.com"]["attempts"] == 3


class TestSuccessRate:
    def test_success_rate(self, tracker):
        tracker.record_result("example.com", success=True)
        tracker.record_result("example.com", success=False)
        assert tracker.success_rate("example.com") == 0.5

    def test_unknown_domain(self, tracker):
        assert tracker.success_rate("unknown.com") is None

    def test_zero_attempts(self, tracker):
        tracker.scores["example.com"] = {"successes": 0, "attempts": 0}
        assert tracker.success_rate("example.com") is None


class TestAutoBlock:
    def test_not_blocked_under_threshold_attempts(self, tracker):
        # 4 failures isn't enough attempts to trigger auto-block
        for _ in range(4):
            tracker.record_result("bad.com", success=False)
        assert not tracker.is_blocked("bad.com")

    def test_blocked_after_threshold(self, tracker):
        # 5 failures = 0% success rate, triggers auto-block
        for _ in range(5):
            tracker.record_result("bad.com", success=False)
        assert tracker.is_blocked("bad.com")

    def test_not_blocked_if_good_rate(self, tracker):
        # 4 successes + 1 failure = 80%, above 20% threshold
        for _ in range(4):
            tracker.record_result("decent.com", success=True)
        tracker.record_result("decent.com", success=False)
        assert not tracker.is_blocked("decent.com")

    def test_blocked_at_boundary(self, tracker):
        # 1 success + 4 failures = 20%, at boundary (not below)
        tracker.record_result("edge.com", success=True)
        for _ in range(4):
            tracker.record_result("edge.com", success=False)
        assert not tracker.is_blocked("edge.com")

    def test_blocked_just_below_boundary(self, tracker):
        # 1 success + 5 failures = ~17%, below 20%
        tracker.record_result("poor.com", success=True)
        for _ in range(5):
            tracker.record_result("poor.com", success=False)
        assert tracker.is_blocked("poor.com")

    def test_manual_blocklist_always_blocked(self, tracker):
        tracker.blocklist.add("spam.com")
        assert tracker.is_blocked("spam.com")


class TestManualBlocklist:
    def test_add_to_blocklist(self, tracker):
        tracker.add_to_blocklist("spam.com")
        assert "spam.com" in tracker.blocklist
        assert tracker.is_blocked("spam.com")

    def test_remove_from_blocklist(self, tracker):
        tracker.add_to_blocklist("spam.com")
        tracker.remove_from_blocklist("spam.com")
        assert "spam.com" not in tracker.blocklist

    def test_remove_nonexistent_no_error(self, tracker):
        tracker.remove_from_blocklist("nonexistent.com")


class TestPersistence:
    def test_scores_roundtrip(self, tracker_dir):
        tracker = DomainTracker(tracker_dir)
        tracker.record_result("example.com", success=True)
        tracker.record_result("example.com", success=False)
        tracker.save()

        loaded = DomainTracker(tracker_dir)
        assert loaded.scores["example.com"]["successes"] == 1
        assert loaded.scores["example.com"]["attempts"] == 2

    def test_blocklist_roundtrip(self, tracker_dir):
        tracker = DomainTracker(tracker_dir)
        tracker.add_to_blocklist("spam.com")
        tracker.add_to_blocklist("junk.org")
        tracker.save()

        loaded = DomainTracker(tracker_dir)
        assert loaded.blocklist == {"spam.com", "junk.org"}

    def test_save_creates_valid_json(self, tracker_dir):
        tracker = DomainTracker(tracker_dir)
        tracker.record_result("example.com", success=True)
        tracker.add_to_blocklist("spam.com")
        tracker.save()

        scores_data = json.loads((tracker_dir / "domain_scores.json").read_text())
        assert "example.com" in scores_data

        blocklist_data = json.loads((tracker_dir / "domain_blocklist.json").read_text())
        assert "spam.com" in blocklist_data


class TestAuditReport:
    def test_audit_report_format(self, tracker):
        tracker.record_result("good.com", success=True)
        tracker.record_result("good.com", success=True)
        for _ in range(5):
            tracker.record_result("bad.com", success=False)
        tracker.add_to_blocklist("manual.com")

        report = tracker.audit_report()
        assert "good.com" in report
        assert "bad.com" in report
        assert "manual.com" in report

    def test_audit_report_empty(self, tracker):
        report = tracker.audit_report()
        assert "No domain" in report or "empty" in report.lower() or report != ""
