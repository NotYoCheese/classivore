#!/usr/bin/env python3
"""Tests for labeling state persistence."""

import json

import pytest

from classivore.labeling.state import LabelState


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "labels" / "iab-2.2"


@pytest.fixture
def state(state_dir):
    return LabelState(state_dir)


class TestInit:
    def test_creates_state_dir(self, state_dir):
        LabelState(state_dir)
        assert state_dir.exists()

    def test_empty_initial_state(self, state):
        assert state.pages == {}
        assert state.started_at is None
        assert state.stage1_batch_ids == []
        assert state.stage2_batch_ids == []


class TestPageLifecycle:
    def test_init_page(self, state):
        state.init_page("abc123", "https://example.com/article")
        assert "abc123" in state.pages
        assert state.pages["abc123"]["status"] == "unlabeled"
        assert state.pages["abc123"]["url"] == "https://example.com/article"

    def test_init_page_idempotent(self, state):
        state.init_page("abc123", "https://example.com/article")
        state.complete_stage1("abc123", [{"name": "Automotive", "confidence": 0.9}])
        state.init_page("abc123", "https://example.com/article")
        assert state.pages["abc123"]["status"] == "stage1_complete"

    def test_complete_stage1(self, state):
        state.init_page("abc123", "https://example.com/article")
        tier1 = [{"name": "Automotive", "confidence": 0.9}]
        state.complete_stage1("abc123", tier1)
        assert state.pages["abc123"]["status"] == "stage1_complete"
        assert state.pages["abc123"]["tier1_categories"] == tier1

    def test_complete_stage2(self, state):
        state.init_page("abc123", "https://example.com/article")
        state.complete_stage1("abc123", [{"name": "Automotive", "confidence": 0.9}])
        labels = [{"name": "Sedan", "confidence": 0.95}]
        state.complete_stage2("abc123", labels, "This is about sedans.")
        assert state.pages["abc123"]["status"] == "stage2_complete"
        assert state.pages["abc123"]["labels"] == labels
        assert state.pages["abc123"]["reasoning"] == "This is about sedans."

    def test_mark_error(self, state):
        state.init_page("abc123", "https://example.com/article")
        state.mark_error("abc123", "JSON parse failed")
        assert state.pages["abc123"]["status"] == "error"
        assert state.pages["abc123"]["error"] == "JSON parse failed"


class TestPageQueries:
    def test_pages_needing_stage1(self, state):
        state.init_page("a", "https://example.com/a")
        state.init_page("b", "https://example.com/b")
        state.init_page("c", "https://example.com/c")
        state.complete_stage1("b", [{"name": "Tech", "confidence": 0.8}])
        state.complete_stage2("c", [{"name": "Science", "confidence": 0.9}], "Reason")

        needing = state.pages_needing_stage1()
        assert needing == ["a"]

    def test_pages_needing_stage2(self, state):
        state.init_page("a", "https://example.com/a")
        state.init_page("b", "https://example.com/b")
        state.init_page("c", "https://example.com/c")
        state.complete_stage1("a", [{"name": "Tech", "confidence": 0.8}])
        state.complete_stage1("b", [{"name": "Sports", "confidence": 0.7}])
        state.complete_stage2("c", [{"name": "Science", "confidence": 0.9}], "Reason")

        needing = state.pages_needing_stage2()
        assert sorted(needing) == ["a", "b"]

    def test_get_tier1_for_page(self, state):
        state.init_page("abc", "https://example.com/abc")
        state.complete_stage1("abc", [
            {"name": "Automotive", "confidence": 0.9},
            {"name": "Science", "confidence": 0.5},
        ])
        tier1 = state.get_tier1_for_page("abc")
        assert tier1 == ["Automotive", "Science"]

    def test_get_tier1_for_unlabeled_page(self, state):
        state.init_page("abc", "https://example.com/abc")
        assert state.get_tier1_for_page("abc") == []

    def test_is_complete(self, state):
        state.init_page("a", "https://example.com/a")
        assert not state.is_complete("a")
        state.complete_stage1("a", [])
        assert not state.is_complete("a")
        state.complete_stage2("a", [], "")
        assert state.is_complete("a")


class TestPersistence:
    def test_roundtrip(self, state_dir):
        state = LabelState(state_dir)
        state.init_page("abc", "https://example.com/abc")
        state.complete_stage1("abc", [{"name": "Tech", "confidence": 0.8}])
        state.stage1_batch_ids.append("batch_123")
        state.save()

        loaded = LabelState(state_dir)
        assert "abc" in loaded.pages
        assert loaded.pages["abc"]["status"] == "stage1_complete"
        assert loaded.stage1_batch_ids == ["batch_123"]
        assert loaded.started_at is not None

    def test_atomic_save_no_temp_files(self, state_dir):
        state = LabelState(state_dir)
        state.save()
        names = [f.name for f in state_dir.iterdir()]
        assert "label_state.json" in names
        assert not any(n.startswith(".label_state") for n in names)

    def test_backward_compat_empty_file(self, state_dir):
        """Loading from a dir with no state file works."""
        state = LabelState(state_dir)
        assert state.pages == {}


class TestStats:
    def test_summary(self, state):
        state.init_page("a", "https://example.com/a")
        state.init_page("b", "https://example.com/b")
        state.init_page("c", "https://example.com/c")
        state.init_page("d", "https://example.com/d")
        state.complete_stage1("b", [])
        state.complete_stage2("c", [], "")
        state.mark_error("d", "Failed")

        stats = state.summary()
        assert stats["total_pages"] == 4
        assert stats["unlabeled"] == 1
        assert stats["stage1_complete"] == 1
        assert stats["stage2_complete"] == 1
        assert stats["error"] == 1

    def test_summary_str(self, state):
        state.init_page("a", "https://example.com/a")
        output = state.summary_str()
        assert "total_pages" in output or "Total" in output
