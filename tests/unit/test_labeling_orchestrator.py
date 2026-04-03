#!/usr/bin/env python3
"""Tests for labeling orchestrator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from classivore.labeling import run_labeling


def _make_config():
    config = MagicMock()
    config.slug = "iab-2.2"
    config.labeling_model = "claude-haiku-4-5-20251001"
    config.stage1_max_tokens = 150
    config.stage2_max_tokens = 300
    config.tier1_confidence_threshold = 0.3
    config.labeling_temperature = 0.0
    config.text_truncation_words = 3000
    config.min_confidence = 0.5
    config.max_labels = 3
    config.excluded_tier1_categories = ["Content Language"]
    return config


def _make_categories():
    return [
        {"id": "1", "name": "Automotive", "depth": 1, "path": ["Automotive"],
         "description": "D", "boundaries": "B", "is_leaf": False, "children_count": 1},
        {"id": "2", "name": "Sedan", "depth": 2, "path": ["Automotive", "Sedan"],
         "description": "D", "boundaries": "B", "is_leaf": True, "children_count": 0},
        {"id": "3", "name": "Content Language", "depth": 1, "path": ["Content Language"],
         "description": "D", "boundaries": "B", "is_leaf": False, "children_count": 1},
        {"id": "4", "name": "English", "depth": 2, "path": ["Content Language", "English"],
         "description": "D", "boundaries": "B", "is_leaf": True, "children_count": 0},
    ]


def _make_hierarchy():
    return {"1": [{"id": "2", "name": "Sedan"}], "3": [{"id": "4", "name": "English"}]}


def _write_corpus(tmp_path, pages):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    with open(corpus_dir / "pages.json", "w") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")


def _make_pages(n=3):
    return [
        {"url": f"https://example.com/article-{i}", "text": f"Article about topic {i}. " * 50,
         "word_count": 300, "content_hash": f"hash{i}", "source": "live_scrape"}
        for i in range(n)
    ]


class TestDryRun:
    def test_dry_run_no_api_calls(self, tmp_path):
        _write_corpus(tmp_path, _make_pages(5))

        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            dry_run=True,
        )

        assert summary["total_pages"] == 5
        assert summary["unlabeled"] == 5

    def test_dry_run_excludes_metadata(self, tmp_path, capsys):
        _write_corpus(tmp_path, _make_pages(1))

        run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            dry_run=True,
        )

        captured = capsys.readouterr()
        # Should show 2 content categories (Automotive, Sedan), not 4
        assert "2 (across 1 tier-1)" in captured.out or "Content categories: 2" in captured.out


class TestStage1:
    @patch("classivore.labeling.iter_succeeded_results")
    @patch("classivore.labeling.poll_until_complete")
    @patch("classivore.labeling.submit_batch")
    @patch("classivore.labeling.get_api_client")
    def test_runs_stage1(self, mock_client, mock_submit, mock_poll, mock_iter, tmp_path):
        _write_corpus(tmp_path, _make_pages(2))
        mock_client.return_value = MagicMock()
        mock_submit.return_value = "batch_s1"
        mock_poll.return_value = MagicMock()

        # Stage 1 responses
        def make_s1_result(hash_id):
            msg = MagicMock()
            block = MagicMock()
            block.text = json.dumps({"categories": [{"name": "Automotive", "confidence": 0.9}]})
            msg.content = [block]
            return (f"s1-{hash_id}", msg)

        mock_iter.return_value = [make_s1_result("hash0"), make_s1_result("hash1")]

        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            stage="1",
        )

        assert summary["stage1_complete"] == 2
        mock_submit.assert_called_once()


class TestStage2:
    @patch("classivore.labeling.iter_succeeded_results")
    @patch("classivore.labeling.poll_until_complete")
    @patch("classivore.labeling.submit_batch")
    @patch("classivore.labeling.get_api_client")
    def test_runs_stage2_after_stage1(self, mock_client, mock_submit, mock_poll, mock_iter, tmp_path):
        _write_corpus(tmp_path, _make_pages(1))
        mock_client.return_value = MagicMock()

        # Stage 1 batch
        s1_msg = MagicMock()
        s1_block = MagicMock()
        s1_block.text = json.dumps({"categories": [{"name": "Automotive", "confidence": 0.9}]})
        s1_msg.content = [s1_block]

        # Stage 2 batch
        s2_msg = MagicMock()
        s2_block = MagicMock()
        s2_block.text = json.dumps({"reasoning": "About sedans.", "categories": [{"name": "Sedan", "confidence": 0.95}]})
        s2_msg.content = [s2_block]

        mock_submit.side_effect = ["batch_s1", "batch_s2"]
        mock_poll.return_value = MagicMock()
        mock_iter.side_effect = [
            [("s1-hash0", s1_msg)],
            [("s2-hash0", s2_msg)],
        ]

        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            stage="all",
        )

        assert summary["stage2_complete"] == 1
        assert mock_submit.call_count == 2


class TestStage1Validation:
    @patch("classivore.labeling.iter_succeeded_results")
    @patch("classivore.labeling.poll_until_complete")
    @patch("classivore.labeling.submit_batch")
    @patch("classivore.labeling.get_api_client")
    def test_drops_hallucinated_tier1_names(self, mock_client, mock_submit, mock_poll, mock_iter, tmp_path):
        """Invalid tier-1 names are filtered out, valid ones kept."""
        _write_corpus(tmp_path, _make_pages(1))
        mock_client.return_value = MagicMock()
        mock_submit.return_value = "batch_s1"

        # Response has one valid and one hallucinated tier-1
        msg = MagicMock()
        block = MagicMock()
        block.text = json.dumps({"categories": [
            {"name": "Automotive", "confidence": 0.9},
            {"name": "Entertainment", "confidence": 0.7},
        ]})
        msg.content = [block]
        mock_iter.return_value = [("s1-hash0", msg)]

        run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            stage="1",
        )

        state_file = tmp_path / "labels" / "iab-2.2" / "label_state.json"
        state = json.loads(state_file.read_text())
        tier1 = state["pages"]["hash0"]["tier1_categories"]
        names = [c["name"] for c in tier1]
        assert "Automotive" in names
        assert "Entertainment" not in names


class TestResume:
    @patch("classivore.labeling.iter_succeeded_results")
    @patch("classivore.labeling.poll_until_complete")
    @patch("classivore.labeling.submit_batch")
    @patch("classivore.labeling.get_api_client")
    def test_skips_stage1_for_triaged_pages(self, mock_client, mock_submit, mock_poll, mock_iter, tmp_path):
        """Pages already at stage1_complete skip directly to stage 2."""
        from classivore.labeling.state import LabelState

        _write_corpus(tmp_path, _make_pages(1))

        # Pre-populate state: page already triaged
        state = LabelState(tmp_path / "labels" / "iab-2.2")
        state.init_page("hash0", "https://example.com/article-0")
        state.complete_stage1("hash0", [{"name": "Automotive", "confidence": 0.9}])
        state.save()

        mock_client.return_value = MagicMock()

        # Only stage 2 should run
        s2_msg = MagicMock()
        s2_block = MagicMock()
        s2_block.text = json.dumps({"reasoning": "R", "categories": [{"name": "Sedan", "confidence": 0.9}]})
        s2_msg.content = [s2_block]

        mock_submit.return_value = "batch_s2"
        mock_poll.return_value = MagicMock()
        mock_iter.return_value = [("s2-hash0", s2_msg)]

        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
            stage="all",
        )

        assert summary["stage2_complete"] == 1
        # Only one batch submitted (stage 2 only)
        mock_submit.assert_called_once()

    @patch("classivore.labeling.get_api_client")
    def test_skips_complete_pages(self, mock_client, tmp_path):
        """Pages already at stage2_complete are not reprocessed."""
        from classivore.labeling.state import LabelState

        _write_corpus(tmp_path, _make_pages(1))

        state = LabelState(tmp_path / "labels" / "iab-2.2")
        state.init_page("hash0", "https://example.com/article-0")
        state.complete_stage1("hash0", [{"name": "Automotive", "confidence": 0.9}])
        state.complete_stage2("hash0", [{"name": "Sedan", "confidence": 0.95}], "Done.")
        state.save()

        mock_client.return_value = MagicMock()

        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
        )

        assert summary["stage2_complete"] == 1


class TestOutput:
    @patch("classivore.labeling.iter_succeeded_results")
    @patch("classivore.labeling.poll_until_complete")
    @patch("classivore.labeling.submit_batch")
    @patch("classivore.labeling.get_api_client")
    def test_writes_ndjson_labels(self, mock_client, mock_submit, mock_poll, mock_iter, tmp_path):
        _write_corpus(tmp_path, _make_pages(1))
        mock_client.return_value = MagicMock()

        s1_msg = MagicMock()
        s1_block = MagicMock()
        s1_block.text = json.dumps({"categories": [{"name": "Automotive", "confidence": 0.9}]})
        s1_msg.content = [s1_block]

        s2_msg = MagicMock()
        s2_block = MagicMock()
        s2_block.text = json.dumps({"reasoning": "R", "categories": [{"name": "Sedan", "confidence": 0.95}]})
        s2_msg.content = [s2_block]

        mock_submit.side_effect = ["batch_s1", "batch_s2"]
        mock_iter.side_effect = [[("s1-hash0", s1_msg)], [("s2-hash0", s2_msg)]]

        run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
        )

        # Verify NDJSON output for collection seeding
        labels_file = tmp_path / "labels" / "iab-2.2" / "labels.json"
        assert labels_file.exists()
        lines = labels_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["categories"] == ["Sedan"]
        assert entry["content_hash"] == "hash0"

    def test_no_corpus_returns_empty_summary(self, tmp_path):
        summary = run_labeling(
            config=_make_config(),
            categories=_make_categories(),
            hierarchy=_make_hierarchy(),
            data_dir=tmp_path,
        )
        assert summary["total_pages"] == 0
