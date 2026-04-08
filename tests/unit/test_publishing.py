#!/usr/bin/env python3
"""Tests for the publishing module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from classivore.publishing.artifact import (
    OPTIONAL,
    REQUIRED,
    load_training_report,
    validate_artifacts,
)
from classivore.publishing.card import generate_model_card
from classivore.publishing.hub import publish_model


def _write_artifact(path, name, content=None):
    """Write a dummy artifact file."""
    f = path / name
    if content is not None:
        f.write_text(json.dumps(content) if isinstance(content, dict) else content)
    else:
        f.write_text("dummy")
    return f


def _make_model_dir(tmp_path):
    """Create a model dir with all required and optional files."""
    for name in REQUIRED:
        if name == "training_report.json":
            _write_artifact(tmp_path, name, {
                "taxonomy": "iab-2.2",
                "model_base": "microsoft/deberta-v3-large",
                "num_categories": 698,
                "metrics": {
                    "eval_f1_micro": 0.6832,
                    "eval_f1_macro": 0.4521,
                    "eval_precision_micro": 0.7100,
                    "eval_recall_micro": 0.6580,
                },
                "hyperparameters": {
                    "learning_rate": 2e-5,
                    "batch_size": 8,
                    "epochs": 3,
                },
                "timestamp": "2026-04-06T19:35:56+00:00",
                "device": "cuda",
                "training_time_seconds": 2580,
            })
        else:
            _write_artifact(tmp_path, name)
    for name in OPTIONAL:
        if name == "quality_report.json":
            _write_artifact(tmp_path, name, {
                "global_threshold": 0.45,
                "global_metrics": {"f1_micro": 0.6512},
            })
        else:
            _write_artifact(tmp_path, name)
    # Also add files that should be excluded
    _write_artifact(tmp_path, "val_probs.npy")
    _write_artifact(tmp_path, "test_probs.npy")
    _write_artifact(tmp_path, "class_weights.json")
    _write_artifact(tmp_path, "confusion_pairs.json")
    return tmp_path


class TestValidateArtifacts:
    def test_valid_dir_passes(self, tmp_path):
        _make_model_dir(tmp_path)
        files = validate_artifacts(tmp_path)
        # All required and optional files included
        for name in REQUIRED | OPTIONAL:
            assert name in files

    def test_excludes_npy_files(self, tmp_path):
        _make_model_dir(tmp_path)
        files = validate_artifacts(tmp_path)
        assert "val_probs.npy" not in files
        assert "test_probs.npy" not in files

    def test_excludes_analysis_files(self, tmp_path):
        _make_model_dir(tmp_path)
        files = validate_artifacts(tmp_path)
        assert "class_weights.json" not in files
        assert "confusion_pairs.json" not in files

    def test_missing_required_raises(self, tmp_path):
        _make_model_dir(tmp_path)
        (tmp_path / "model.safetensors").unlink()
        with pytest.raises(ValueError, match="model.safetensors"):
            validate_artifacts(tmp_path)

    def test_missing_optional_warns(self, tmp_path, capsys):
        from classivore.logging_config import configure_logging
        configure_logging()

        _make_model_dir(tmp_path)
        (tmp_path / "quality_report.json").unlink()
        files = validate_artifacts(tmp_path)
        assert "quality_report.json" not in files
        output = capsys.readouterr()
        assert "quality_report.json" in output.out or "quality_report.json" in output.err


class TestLoadTrainingReport:
    def test_loads_report(self, tmp_path):
        _make_model_dir(tmp_path)
        report = load_training_report(tmp_path)
        assert report["taxonomy"] == "iab-2.2"
        assert report["model_base"] == "microsoft/deberta-v3-large"


class TestGenerateModelCard:
    def test_contains_key_fields(self):
        training_report = {
            "taxonomy": "iab-2.2",
            "model_base": "microsoft/deberta-v3-large",
            "num_categories": 698,
            "metrics": {
                "eval_f1_micro": 0.6832,
                "eval_f1_macro": 0.4521,
                "eval_precision_micro": 0.7100,
                "eval_recall_micro": 0.6580,
            },
            "hyperparameters": {
                "learning_rate": 2e-5,
                "batch_size": 8,
                "epochs": 3,
            },
            "timestamp": "2026-04-06T19:35:56+00:00",
            "device": "cuda",
            "training_time_seconds": 2580,
        }
        quality_report = {
            "global_threshold": 0.45,
            "global_metrics": {"f1_micro": 0.6512},
        }
        card = generate_model_card(
            training_report, quality_report,
            repo_id="classivore/iab22-deberta-large", version="v1.0.0",
        )
        assert "v1.0.0" in card
        assert "classivore/iab22-deberta-large" in card
        assert "0.6832" in card
        assert "microsoft/deberta-v3-large" in card
        assert "MIT" in card
        assert "698" in card

    def test_without_quality_report(self):
        training_report = {
            "taxonomy": "iab-2.2",
            "model_base": "microsoft/deberta-v3-large",
            "num_categories": 698,
            "metrics": {"eval_f1_micro": 0.6832},
            "hyperparameters": {},
            "timestamp": "2026-04-06T19:35:56+00:00",
            "device": "cuda",
            "training_time_seconds": 2580,
        }
        card = generate_model_card(
            training_report, None,
            repo_id="classivore/iab22-deberta-large", version="v1.0.0",
        )
        assert "v1.0.0" in card
        assert "global_threshold" not in card


class TestPublishModel:
    def test_bad_version_raises(self, tmp_path):
        _make_model_dir(tmp_path)
        with pytest.raises(ValueError, match="version"):
            publish_model(tmp_path, "org/repo", "1.0.0", token="tok")

    def test_bad_version_no_network(self, tmp_path):
        """Bad version raises before any HF calls."""
        _make_model_dir(tmp_path)
        with patch("classivore.publishing.hub.upload_folder") as mock_upload:
            with pytest.raises(ValueError):
                publish_model(tmp_path, "org/repo", "bad", token="tok")
            mock_upload.assert_not_called()

    @patch("classivore.publishing.hub.create_tag")
    @patch("classivore.publishing.hub.upload_folder")
    @patch("classivore.publishing.hub.init_repo")
    def test_dry_run_no_network(self, mock_init, mock_upload, mock_tag, tmp_path, capsys):
        _make_model_dir(tmp_path)
        result = publish_model(
            tmp_path, "org/repo", "v1.0.0", token="tok", dry_run=True,
        )
        assert result is None
        mock_init.assert_not_called()
        mock_upload.assert_not_called()
        mock_tag.assert_not_called()
        output = capsys.readouterr().out
        assert "model.safetensors" in output

    @patch("classivore.publishing.hub.create_tag")
    @patch("classivore.publishing.hub.upload_folder")
    @patch("classivore.publishing.hub.init_repo")
    def test_publish_calls_hub(self, mock_init, mock_upload, mock_tag, tmp_path):
        _make_model_dir(tmp_path)
        mock_upload.return_value = MagicMock(commit_url="https://hf.co/commit/abc")
        result = publish_model(
            tmp_path, "org/repo", "v1.0.0", token="tok",
        )
        assert result == "https://hf.co/commit/abc"
        mock_init.assert_called_once_with("org/repo", "tok", private=True)
        mock_upload.assert_called_once()
        mock_tag.assert_called_once_with("org/repo", tag="v1.0.0", token="tok", repo_type="model")

    @patch("classivore.publishing.hub.create_tag")
    @patch("classivore.publishing.hub.upload_folder")
    @patch("classivore.publishing.hub.init_repo")
    def test_readme_cleaned_up(self, mock_init, mock_upload, mock_tag, tmp_path):
        _make_model_dir(tmp_path)
        mock_upload.return_value = MagicMock(commit_url="https://hf.co/commit/abc")
        publish_model(tmp_path, "org/repo", "v1.0.0", token="tok")
        # README.md should be cleaned up after publish
        assert not (tmp_path / "README.md").exists()
