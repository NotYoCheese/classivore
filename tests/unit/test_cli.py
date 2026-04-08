#!/usr/bin/env python3
"""Tests for CLI argument parsing and subcommand routing."""

import pytest
from unittest.mock import patch
from classivore.cli.main import main


class TestCLIParsing:
    """Test that CLI commands parse arguments correctly."""

    def test_no_command_exits(self):
        """Running with no command prints help and exits."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["classivore"]):
                main()
        assert exc_info.value.code == 1

    def test_classify_text(self, capsys):
        """classify --text loads model and runs prediction."""
        from unittest.mock import MagicMock
        mock_cls = MagicMock()
        mock_cls.return_value.predict.return_value = [
            {"name": "Sedan", "id": "4", "confidence": 0.92, "path": ["Automotive", "Sedan"]},
        ]
        with patch.dict("sys.modules", {"classivore.inference": MagicMock(Classifier=mock_cls)}):
            with patch("sys.argv", ["classivore", "classify", "--text", "test content",
                                     "--model-dir", "/tmp/fake-model"]):
                main()
        captured = capsys.readouterr()
        assert "Sedan" in captured.out
        assert "0.92" in captured.out

    def test_train_dry_run(self, capsys):
        """train --dry-run shows data summary."""
        with patch("sys.argv", ["classivore", "train", "--dry-run"]):
            main()
        captured = capsys.readouterr()
        assert "Dry Run" in captured.out
        assert "iab-2.2" in captured.out

    def test_collect_audit_domains(self, capsys):
        """collect --audit-domains runs without error."""
        with patch("sys.argv", ["classivore", "collect", "--audit-domains"]):
            main()
        captured = capsys.readouterr()
        assert "domain" in captured.out.lower() or "No domain" in captured.out
