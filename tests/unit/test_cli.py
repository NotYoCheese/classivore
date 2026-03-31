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
        """classify --text passes text argument."""
        with patch("sys.argv", ["classivore", "classify", "--text", "test content"]):
            main()
        captured = capsys.readouterr()
        assert "classify" in captured.out.lower()

    def test_train_default_taxonomy(self, capsys):
        """train defaults to iab-2.2 taxonomy."""
        with patch("sys.argv", ["classivore", "train"]):
            main()
        captured = capsys.readouterr()
        assert "iab-2.2" in captured.out

    def test_collect_pages(self, capsys):
        """collect --pages passes page count."""
        with patch("sys.argv", ["classivore", "collect", "--pages", "500"]):
            main()
        captured = capsys.readouterr()
        assert "500" in captured.out

    def test_taxonomy_override(self, capsys):
        """--taxonomy flag overrides default."""
        with patch("sys.argv", ["classivore", "train", "--taxonomy", "iptc-media"]):
            main()
        captured = capsys.readouterr()
        assert "iptc-media" in captured.out
