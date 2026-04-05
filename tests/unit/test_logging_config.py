#!/usr/bin/env python3
"""Tests for shared structlog configuration."""

import json
import logging

import structlog

from classivore.logging_config import configure_logging, get_logger


class TestConfigureLogging:
    """Test logging configuration."""

    def test_verbose_sets_debug_level(self):
        configure_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_non_verbose_sets_info_level(self):
        configure_logging(verbose=False)
        assert logging.getLogger().level == logging.INFO

    def test_json_output_produces_valid_json(self, capsys):
        configure_logging(json_output=True)
        logger = get_logger("test.json")
        logger.info("test_event", key="value")

        captured = capsys.readouterr()
        line = captured.err.strip()
        parsed = json.loads(line)
        assert parsed["event"] == "test_event"
        assert parsed["key"] == "value"
        assert "timestamp" in parsed

    def test_console_output_has_timestamp(self, capsys):
        configure_logging(json_output=False)
        logger = get_logger("test.console")
        logger.info("hello")

        captured = capsys.readouterr()
        # Console renderer includes ISO timestamp
        assert "hello" in captured.err
        # Timestamp format: YYYY-MM-DD
        assert "20" in captured.err

    def test_context_vars_propagate(self, capsys):
        configure_logging(json_output=True)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(run_id="test-123")

        logger = get_logger("test.ctx")
        logger.info("with_context")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["run_id"] == "test-123"

        structlog.contextvars.clear_contextvars()

    def test_get_logger_returns_bound_logger(self):
        configure_logging()
        logger = get_logger("my.module")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
