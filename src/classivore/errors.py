#!/usr/bin/env python3
"""Typed exception hierarchy for cross-module error propagation."""


class ClassivoreError(Exception):
    """Base for all classivore errors."""


class ConfigError(ClassivoreError):
    """Invalid or missing configuration."""


class CorpusError(ClassivoreError):
    """Problems with corpus data (missing, corrupt, empty)."""


class BatchAPIError(ClassivoreError):
    """Batch API submission or polling failure."""


class SearchExhaustedError(ClassivoreError):
    """All search providers exhausted with no results."""


class BudgetExhaustedError(ClassivoreError):
    """Agent budget (iterations, API calls, cost) exceeded."""
