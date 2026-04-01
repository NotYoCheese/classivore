#!/usr/bin/env python3
"""Tests for the taxonomy enricher."""

from unittest.mock import MagicMock

import pytest

from classivore.taxonomy.enricher import (
    SYSTEM_PROMPT,
    apply_results,
    build_batch_requests,
    build_prompt,
    parse_enrichment,
)
from classivore.taxonomy.loader import build_hierarchy


def _make_categories():
    """Create a small test taxonomy."""
    return [
        {
            "id": "1", "name": "Automotive", "display_name": "Automotive: Automotive",
            "parent_id": "", "path": ["Automotive"], "depth": 1,
            "is_leaf": False, "children_count": 2, "description": "", "boundaries": "",
        },
        {
            "id": "2", "name": "Sedan", "display_name": "Automotive: Sedan",
            "parent_id": "1", "path": ["Automotive", "Sedan"], "depth": 2,
            "is_leaf": True, "children_count": 0, "description": "", "boundaries": "",
        },
        {
            "id": "3", "name": "SUV", "display_name": "Automotive: SUV",
            "parent_id": "1", "path": ["Automotive", "SUV"], "depth": 2,
            "is_leaf": True, "children_count": 0, "description": "", "boundaries": "",
        },
    ]


def _make_config(model="claude-haiku-4-5-20251001", max_tokens=150):
    config = MagicMock()
    config.enrichment_model = model
    config.enrichment_max_tokens = max_tokens
    return config


class TestBuildPrompt:
    def test_leaf_node(self):
        cats = _make_categories()
        sedan = cats[1]
        messages = build_prompt(sedan, siblings=["SUV"], children=[])

        assert len(messages) == 1
        content = messages[0]["content"]
        assert "Automotive: Sedan" in content
        assert "SUV" in content
        assert "None (leaf node)" in content

    def test_parent_node(self):
        cats = _make_categories()
        root = cats[0]
        messages = build_prompt(root, siblings=[], children=["Sedan", "SUV"])

        content = messages[0]["content"]
        assert "Sedan, SUV" in content
        assert "None (only child or root)" in content

    def test_root_no_siblings(self):
        cats = _make_categories()
        root = cats[0]
        messages = build_prompt(root, siblings=[], children=["Sedan"])

        content = messages[0]["content"]
        assert "None (only child or root)" in content


class TestBuildBatchRequests:
    def test_creates_requests_for_all(self):
        cats = _make_categories()
        hierarchy = build_hierarchy(cats)
        config = _make_config()

        requests = build_batch_requests(cats, hierarchy, config)

        assert len(requests) == 3
        assert requests[0]["custom_id"] == "cat-1"
        assert requests[0]["params"]["model"] == "claude-haiku-4-5-20251001"
        assert requests[0]["params"]["max_tokens"] == 150
        assert requests[0]["params"]["system"] == SYSTEM_PROMPT

    def test_skips_already_enriched(self):
        cats = _make_categories()
        cats[0]["description"] = "Already enriched"
        hierarchy = build_hierarchy(cats)
        config = _make_config()

        requests = build_batch_requests(cats, hierarchy, config)

        assert len(requests) == 2
        ids = [r["custom_id"] for r in requests]
        assert "cat-1" not in ids

    def test_uses_config_model(self):
        cats = _make_categories()
        hierarchy = build_hierarchy(cats)
        config = _make_config(model="claude-sonnet-4-6", max_tokens=200)

        requests = build_batch_requests(cats, hierarchy, config)

        assert requests[0]["params"]["model"] == "claude-sonnet-4-6"
        assert requests[0]["params"]["max_tokens"] == 200


class TestParseEnrichment:
    def test_json_response(self):
        message = MagicMock()
        block = MagicMock()
        block.text = '{"description": "Content about vehicles.", "boundary": "Distinguished from Travel by focus on vehicles."}'
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == "Content about vehicles."
        assert bounds == "Distinguished from Travel by focus on vehicles."

    def test_json_with_code_fences(self):
        message = MagicMock()
        block = MagicMock()
        block.text = '```json\n{"description": "About vehicles.", "boundary": "Not about travel."}\n```'
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == "About vehicles."
        assert bounds == "Not about travel."

    def test_json_missing_boundary(self):
        message = MagicMock()
        block = MagicMock()
        block.text = '{"description": "Content about vehicles."}'
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == "Content about vehicles."
        assert bounds == ""

    def test_fallback_two_lines(self):
        message = MagicMock()
        block = MagicMock()
        block.text = "Content about vehicles.\nDistinguished from Travel."
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == "Content about vehicles."
        assert bounds == "Distinguished from Travel."

    def test_fallback_single_line(self):
        message = MagicMock()
        block = MagicMock()
        block.text = "Content about vehicles."
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == "Content about vehicles."
        assert bounds == ""

    def test_empty_response(self):
        message = MagicMock()
        block = MagicMock()
        block.text = ""
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert desc == ""
        assert bounds == ""

    def test_fallback_strips_markdown_headings(self):
        message = MagicMock()
        block = MagicMock()
        block.text = "# Category Name\nFootage captured by cameras.\nDistinguished from Auto Shows."
        message.content = [block]

        desc, bounds = parse_enrichment(message)

        assert not desc.startswith("#")
        assert desc == "Footage captured by cameras."
        assert bounds == "Distinguished from Auto Shows."

    def test_multiple_content_blocks(self):
        message = MagicMock()
        block1 = MagicMock()
        block1.text = '{"description": "Part one."'
        block2 = MagicMock()
        block2.text = ', "boundary": "Part two."}'
        message.content = [block1, block2]

        desc, bounds = parse_enrichment(message)

        assert desc == "Part one."
        assert bounds == "Part two."


class TestApplyResults:
    def test_applies_to_matching_categories(self):
        cats = _make_categories()
        results = {
            "1": ("About vehicles.", "Not about travel."),
            "2": ("Four-door cars.", "Distinguished from SUV by enclosed trunk."),
        }

        apply_results(cats, results)

        assert cats[0]["description"] == "About vehicles."
        assert cats[0]["boundaries"] == "Not about travel."
        assert cats[1]["description"] == "Four-door cars."
        assert cats[2]["description"] == ""  # Not in results

    def test_preserves_existing_when_not_in_results(self):
        cats = _make_categories()
        cats[2]["description"] = "Existing description"
        results = {"1": ("New desc.", "New bounds.")}

        apply_results(cats, results)

        assert cats[2]["description"] == "Existing description"
