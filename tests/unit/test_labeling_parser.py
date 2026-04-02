#!/usr/bin/env python3
"""Tests for labeling response parser."""

from unittest.mock import MagicMock

import pytest

from classivore.labeling.parser import (
    parse_stage1_response,
    parse_stage2_response,
    validate_category_name,
)


def _make_message(text):
    msg = MagicMock()
    block = MagicMock()
    block.text = text
    msg.content = [block]
    return msg


class TestParseStage1:
    def test_valid_json(self):
        msg = _make_message('{"categories": [{"name": "Automotive", "confidence": 0.9}]}')
        result = parse_stage1_response(msg)
        assert len(result) == 1
        assert result[0]["name"] == "Automotive"
        assert result[0]["confidence"] == 0.9

    def test_multiple_categories(self):
        msg = _make_message('{"categories": [{"name": "Automotive", "confidence": 0.9}, {"name": "Science", "confidence": 0.5}]}')
        result = parse_stage1_response(msg)
        assert len(result) == 2

    def test_with_code_fences(self):
        msg = _make_message('```json\n{"categories": [{"name": "Tech", "confidence": 0.8}]}\n```')
        result = parse_stage1_response(msg)
        assert len(result) == 1
        assert result[0]["name"] == "Tech"

    def test_empty_categories(self):
        msg = _make_message('{"categories": []}')
        result = parse_stage1_response(msg)
        assert result == []

    def test_malformed_json(self):
        msg = _make_message('not json at all')
        result = parse_stage1_response(msg)
        assert result == []

    def test_list_fallback(self):
        """If top-level is a list, treat as categories."""
        msg = _make_message('[{"name": "Automotive", "confidence": 0.9}]')
        result = parse_stage1_response(msg)
        assert len(result) == 1

    def test_multiple_content_blocks(self):
        msg = MagicMock()
        b1 = MagicMock()
        b1.text = '{"categories": [{"name": "Tech"'
        b2 = MagicMock()
        b2.text = ', "confidence": 0.8}]}'
        msg.content = [b1, b2]
        result = parse_stage1_response(msg)
        assert len(result) == 1


class TestParseStage2:
    def test_valid_json(self):
        msg = _make_message('{"reasoning": "About cars.", "categories": [{"name": "Sedan", "confidence": 0.95}]}')
        valid = {"Sedan", "SUV", "Automotive"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert result["reasoning"] == "About cars."
        assert len(result["categories"]) == 1
        assert result["categories"][0]["name"] == "Sedan"

    def test_filters_invalid_names(self):
        msg = _make_message('{"reasoning": "R", "categories": [{"name": "Sedan", "confidence": 0.9}, {"name": "Nonexistent", "confidence": 0.8}]}')
        valid = {"Sedan"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert len(result["categories"]) == 1

    def test_filters_by_min_confidence(self):
        msg = _make_message('{"reasoning": "R", "categories": [{"name": "Sedan", "confidence": 0.9}, {"name": "SUV", "confidence": 0.3}]}')
        valid = {"Sedan", "SUV"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert len(result["categories"]) == 1
        assert result["categories"][0]["name"] == "Sedan"

    def test_enforces_max_labels(self):
        msg = _make_message('{"reasoning": "R", "categories": [{"name": "A", "confidence": 0.9}, {"name": "B", "confidence": 0.8}, {"name": "C", "confidence": 0.7}, {"name": "D", "confidence": 0.6}]}')
        valid = {"A", "B", "C", "D"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert len(result["categories"]) == 3
        # Should keep highest confidence
        assert result["categories"][0]["name"] == "A"

    def test_with_code_fences(self):
        msg = _make_message('```json\n{"reasoning": "R", "categories": [{"name": "Tech", "confidence": 0.8}]}\n```')
        valid = {"Tech"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert len(result["categories"]) == 1

    def test_malformed_returns_error(self):
        msg = _make_message('garbage')
        result = parse_stage2_response(msg, set(), min_confidence=0.5, max_labels=3)
        assert result["categories"] == []
        assert "error" in result

    def test_missing_reasoning(self):
        msg = _make_message('{"categories": [{"name": "Sedan", "confidence": 0.9}]}')
        valid = {"Sedan"}
        result = parse_stage2_response(msg, valid, min_confidence=0.5, max_labels=3)
        assert result["reasoning"] == ""
        assert len(result["categories"]) == 1


class TestValidateCategoryName:
    def test_exact_match(self):
        valid = {"Sedan", "SUV", "Coupe"}
        assert validate_category_name("Sedan", valid) == "Sedan"

    def test_case_insensitive(self):
        valid = {"Sedan", "SUV"}
        assert validate_category_name("sedan", valid) == "Sedan"
        assert validate_category_name("SEDAN", valid) == "Sedan"

    def test_display_name_correction(self):
        """If model outputs 'Automotive: Sedan', extract 'Sedan'."""
        valid = {"Sedan", "SUV"}
        assert validate_category_name("Automotive: Sedan", valid) == "Sedan"

    def test_display_name_case_insensitive(self):
        valid = {"Sedan"}
        assert validate_category_name("Automotive: sedan", valid) == "Sedan"

    def test_no_match_returns_none(self):
        valid = {"Sedan", "SUV"}
        assert validate_category_name("Hatchback", valid) is None

    def test_empty_name(self):
        assert validate_category_name("", {"Sedan"}) is None
