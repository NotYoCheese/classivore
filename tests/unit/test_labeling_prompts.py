#!/usr/bin/env python3
"""Tests for labeling prompt construction."""

import pytest

from classivore.labeling.prompts import (
    FEW_SHOT_EXAMPLES,
    build_stage1_system,
    build_stage2_system,
    build_user_message,
    get_subtree_categories,
)


def _make_tier1_categories():
    return [
        {
            "name": "Automotive", "depth": 1, "path": ["Automotive"],
            "description": "Content about vehicles.", "boundaries": "Focus on vehicles.",
        },
        {
            "name": "Science", "depth": 1, "path": ["Science"],
            "description": "Scientific research.", "boundaries": "Academic and research content.",
        },
    ]


def _make_full_taxonomy():
    return [
        {"name": "Automotive", "depth": 1, "path": ["Automotive"], "description": "Vehicles.", "boundaries": "B1."},
        {"name": "Auto Body Styles", "depth": 2, "path": ["Automotive", "Auto Body Styles"], "description": "Body types.", "boundaries": "B2."},
        {"name": "Sedan", "depth": 3, "path": ["Automotive", "Auto Body Styles", "Sedan"], "description": "Four-door.", "boundaries": "B3."},
        {"name": "SUV", "depth": 3, "path": ["Automotive", "Auto Body Styles", "SUV"], "description": "Sport utility.", "boundaries": "B4."},
        {"name": "Science", "depth": 1, "path": ["Science"], "description": "Research.", "boundaries": "B5."},
        {"name": "Physics", "depth": 2, "path": ["Science", "Physics"], "description": "Physical sciences.", "boundaries": "B6."},
    ]


class TestBuildStage1System:
    def test_includes_all_tier1(self):
        cats = _make_tier1_categories()
        prompt = build_stage1_system(cats)
        assert "Automotive" in prompt
        assert "Science" in prompt

    def test_includes_descriptions(self):
        cats = _make_tier1_categories()
        prompt = build_stage1_system(cats)
        assert "Content about vehicles" in prompt
        assert "Scientific research" in prompt

    def test_includes_boundaries(self):
        cats = _make_tier1_categories()
        prompt = build_stage1_system(cats)
        assert "Focus on vehicles" in prompt
        assert "Academic and research" in prompt

    def test_uses_name_not_display_name(self):
        cats = [{"name": "Sedan", "display_name": "Automotive: Sedan", "depth": 1, "path": ["Sedan"], "description": "D", "boundaries": "B"}]
        prompt = build_stage1_system(cats)
        assert 'name="Sedan"' in prompt
        assert "Automotive: Sedan" not in prompt

    def test_includes_category_count(self):
        cats = _make_tier1_categories()
        prompt = build_stage1_system(cats)
        assert "2 top-level" in prompt

    def test_json_output_format(self):
        cats = _make_tier1_categories()
        prompt = build_stage1_system(cats)
        assert '"categories"' in prompt
        assert '"confidence"' in prompt


class TestBuildStage2System:
    def test_includes_subtree_categories(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert "Sedan" in prompt
        assert "SUV" in prompt
        assert "Auto Body Styles" in prompt

    def test_excludes_unselected_subtrees(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert "Physics" not in prompt

    def test_includes_multiple_subtrees(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive", "Science"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert "Sedan" in prompt
        assert "Physics" in prompt

    def test_includes_depth_and_parent(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert 'depth="3"' in prompt
        assert 'parent="Auto Body Styles"' in prompt

    def test_includes_descriptions_and_boundaries(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert "Four-door" in prompt
        assert "B3." in prompt

    def test_includes_few_shot_examples(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.5)
        assert "<example>" in prompt
        assert "Honda Civic" in prompt

    def test_injects_max_labels(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=5, min_confidence=0.5)
        assert "up to 5 categories" in prompt

    def test_injects_min_confidence(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        prompt = build_stage2_system(subtrees, max_labels=3, min_confidence=0.4)
        assert "above 0.4" in prompt


class TestBuildUserMessage:
    def test_includes_text(self):
        msg = build_user_message("Article about cars.")
        assert "Article about cars" in msg

    def test_includes_url(self):
        msg = build_user_message("Text.", url="https://example.com")
        assert "https://example.com" in msg

    def test_no_url(self):
        msg = build_user_message("Text.")
        assert "URL:" not in msg

    def test_truncates_long_text(self):
        long_text = " ".join(f"word{i}" for i in range(5000))
        msg = build_user_message(long_text, truncation_words=100)
        assert "word99" in msg
        assert "word100" not in msg


class TestGetSubtreeCategories:
    def test_single_subtree(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive"})
        assert "Automotive" in subtrees
        assert len(subtrees["Automotive"]) == 4  # Automotive + Auto Body Styles + Sedan + SUV

    def test_multiple_subtrees(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Automotive", "Science"})
        assert len(subtrees) == 2
        assert len(subtrees["Science"]) == 2  # Science + Physics

    def test_empty_selection(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, set())
        assert subtrees == {}

    def test_nonexistent_tier1(self):
        taxonomy = _make_full_taxonomy()
        subtrees = get_subtree_categories(taxonomy, {"Nonexistent"})
        assert subtrees["Nonexistent"] == []


class TestFewShotExamples:
    def test_three_examples(self):
        assert len(FEW_SHOT_EXAMPLES) == 3

    def test_each_has_text_and_response(self):
        for ex in FEW_SHOT_EXAMPLES:
            assert "text" in ex
            assert "response" in ex
            assert "reasoning" in ex["response"]
            assert "categories" in ex["response"]
