#!/usr/bin/env python3
"""Tests for search query generation."""

from unittest.mock import MagicMock

import pytest

from classivore.collection.queries import (
    build_llm_prompt,
    generate_template_queries,
    parse_llm_queries,
)


def _make_category(name="Sedan", description="Four-door passenger cars.", path=None):
    return {
        "id": "3",
        "name": name,
        "display_name": f"Automotive: {name}",
        "description": description,
        "boundaries": "Distinguished from SUV by enclosed trunk.",
        "path": path or ["Automotive", "Auto Body Styles", name],
        "depth": 3,
        "is_leaf": True,
        "children_count": 0,
    }


class TestGenerateTemplateQueries:
    def test_generates_queries(self):
        cat = _make_category()
        queries = generate_template_queries(cat)
        assert len(queries) >= 2

    def test_includes_category_name(self):
        cat = _make_category()
        queries = generate_template_queries(cat)
        assert any("Sedan" in q for q in queries)

    def test_includes_description_keywords(self):
        cat = _make_category()
        queries = generate_template_queries(cat)
        # Should use keywords from description
        assert any("passenger" in q.lower() or "four-door" in q.lower() for q in queries)

    def test_includes_tier1_context(self):
        cat = _make_category()
        queries = generate_template_queries(cat)
        assert any("Automotive" in q for q in queries)

    def test_no_description_still_works(self):
        cat = _make_category(description="")
        queries = generate_template_queries(cat)
        assert len(queries) >= 1

    def test_excludes_tried_queries(self):
        cat = _make_category()
        all_queries = generate_template_queries(cat)
        # Exclude all but leave room for at least testing
        filtered = generate_template_queries(cat, tried=set(all_queries[:1]))
        assert len(filtered) < len(all_queries)

    def test_root_category(self):
        cat = _make_category(name="Automotive", path=["Automotive"])
        queries = generate_template_queries(cat)
        assert len(queries) >= 1


class TestBuildLlmPrompt:
    def test_includes_category_info(self):
        cat = _make_category()
        prompt = build_llm_prompt(
            cat,
            siblings=["SUV", "Coupe"],
            tried_queries=["sedan article automotive"],
        )
        content = prompt[0]["content"]
        assert "Sedan" in content
        assert "Four-door passenger cars" in content

    def test_includes_siblings(self):
        cat = _make_category()
        prompt = build_llm_prompt(cat, siblings=["SUV", "Coupe"], tried_queries=[])
        content = prompt[0]["content"]
        assert "SUV" in content

    def test_includes_tried_queries(self):
        cat = _make_category()
        prompt = build_llm_prompt(
            cat, siblings=[], tried_queries=["sedan article", "sedan review"]
        )
        content = prompt[0]["content"]
        assert "sedan article" in content
        assert "sedan review" in content

    def test_no_tried_queries(self):
        cat = _make_category()
        prompt = build_llm_prompt(cat, siblings=[], tried_queries=[])
        content = prompt[0]["content"]
        assert "None yet" in content or "none" in content.lower()


class TestParseLlmQueries:
    def test_parses_numbered_list(self):
        text = "1. sedan buying guide 2026\n2. best sedans for families\n3. sedan vs coupe comparison\n4. luxury sedan reviews\n5. affordable sedan options"
        queries = parse_llm_queries(text)
        assert len(queries) == 5
        assert queries[0] == "sedan buying guide 2026"

    def test_parses_dash_list(self):
        text = "- sedan buying guide\n- best sedans for families\n- sedan reviews"
        queries = parse_llm_queries(text)
        assert len(queries) == 3

    def test_parses_plain_lines(self):
        text = "sedan buying guide\nbest sedans for families\nsedan reviews"
        queries = parse_llm_queries(text)
        assert len(queries) == 3

    def test_strips_quotes(self):
        text = '1. "sedan buying guide 2026"\n2. "best sedans"'
        queries = parse_llm_queries(text)
        assert queries[0] == "sedan buying guide 2026"

    def test_skips_empty_lines(self):
        text = "sedan buying guide\n\nbest sedans\n\n"
        queries = parse_llm_queries(text)
        assert len(queries) == 2

    def test_empty_input(self):
        queries = parse_llm_queries("")
        assert queries == []

    def test_skips_preamble(self):
        text = "Here are 5 search queries:\n1. sedan guide\n2. sedan review"
        queries = parse_llm_queries(text)
        assert len(queries) == 2
        assert "Here are" not in queries[0]
