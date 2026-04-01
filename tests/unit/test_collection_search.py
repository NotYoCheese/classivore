#!/usr/bin/env python3
"""Tests for Brave Search API client."""

import time
from unittest.mock import MagicMock, patch

import pytest

from classivore.collection.search import (
    BRAVE_API_URL,
    parse_search_results,
    search_brave,
)


SAMPLE_RESPONSE = {
    "web": {
        "results": [
            {
                "url": "https://example.com/article-1",
                "title": "Great Article",
                "description": "An article about stuff.",
            },
            {
                "url": "https://other.com/post-2",
                "title": "Another Post",
                "description": "More content here.",
            },
            {
                "url": "https://third.com/page-3",
                "title": "Third Result",
                "description": "Even more content.",
            },
        ]
    }
}

EMPTY_RESPONSE = {"web": {"results": []}}

NO_WEB_RESPONSE = {"query": {"original": "test"}}


class TestParseSearchResults:
    def test_extracts_urls(self):
        results = parse_search_results(SAMPLE_RESPONSE)
        assert len(results) == 3
        assert results[0]["url"] == "https://example.com/article-1"
        assert results[1]["url"] == "https://other.com/post-2"

    def test_extracts_title_and_description(self):
        results = parse_search_results(SAMPLE_RESPONSE)
        assert results[0]["title"] == "Great Article"
        assert results[0]["description"] == "An article about stuff."

    def test_empty_results(self):
        results = parse_search_results(EMPTY_RESPONSE)
        assert results == []

    def test_missing_web_key(self):
        results = parse_search_results(NO_WEB_RESPONSE)
        assert results == []

    def test_missing_results_key(self):
        results = parse_search_results({"web": {}})
        assert results == []


class TestSearchBrave:
    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_successful_search(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_resp

        results = search_brave("test query", api_key="test-key")
        assert len(results) == 3

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_sends_api_key_header(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_resp

        search_brave("test query", api_key="my-secret-key")
        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers["X-Subscription-Token"] == "my-secret-key"

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_rate_limits(self, mock_get, mock_sleep):
        """Sleeps 1 second after each request to respect rate limit."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_resp

        search_brave("query 1", api_key="key")
        mock_sleep.assert_called_with(1)

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_returns_empty_on_error(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        results = search_brave("test query", api_key="key")
        assert results == []

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_returns_empty_on_exception(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("Network error")

        results = search_brave("test query", api_key="key")
        assert results == []

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_passes_count_param(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_resp

        search_brave("test query", api_key="key", count=20)
        params = mock_get.call_args.kwargs.get("params", {})
        assert params["count"] == 20

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_uses_correct_api_url(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_resp

        search_brave("test query", api_key="key")
        url = mock_get.call_args.args[0]
        assert url == BRAVE_API_URL
