#!/usr/bin/env python3
"""Tests for search providers and SearchClient."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from classivore.collection.search import (
    BRAVE_API_URL,
    BRAVE_REQUEST_INTERVAL,
    SERPER_API_URL,
    SearchClient,
    _parse_brave_results,
    _parse_serper_results,
    search_brave,
    search_serper,
)


# --- Sample responses ---

BRAVE_RESPONSE = {
    "web": {
        "results": [
            {"url": "https://example.com/a", "title": "Article A", "description": "Desc A"},
            {"url": "https://example.com/b", "title": "Article B", "description": "Desc B"},
        ]
    }
}

SERPER_RESPONSE = {
    "organic": [
        {"link": "https://example.com/a", "title": "Article A", "snippet": "Snippet A"},
        {"link": "https://example.com/b", "title": "Article B", "snippet": "Snippet B"},
    ]
}


def _make_brave_resp(status=200, data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data or BRAVE_RESPONSE
    resp.headers = headers or {}
    return resp


def _make_serper_resp(status=200, data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data or SERPER_RESPONSE
    resp.headers = {}
    return resp


# --- Brave parsing ---

class TestParseBraveResults:
    def test_extracts_results(self):
        results = _parse_brave_results(BRAVE_RESPONSE)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["title"] == "Article A"

    def test_empty_response(self):
        assert _parse_brave_results({"web": {"results": []}}) == []

    def test_missing_web_key(self):
        assert _parse_brave_results({}) == []


# --- Serper parsing ---

class TestParseSerperResults:
    def test_extracts_results(self):
        results = _parse_serper_results(SERPER_RESPONSE)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["description"] == "Snippet A"

    def test_empty_response(self):
        assert _parse_serper_results({"organic": []}) == []

    def test_missing_organic_key(self):
        assert _parse_serper_results({}) == []


# --- Brave provider ---

class TestSearchBrave:
    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_successful_search(self, mock_get, mock_sleep):
        mock_get.return_value = _make_brave_resp()
        results = search_brave("test query", api_key="key")
        assert len(results) == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_sends_api_key(self, mock_get, mock_sleep):
        mock_get.return_value = _make_brave_resp()
        search_brave("test", api_key="my-key")
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["X-Subscription-Token"] == "my-key"

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _make_brave_resp(status=429, headers={}),
            _make_brave_resp(),
        ]
        results = search_brave("test", api_key="key")
        assert len(results) == 2
        assert mock_get.call_count == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.ConnectionError("DNS failed"),
            _make_brave_resp(),
        ]
        results = search_brave("test", api_key="key")
        assert len(results) == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_retries_on_timeout(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.Timeout("timed out"),
            _make_brave_resp(),
        ]
        results = search_brave("test", api_key="key")
        assert len(results) == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_retries_on_dns_failure(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            OSError("Name or service not known"),
            _make_brave_resp(),
        ]
        results = search_brave("test", api_key="key")
        assert len(results) == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_returns_none_on_transient_exhaustion(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.ConnectionError("DNS failed")
        results = search_brave("test", api_key="key")
        assert results is None
        assert mock_get.call_count == 3

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_returns_empty_on_500(self, mock_get, mock_sleep):
        mock_get.return_value = _make_brave_resp(status=500)
        results = search_brave("test", api_key="key")
        assert results == []

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.get")
    def test_warns_on_low_quota(self, mock_get, mock_sleep, caplog):
        import logging
        resp = _make_brave_resp(headers={"X-RateLimit-Remaining": "1, 200"})
        mock_get.return_value = resp
        with caplog.at_level(logging.WARNING):
            search_brave("test", api_key="key")
        assert any("monthly quota low" in r.message for r in caplog.records)


# --- Serper provider ---

class TestSearchSerper:
    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_successful_search(self, mock_post, mock_sleep):
        mock_post.return_value = _make_serper_resp()
        results = search_serper("test query", api_key="key")
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_sends_api_key(self, mock_post, mock_sleep):
        mock_post.return_value = _make_serper_resp()
        search_serper("test", api_key="my-key")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-API-KEY"] == "my-key"

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_posts_json_body(self, mock_post, mock_sleep):
        mock_post.return_value = _make_serper_resp()
        search_serper("test query", api_key="key", count=15)
        body = mock_post.call_args.kwargs["json"]
        assert body["q"] == "test query"
        assert body["num"] == 15

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            requests.ConnectionError("failed"),
            _make_serper_resp(),
        ]
        results = search_serper("test", api_key="key")
        assert len(results) == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_returns_none_on_transient_exhaustion(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.Timeout("timed out")
        results = search_serper("test", api_key="key")
        assert results is None

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search.requests.post")
    def test_returns_empty_on_500(self, mock_post, mock_sleep):
        mock_post.return_value = _make_serper_resp(status=500)
        results = search_serper("test", api_key="key")
        assert results == []


# --- SearchClient ---

class TestSearchClient:
    def _make_provider(self, name, fn):
        return {"name": name, "fn": fn, "api_key": "test-key"}

    def test_uses_first_provider(self):
        def brave_fn(q, k, c):
            return [{"url": "https://brave.com/a", "title": "Brave", "description": ""}]

        def serper_fn(q, k, c):
            return [{"url": "https://serper.com/a", "title": "Serper", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn), self._make_provider("serper", serper_fn)]
        client.exhausted = set()

        results = client.search("test")
        assert results[0]["url"] == "https://brave.com/a"

    def test_falls_through_on_none(self):
        def brave_fn(q, k, c):
            return None  # transient failure

        def serper_fn(q, k, c):
            return [{"url": "https://serper.com/a", "title": "Serper", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn), self._make_provider("serper", serper_fn)]
        client.exhausted = set()

        results = client.search("test")
        assert results[0]["url"] == "https://serper.com/a"
        assert "brave" in client.exhausted

    def test_returns_none_when_all_exhausted(self):
        def failing_fn(q, k, c):
            return None

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", failing_fn), self._make_provider("serper", failing_fn)]
        client.exhausted = set()

        results = client.search("test")
        assert results is None

    def test_skips_exhausted_providers(self):
        call_count = {"brave": 0, "serper": 0}

        def brave_fn(q, k, c):
            call_count["brave"] += 1
            return [{"url": "https://brave.com/a", "title": "", "description": ""}]

        def serper_fn(q, k, c):
            call_count["serper"] += 1
            return [{"url": "https://serper.com/a", "title": "", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn), self._make_provider("serper", serper_fn)]
        client.exhausted = {"brave"}

        client.search("test")
        assert call_count["brave"] == 0
        assert call_count["serper"] == 1

    def test_reset_exhausted(self):
        client = SearchClient.__new__(SearchClient)
        client.providers = []
        client.exhausted = {"brave", "serper"}
        client.reset_exhausted()
        assert client.exhausted == set()

    def test_empty_results_not_treated_as_failure(self):
        def brave_fn(q, k, c):
            return []  # successful but empty

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn)]
        client.exhausted = set()

        results = client.search("test")
        assert results == []
        assert "brave" not in client.exhausted

    def test_active_provider_count(self):
        client = SearchClient.__new__(SearchClient)
        client.providers = [
            self._make_provider("brave", lambda q, k, c: []),
            self._make_provider("serper", lambda q, k, c: []),
        ]
        client.exhausted = set()
        assert client.active_provider_count == 2
        client.exhausted.add("brave")
        assert client.active_provider_count == 1

    @patch.dict("os.environ", {"BRAVE_API_KEY": "test-brave", "SERPER_API_KEY": "test-serper"})
    def test_from_config_default_providers(self):
        config = MagicMock()
        config.search_providers = None
        client = SearchClient.from_config(config)
        assert len(client.providers) == 2
        assert client.providers[0]["name"] == "brave"
        assert client.providers[1]["name"] == "serper"

    @patch.dict("os.environ", {"BRAVE_API_KEY": "test-brave"}, clear=False)
    def test_from_config_skips_missing_keys(self):
        config = MagicMock()
        config.search_providers = [
            {"name": "brave", "api_key_env": "BRAVE_API_KEY"},
            {"name": "serper", "api_key_env": "MISSING_KEY"},
        ]
        # Ensure MISSING_KEY is not set
        import os
        os.environ.pop("MISSING_KEY", None)
        client = SearchClient.from_config(config)
        assert len(client.providers) == 1
        assert client.providers[0]["name"] == "brave"
