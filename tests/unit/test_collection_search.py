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
    _parse_exa_results,
    _parse_serper_results,
    fetch_exa_content,
    search_brave,
    search_exa,
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
        assert any("brave_quota_low" in str(r.message) for r in caplog.records)


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


# --- Exa parsing ---

def _make_exa_result(url="https://example.com/a", title="Article A", text="Full article text."):
    r = MagicMock()
    r.url = url
    r.title = title
    r.text = text
    return r


def _make_exa_response(results=None):
    resp = MagicMock()
    resp.results = [_make_exa_result()] if results is None else results
    return resp


class TestParseExaResults:
    def test_extracts_results(self):
        response = _make_exa_response([
            _make_exa_result("https://example.com/a", "Article A", "Text A"),
            _make_exa_result("https://example.com/b", "Article B", "Text B"),
        ])
        results = _parse_exa_results(response)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["title"] == "Article A"
        assert results[0]["text"] == "Text A"

    def test_includes_text_field(self):
        response = _make_exa_response([_make_exa_result(text="Full page content here.")])
        results = _parse_exa_results(response)
        assert results[0]["text"] == "Full page content here."

    def test_empty_response(self):
        response = _make_exa_response([])
        assert _parse_exa_results(response) == []

    def test_skips_results_without_url(self):
        r = _make_exa_result()
        r.url = None
        response = _make_exa_response([r])
        assert _parse_exa_results(response) == []

    def test_handles_none_text(self):
        response = _make_exa_response([_make_exa_result(text=None)])
        results = _parse_exa_results(response)
        assert results[0]["text"] == ""

    def test_handles_none_title(self):
        response = _make_exa_response([_make_exa_result(title=None)])
        results = _parse_exa_results(response)
        assert results[0]["title"] == ""


# --- Exa provider ---

class TestSearchExa:
    def _make_mock_exa(self, response=None):
        mock_exa = MagicMock()
        mock_exa.search_and_contents.return_value = response or _make_exa_response()
        return mock_exa

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search._Exa")
    def test_successful_search(self, mock_exa_cls, mock_sleep):
        mock_exa_cls.return_value = self._make_mock_exa()
        results = search_exa("test query", api_key="key")
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["text"] == "Full article text."

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search._Exa")
    def test_passes_query_and_count(self, mock_exa_cls, mock_sleep):
        mock_exa = self._make_mock_exa()
        mock_exa_cls.return_value = mock_exa
        search_exa("sedan guide", api_key="key", count=5)
        call_kwargs = mock_exa.search_and_contents.call_args
        assert call_kwargs.args[0] == "sedan guide"
        assert call_kwargs.kwargs["num_results"] == 5

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search._Exa")
    def test_returns_none_on_auth_error(self, mock_exa_cls, mock_sleep):
        mock_exa = MagicMock()
        mock_exa.search_and_contents.side_effect = Exception("401 Unauthorized")
        mock_exa_cls.return_value = mock_exa
        result = search_exa("test", api_key="bad-key")
        assert result is None
        assert mock_exa.search_and_contents.call_count == 1  # no retry on auth error

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search._Exa")
    def test_retries_on_transient_error(self, mock_exa_cls, mock_sleep):
        mock_exa = MagicMock()
        mock_exa.search_and_contents.side_effect = [
            Exception("connection reset"),
            _make_exa_response(),
        ]
        mock_exa_cls.return_value = mock_exa
        results = search_exa("test", api_key="key")
        assert results is not None
        assert mock_exa.search_and_contents.call_count == 2

    @patch("classivore.collection.search.time.sleep")
    @patch("classivore.collection.search._Exa")
    def test_returns_none_after_retries_exhausted(self, mock_exa_cls, mock_sleep):
        mock_exa = MagicMock()
        mock_exa.search_and_contents.side_effect = Exception("network error")
        mock_exa_cls.return_value = mock_exa
        result = search_exa("test", api_key="key")
        assert result is None
        assert mock_exa.search_and_contents.call_count == 3

    @patch("classivore.collection.search._EXA_AVAILABLE", False)
    def test_returns_none_when_not_installed(self):
        result = search_exa("test", api_key="key")
        assert result is None


# --- Exa content fetch ---

class TestFetchExaContent:
    def _make_mock_exa_with_content(self, text="Page content here."):
        result = MagicMock()
        result.text = text
        response = MagicMock()
        response.results = [result]
        mock_exa = MagicMock()
        mock_exa.get_contents.return_value = response
        return mock_exa

    @patch("classivore.collection.search._Exa")
    def test_returns_text(self, mock_exa_cls):
        mock_exa_cls.return_value = self._make_mock_exa_with_content("Article text.")
        text = fetch_exa_content("https://example.com/article", api_key="key")
        assert text == "Article text."

    @patch("classivore.collection.search._Exa")
    def test_passes_url(self, mock_exa_cls):
        mock_exa = self._make_mock_exa_with_content()
        mock_exa_cls.return_value = mock_exa
        fetch_exa_content("https://example.com/article", api_key="key")
        mock_exa.get_contents.assert_called_once_with(
            ["https://example.com/article"],
            text={"max_characters": 20000},
        )

    @patch("classivore.collection.search._Exa")
    def test_returns_none_on_empty_results(self, mock_exa_cls):
        mock_exa = MagicMock()
        mock_exa.get_contents.return_value = MagicMock(results=[])
        mock_exa_cls.return_value = mock_exa
        result = fetch_exa_content("https://example.com/article", api_key="key")
        assert result is None

    @patch("classivore.collection.search._Exa")
    def test_returns_none_on_exception(self, mock_exa_cls):
        mock_exa = MagicMock()
        mock_exa.get_contents.side_effect = Exception("not found")
        mock_exa_cls.return_value = mock_exa
        result = fetch_exa_content("https://example.com/article", api_key="key")
        assert result is None

    @patch("classivore.collection.search._EXA_AVAILABLE", False)
    def test_returns_none_when_not_installed(self):
        result = fetch_exa_content("https://example.com/article", api_key="key")
        assert result is None


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

    @patch("classivore.collection.search.fetch_exa_content")
    def test_fetch_content_delegates_to_exa(self, mock_fetch):
        mock_fetch.return_value = "Article text."
        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("exa", search_exa)]
        client.exhausted = set()
        result = client.fetch_content("https://example.com/article")
        assert result == "Article text."
        mock_fetch.assert_called_once_with("https://example.com/article", "test-key")

    def test_fetch_content_returns_none_when_no_exa(self):
        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", search_brave)]
        client.exhausted = set()
        result = client.fetch_content("https://example.com/article")
        assert result is None

    def test_fetch_content_returns_none_when_exa_exhausted(self):
        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("exa", search_exa)]
        client.exhausted = {"exa"}
        result = client.fetch_content("https://example.com/article")
        assert result is None

    @patch.dict("os.environ", {"BRAVE_API_KEY": "test-brave", "SERPER_API_KEY": "test-serper", "EXA_API_KEY": "test-exa"})
    def test_from_config_default_providers(self):
        config = MagicMock()
        config.search_providers = None
        client = SearchClient.from_config(config)
        assert len(client.providers) == 3
        assert client.providers[0]["name"] == "brave"
        assert client.providers[1]["name"] == "serper"
        assert client.providers[2]["name"] == "exa"

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


class TestSearchClientCounters:
    def _make_provider(self, name, fn):
        return {"name": name, "fn": fn, "api_key": "test-key"}

    def test_counts_query_per_provider(self):
        def brave_fn(q, k, c):
            return [{"url": "https://b/a", "title": "", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn)]
        client.exhausted = set()

        client.search("q1")
        client.search("q2")
        client.search("q3")

        assert client.usage_counters["queries_by_provider"]["brave"] == 3

    def test_counts_results_per_provider(self):
        def brave_fn(q, k, c):
            return [{"url": "https://b/a", "title": "", "description": ""},
                    {"url": "https://b/b", "title": "", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn)]
        client.exhausted = set()

        client.search("q1")
        client.search("q2")

        assert client.usage_counters["results_by_provider"]["brave"] == 4

    def test_attributes_query_to_fallback_provider(self):
        """When the first provider returns None, both it and the fallback should
        be counted (one query each)."""
        def brave_fn(q, k, c):
            return None

        def serper_fn(q, k, c):
            return [{"url": "https://s/a", "title": "", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [
            self._make_provider("brave", brave_fn),
            self._make_provider("serper", serper_fn),
        ]
        client.exhausted = set()

        client.search("q")

        assert client.usage_counters["queries_by_provider"]["brave"] == 1
        assert client.usage_counters["queries_by_provider"]["serper"] == 1
        # Brave returned None — no results from it
        assert client.usage_counters["results_by_provider"].get("brave", 0) == 0
        assert client.usage_counters["results_by_provider"]["serper"] == 1

    def test_does_not_count_exhausted_providers(self):
        def brave_fn(q, k, c):
            return [{"url": "https://b/a", "title": "", "description": ""}]

        def serper_fn(q, k, c):
            return [{"url": "https://s/a", "title": "", "description": ""}]

        client = SearchClient.__new__(SearchClient)
        client.providers = [
            self._make_provider("brave", brave_fn),
            self._make_provider("serper", serper_fn),
        ]
        client.exhausted = {"brave"}

        client.search("q")

        assert "brave" not in client.usage_counters["queries_by_provider"]
        assert client.usage_counters["queries_by_provider"]["serper"] == 1

    def test_counts_zero_results_call(self):
        """A successful-but-empty result still counts as a query."""
        def brave_fn(q, k, c):
            return []

        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("brave", brave_fn)]
        client.exhausted = set()

        client.search("q")

        assert client.usage_counters["queries_by_provider"]["brave"] == 1
        assert client.usage_counters["results_by_provider"].get("brave", 0) == 0

    @patch("classivore.collection.search.fetch_exa_content")
    def test_counts_content_fetches(self, mock_fetch):
        mock_fetch.return_value = "text"
        client = SearchClient.__new__(SearchClient)
        client.providers = [self._make_provider("exa", search_exa)]
        client.exhausted = set()

        client.fetch_content("https://example.com/a")
        client.fetch_content("https://example.com/b")

        assert client.usage_counters["content_fetches_by_provider"]["exa"] == 2

    def test_counters_lazy_init_with_new(self):
        """Tests that bypass __init__ via __new__ still get counters via property."""
        client = SearchClient.__new__(SearchClient)
        # No __init__ called, no _usage_counters attribute set
        # Property should lazy-init
        assert client.usage_counters == {
            "queries_by_provider": {},
            "results_by_provider": {},
            "content_fetches_by_provider": {},
        }
