#!/usr/bin/env python3
"""Tests for web scraper."""

from unittest.mock import MagicMock, patch

import pytest

from classivore.collection.scraper import (
    USER_AGENTS,
    extract_text,
    fetch_page,
)


SAMPLE_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
<div id="cookie-banner">Accept cookies</div>
<article>
<h1>Great Article Title</h1>
<p>This is a substantial article about an interesting topic.
It has multiple sentences and paragraphs to ensure it passes
word count filters. The content is meaningful and relevant.</p>
<p>Another paragraph with more detailed information about the
subject matter being discussed in this article.</p>
</article>
</body>
</html>
"""

MINIMAL_HTML = """
<html><body><p>Just some text content here.</p></body></html>
"""


class TestExtractText:
    @patch("classivore.collection.scraper.trafilatura")
    def test_trafilatura_primary(self, mock_traf):
        mock_traf.extract.return_value = "Extracted article text."
        result = extract_text(SAMPLE_HTML)
        assert result == "Extracted article text."
        mock_traf.extract.assert_called_once()

    @patch("classivore.collection.scraper.trafilatura")
    def test_bs4_fallback(self, mock_traf):
        mock_traf.extract.return_value = None
        result = extract_text(MINIMAL_HTML)
        # BS4 fallback should extract something
        assert result is not None
        assert "text content" in result

    @patch("classivore.collection.scraper.trafilatura")
    def test_strips_cookie_banners(self, mock_traf):
        mock_traf.extract.return_value = "Article text without cookies."
        extract_text(SAMPLE_HTML)
        # The HTML passed to trafilatura should have cookie banner stripped
        call_args = mock_traf.extract.call_args
        html_passed = call_args[0][0]
        assert "Accept cookies" not in html_passed

    @patch("classivore.collection.scraper.trafilatura")
    def test_returns_none_on_total_failure(self, mock_traf):
        mock_traf.extract.return_value = None
        result = extract_text("")
        assert result is None

    @patch("classivore.collection.scraper.trafilatura")
    def test_favor_precision(self, mock_traf):
        mock_traf.extract.return_value = "Text."
        extract_text(SAMPLE_HTML)
        call_kwargs = mock_traf.extract.call_args[1]
        assert call_kwargs.get("favor_precision") is True


class TestFetchPage:
    @patch("classivore.collection.scraper.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_get.return_value = mock_resp

        html = fetch_page("https://example.com/article")
        assert html == SAMPLE_HTML

    @patch("classivore.collection.scraper.requests.get")
    def test_sets_user_agent(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_get.return_value = mock_resp

        fetch_page("https://example.com/article")
        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers.get("User-Agent") in USER_AGENTS

    @patch("classivore.collection.scraper.requests.get")
    def test_returns_none_on_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        result = fetch_page("https://example.com/blocked")
        assert result is None

    @patch("classivore.collection.scraper.requests.get")
    def test_returns_none_on_non_html(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_get.return_value = mock_resp

        result = fetch_page("https://example.com/file.pdf")
        assert result is None

    @patch("classivore.collection.scraper.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = fetch_page("https://example.com/down")
        assert result is None

    def test_multiple_user_agents(self):
        assert len(USER_AGENTS) >= 3
