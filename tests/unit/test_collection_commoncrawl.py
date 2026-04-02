#!/usr/bin/env python3
"""Tests for Common Crawl CDX lookup and WARC download."""

import io
from unittest.mock import MagicMock, patch

import pytest

from classivore.collection.commoncrawl import (
    CDX_BASE_URL,
    fetch_warc_record,
    lookup_cdx,
    parse_cdx_response,
)


# --- CDX response parsing ---

SAMPLE_CDX_LINES = [
    '{"url": "https://example.com/article", "timestamp": "20260201120000", '
    '"filename": "crawl-data/CC-MAIN-2026-08/segments/123/warc/CC-MAIN-456.warc.gz", '
    '"offset": "12345", "length": "6789", "status": "200", "mime": "text/html"}',
    '{"url": "https://example.com/article2", "timestamp": "20260201130000", '
    '"filename": "crawl-data/CC-MAIN-2026-08/segments/789/warc/CC-MAIN-012.warc.gz", '
    '"offset": "54321", "length": "9876", "status": "200", "mime": "text/html"}',
]


class TestParseCdxResponse:
    def test_parses_ndjson(self):
        text = "\n".join(SAMPLE_CDX_LINES)
        records = parse_cdx_response(text)
        assert len(records) == 2
        assert records[0]["url"] == "https://example.com/article"
        assert records[0]["offset"] == 12345
        assert records[0]["length"] == 6789

    def test_skips_invalid_lines(self):
        text = "not json\n" + SAMPLE_CDX_LINES[0]
        records = parse_cdx_response(text)
        assert len(records) == 1

    def test_skips_html_info_page(self):
        text = "<html>CDX Server Info</html>"
        records = parse_cdx_response(text)
        assert len(records) == 0

    def test_empty_response(self):
        records = parse_cdx_response("")
        assert len(records) == 0

    def test_filters_non_200(self):
        text = SAMPLE_CDX_LINES[0].replace('"200"', '"301"')
        records = parse_cdx_response(text)
        assert len(records) == 0


# --- CDX lookup ---

class TestLookupCdx:
    @patch("classivore.collection.commoncrawl.requests.get")
    def test_successful_lookup(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_CDX_LINES[0]
        mock_get.return_value = mock_resp

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert len(records) == 1
        assert records[0]["url"] == "https://example.com/article"

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_endpoint_fallback(self, mock_get):
        """Falls back to alternate endpoint format on 404."""
        not_found = MagicMock()
        not_found.status_code = 404

        success = MagicMock()
        success.status_code = 200
        success.text = SAMPLE_CDX_LINES[0]

        mock_get.side_effect = [not_found, success]

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert len(records) == 1
        # Verify both endpoint formats were tried
        calls = mock_get.call_args_list
        assert len(calls) == 2
        assert "-index" in calls[0].args[0]
        assert "-index" not in calls[1].args[0]

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_html_info_page_triggers_fallback(self, mock_get):
        """HTML info page (not JSON) triggers alternate endpoint."""
        html_resp = MagicMock()
        html_resp.status_code = 200
        html_resp.text = "<html>CDX Server Info</html>"

        json_resp = MagicMock()
        json_resp.status_code = 200
        json_resp.text = SAMPLE_CDX_LINES[0]

        mock_get.side_effect = [html_resp, json_resp]

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert len(records) == 1

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_both_endpoints_fail(self, mock_get):
        not_found = MagicMock()
        not_found.status_code = 404
        mock_get.return_value = not_found

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert records == []

    @patch("classivore.collection.commoncrawl.time.sleep")
    @patch("classivore.collection.commoncrawl.requests.get")
    def test_backoff_on_429(self, mock_get, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429

        success = MagicMock()
        success.status_code = 200
        success.text = SAMPLE_CDX_LINES[0]

        mock_get.side_effect = [rate_limited, success]

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert len(records) == 1
        mock_sleep.assert_called()

    @patch("classivore.collection.commoncrawl.time.sleep")
    @patch("classivore.collection.commoncrawl.requests.get")
    def test_backoff_on_503(self, mock_get, mock_sleep):
        unavailable = MagicMock()
        unavailable.status_code = 503

        success = MagicMock()
        success.status_code = 200
        success.text = SAMPLE_CDX_LINES[0]

        mock_get.side_effect = [unavailable, success]

        records = lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        assert len(records) == 1

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_uses_wildcard_query(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_CDX_LINES[0]
        mock_get.return_value = mock_resp

        lookup_cdx("https://example.com/article", crawl_id="CC-MAIN-2026-08")
        params = mock_get.call_args.kwargs.get("params", mock_get.call_args[1].get("params", {}))
        assert params["url"] == "https://example.com/article"
        assert params["output"] == "json"


# --- WARC download ---

class TestFetchWarcRecord:
    @patch("classivore.collection.commoncrawl.requests.get")
    def test_constructs_range_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = b""
        mock_get.return_value = mock_resp

        record = {
            "filename": "crawl-data/segment/warc.gz",
            "offset": 1000,
            "length": 500,
        }

        with patch("classivore.collection.commoncrawl._extract_text_from_warc") as mock_extract:
            mock_extract.return_value = None
            fetch_warc_record(record)

        headers = mock_get.call_args.kwargs.get("headers", mock_get.call_args[1].get("headers", {}))
        assert headers["Range"] == "bytes=1000-1499"

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_returns_extracted_text(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = b"fake warc data"
        mock_get.return_value = mock_resp

        record = {
            "filename": "crawl-data/segment/warc.gz",
            "offset": 0,
            "length": 100,
        }

        with patch("classivore.collection.commoncrawl._extract_text_from_warc") as mock_extract:
            mock_extract.return_value = "<html><body>Article text</body></html>"
            result = fetch_warc_record(record)

        assert result == "<html><body>Article text</body></html>"

    @patch("classivore.collection.commoncrawl.requests.get")
    def test_returns_none_on_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        record = {
            "filename": "crawl-data/segment/warc.gz",
            "offset": 0,
            "length": 100,
        }

        result = fetch_warc_record(record)
        assert result is None
