#!/usr/bin/env python3
"""Tests for tools/scraper_bench.py."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_BENCH_PATH = Path(__file__).resolve().parents[2] / "tools" / "scraper_bench.py"


def _load_bench_module():
    """Load tools/scraper_bench.py as a module without an __init__.py."""
    spec = importlib.util.spec_from_file_location("scraper_bench", _BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scraper_bench"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench_module()


def _mock_response(status=200, body="<html><body>ok</body></html>",
                   content_type="text/html; charset=utf-8"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.content = body.encode("utf-8") if isinstance(body, str) else body
    resp.headers = {"content-type": content_type}
    return resp


class TestHostOf:
    def test_strips_scheme_and_path(self):
        assert bench._host_of("https://Example.COM/path?q=1") == "example.com"

    def test_handles_garbage(self):
        assert bench._host_of("not-a-url") == ""


class TestBlockMarkers:
    def test_tiny_body_is_a_marker(self):
        markers = bench._detect_block_markers("ok", body_bytes=500)
        assert "tiny_body" in markers

    def test_just_a_moment(self):
        markers = bench._detect_block_markers(
            "<title>Just a moment...</title>" * 10, body_bytes=10_000,
        )
        assert "just_a_moment" in markers

    def test_no_markers_on_clean_body(self):
        markers = bench._detect_block_markers("<p>hello world</p>", body_bytes=20_000)
        assert markers == []

    def test_large_body_with_markers_is_not_blocked(self):
        markers = ["captcha_present"]
        assert bench._is_blocked(markers, body_bytes=80_000) is False

    def test_small_body_with_markers_is_blocked(self):
        markers = ["just_a_moment"]
        assert bench._is_blocked(markers, body_bytes=1_500) is True

    def test_tiny_body_alone_is_blocked(self):
        assert bench._is_blocked(["tiny_body"], body_bytes=900) is True

    def test_no_markers_means_not_blocked(self):
        assert bench._is_blocked([], body_bytes=1_000) is False


class TestProcessUrlSync:
    def test_ok_path(self, monkeypatch):
        article_html = "<html><body>" + ("<p>plenty of content here. " * 500) + "</p></body></html>"
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(body=article_html),
        )
        monkeypatch.setattr(
            bench.scraper, "extract_text",
            lambda html: "x" * 500,
        )
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "ok"
        assert rec["http_status"] == 200
        assert rec["extracted_chars"] == 500
        assert rec["would_have_used_exa"] is False
        assert rec["block_markers"] == []
        assert rec["fetch_ms"] is not None
        assert rec["extract_ms"] is not None

    def test_empty_extraction(self, monkeypatch):
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(body="<html><body><p>x</p></body></html>" * 200),
        )
        monkeypatch.setattr(bench.scraper, "extract_text", lambda html: "tiny")
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "empty_extraction"
        assert rec["would_have_used_exa"] is True

    def test_http_error_403(self, monkeypatch):
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(status=403, body="forbidden"),
        )
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "http_error"
        assert rec["http_status"] == 403
        assert rec["fetch_failed_reason"] == "http_403"
        assert rec["would_have_used_exa"] is True

    def test_429_marks_host_aborted(self, monkeypatch):
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(status=429, body="slow down"),
        )
        host_state = {}
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state=host_state,
        )
        assert rec["outcome"] == "http_error"
        assert host_state.get("example.com") == "429_aborted"

    def test_429_aborted_short_circuits_subsequent_urls(self, monkeypatch):
        # First URL would normally succeed, but host is already 429-aborted.
        called = {"n": 0}

        def fetch(url):
            called["n"] += 1
            return _mock_response()

        monkeypatch.setattr(bench.scraper, "_fetch_response", fetch)
        host_state = {"example.com": "429_aborted"}
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state=host_state,
        )
        assert called["n"] == 0
        assert rec["outcome"] == "http_error"
        assert rec["http_status"] == 429
        assert rec["fetch_failed_reason"] == "upstream_429_aborted_run"
        assert rec["would_have_used_exa"] is True

    def test_connection_exception(self, monkeypatch):
        class ConnError(Exception):
            pass
        ConnError.__name__ = "ConnectionError"

        def fail(url):
            raise ConnError("dns fail")

        monkeypatch.setattr(bench.scraper, "_fetch_response", fail)
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "connection_error"
        assert rec["fetch_failed_reason"] == "connection_error"
        assert "dns fail" in rec["error"]

    def test_timeout_exception(self, monkeypatch):
        class TimeoutErr(Exception):
            pass
        TimeoutErr.__name__ = "ReadTimeout"

        def fail(url):
            raise TimeoutErr("read timed out")

        monkeypatch.setattr(bench.scraper, "_fetch_response", fail)
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "timeout"
        assert rec["fetch_failed_reason"] == "timeout"

    def test_block_markers_in_small_body(self, monkeypatch):
        body = "<title>Just a moment...</title><p>checking your browser</p>"
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(body=body),
        )
        # extract_text shouldn't even be called once outcome=blocked,
        # but make it safe anyway.
        monkeypatch.setattr(bench.scraper, "extract_text", lambda html: None)
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "blocked"
        assert "just_a_moment" in rec["block_markers"]
        assert rec["would_have_used_exa"] is True
        assert rec["extracted_chars"] is None  # extract was skipped

    def test_non_html_content_type(self, monkeypatch):
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(content_type="application/pdf"),
        )
        rec = bench._process_url_sync(
            "https://example.com/a.pdf", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "http_error"
        assert rec["fetch_failed_reason"] == "non_html_content_type"

    def test_domain_blocked_skips_fetch(self, monkeypatch):
        called = {"n": 0}

        def fetch(url):
            called["n"] += 1
            return _mock_response()

        monkeypatch.setattr(bench.scraper, "_fetch_response", fetch)

        domains = MagicMock()
        domains.is_blocked.return_value = True
        rec = bench._process_url_sync(
            "https://blocked.example/a", "lbl", domains=domains, host_state={},
        )
        assert called["n"] == 0
        assert rec["outcome"] == "domain_blocked"
        assert rec["fetch_failed_reason"] == "domain_quality_blocked"
        # would_have_used_exa is False — production wouldn't try Exa for
        # a domain-quality skip; it would just move on.
        assert rec["would_have_used_exa"] is False

    def test_extract_exception_is_caught(self, monkeypatch):
        article = "<html><body>" + ("<p>x</p>" * 1000) + "</body></html>"
        monkeypatch.setattr(
            bench.scraper, "_fetch_response",
            lambda url: _mock_response(body=article),
        )

        def boom(html):
            raise RuntimeError("trafilatura died")

        monkeypatch.setattr(bench.scraper, "extract_text", boom)
        rec = bench._process_url_sync(
            "https://example.com/a", "lbl", domains=None, host_state={},
        )
        assert rec["outcome"] == "exception"
        assert "trafilatura died" in rec["error"]


class TestReadUrls:
    def test_strips_comments_and_blanks(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text(
            "# header comment\n\nhttps://a.test/\n   \nhttps://b.test/\n"
            "# trailing\n",
            encoding="utf-8",
        )
        assert bench._read_urls(f, limit=0) == [
            "https://a.test/", "https://b.test/",
        ]

    def test_limit(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("https://a.test/\nhttps://b.test/\nhttps://c.test/\n")
        assert bench._read_urls(f, limit=2) == [
            "https://a.test/", "https://b.test/",
        ]


class TestRunAsync:
    def test_writes_jsonl_records_in_expected_shape(self, monkeypatch, tmp_path):
        urls = [
            "https://a.test/x",
            "https://a.test/y",
            "https://b.test/z",
        ]

        def fetch(url):
            return _mock_response(body="<html><body>" + "p" * 5000 + "</body></html>")

        monkeypatch.setattr(bench.scraper, "_fetch_response", fetch)
        monkeypatch.setattr(bench.scraper, "extract_text", lambda html: "x" * 500)

        out_path = tmp_path / "out.jsonl"
        counts = asyncio.run(
            bench._run_async(
                urls=urls, label="t", domains=None, output_path=out_path,
                concurrency=4, per_domain_max=2, progress_every=0,
            )
        )

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)
            for required in (
                "ts", "label", "url", "host", "outcome", "http_status",
                "bytes_received", "extracted_chars", "fetch_ms", "extract_ms",
                "total_ms", "block_markers", "fetch_failed_reason",
                "would_have_used_exa", "error",
            ):
                assert required in rec, required
            assert rec["label"] == "t"
            assert rec["outcome"] == "ok"

        assert counts["ok"] == 3

    def test_429_on_one_host_aborts_remaining_urls_for_that_host(
        self, monkeypatch, tmp_path,
    ):
        # Hammer host A with 429s after the first response. Host B
        # should be unaffected.
        a_calls = {"n": 0}

        def fetch(url):
            if "a.test" in url:
                a_calls["n"] += 1
                return _mock_response(status=429, body="slow")
            return _mock_response(body="<html>" + "p" * 5000 + "</html>")

        monkeypatch.setattr(bench.scraper, "_fetch_response", fetch)
        monkeypatch.setattr(bench.scraper, "extract_text", lambda html: "x" * 500)

        # 5 URLs on host A, 2 on host B. With per_domain_max=1 the A
        # fetches are serialized; the first one trips the 429 and the
        # remaining four short-circuit without calling fetch again.
        urls = [
            "https://a.test/1", "https://a.test/2", "https://a.test/3",
            "https://a.test/4", "https://a.test/5",
            "https://b.test/1", "https://b.test/2",
        ]
        out_path = tmp_path / "out.jsonl"
        asyncio.run(
            bench._run_async(
                urls=urls, label="t", domains=None, output_path=out_path,
                concurrency=1, per_domain_max=1, progress_every=0,
            )
        )

        records = [json.loads(l) for l in out_path.read_text().strip().splitlines()]
        a_records = [r for r in records if r["host"] == "a.test"]
        b_records = [r for r in records if r["host"] == "b.test"]
        assert len(a_records) == 5
        assert len(b_records) == 2
        # All host B URLs succeeded.
        assert all(r["outcome"] == "ok" for r in b_records)
        # Exactly one fetch hit host A (the rest short-circuited).
        assert a_calls["n"] == 1
        aborted = [
            r for r in a_records
            if r["fetch_failed_reason"] == "upstream_429_aborted_run"
        ]
        assert len(aborted) == 4
