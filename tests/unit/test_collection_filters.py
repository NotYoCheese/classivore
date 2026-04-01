#!/usr/bin/env python3
"""Tests for collection content filters."""

import pytest

from classivore.collection.filters import (
    content_hash,
    filter_page,
    is_url_blocked,
    strip_cookie_banners,
)


class TestUrlBlocked:
    def test_marketplace_blocked(self):
        assert is_url_blocked("https://marketplace.example.com/page") is not None

    def test_shop_path_blocked(self):
        assert is_url_blocked("https://example.com/shop/item") is not None

    def test_search_blocked(self):
        assert is_url_blocked("https://example.com/search?q=test") is not None

    def test_jobs_blocked(self):
        assert is_url_blocked("https://example.com/jobs/listing") is not None

    def test_gift_guide_blocked(self):
        assert is_url_blocked("https://example.com/gift-guide-2026") is not None

    def test_cart_blocked(self):
        assert is_url_blocked("https://example.com/cart/checkout") is not None

    def test_article_path_allowed(self):
        assert is_url_blocked("https://example.com/news/article/great-story") is None

    def test_shallow_path_blocked(self):
        assert is_url_blocked("https://example.com/a") is not None

    def test_deep_path_allowed(self):
        assert is_url_blocked("https://example.com/2026/03/my-article-title") is None


class TestFilterPage:
    def _make_text(self, word_count):
        """Generate text with high unique word ratio."""
        return " ".join(f"word{i}" for i in range(word_count))

    def _make_article(self, word_count=200):
        """Generate article-like text that passes all filters."""
        return "This published article " + self._make_text(word_count)

    def test_passes_good_content(self):
        text = self._make_article(200)
        filtered, reason = filter_page(text)
        assert filtered is not None
        assert reason is None

    def test_rejects_empty(self):
        filtered, reason = filter_page("")
        assert filtered is None
        assert reason == "empty_content"

    def test_rejects_too_short(self):
        text = "Short text. " * 10
        filtered, reason = filter_page(text, word_count=20)
        assert filtered is None
        assert "too_short" in reason

    def test_rejects_too_long(self):
        text = self._make_text(60000)
        filtered, reason = filter_page(text, word_count=60000)
        assert filtered is None
        assert "too_long" in reason

    def test_rejects_cookie_boilerplate(self):
        text = "This website uses cookies to improve your experience. " * 10
        filtered, reason = filter_page(text, word_count=200)
        assert filtered is None
        assert reason == "boilerplate"

    def test_boilerplate_skipped_for_long_pages(self):
        # Pages over 300 words skip boilerplate check
        text = "This website uses cookies. " + self._make_article(400)
        filtered, reason = filter_page(text)
        assert filtered is not None

    def test_rejects_low_unique_ratio(self):
        # Repetitive listing content
        text = ("price filter sort price filter sort " * 50).strip()
        filtered, reason = filter_page(text)
        assert filtered is None
        assert "low_unique_ratio" in reason

    def test_rejects_listing_page(self):
        text = "filter search results sort by price compare " + self._make_text(200)
        filtered, reason = filter_page(text)
        assert filtered is None
        assert reason == "listing_page"

    def test_allows_article_with_listing_words(self):
        # Article that mentions listing terms but also has article indicators
        text = "This published article discusses how to compare and filter " + self._make_text(200)
        filtered, reason = filter_page(text)
        assert filtered is not None

    def test_truncates_long_content(self):
        words = [f"word{i}" for i in range(3000)]
        text = " ".join(words)
        filtered, reason = filter_page(text)
        assert filtered is not None
        assert len(filtered.split()) == 1000

    def test_no_truncation_under_threshold(self):
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        filtered, reason = filter_page(text)
        assert filtered is not None
        assert len(filtered.split()) == 500

    def test_rejects_captcha(self):
        text = "Verify you are human. Please complete the captcha. " * 10
        filtered, reason = filter_page(text, word_count=200)
        assert filtered is None
        assert reason == "boilerplate"

    def test_rejects_cloudflare(self):
        text = "Just a moment... Checking your browser before accessing cloudflare. " * 10
        filtered, reason = filter_page(text, word_count=200)
        assert filtered is None
        assert reason == "boilerplate"


class TestContentHash:
    def test_consistent(self):
        assert content_hash("hello world") == content_hash("hello world")

    def test_different_for_different_text(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


class TestStripCookieBanners:
    def test_removes_cookie_div(self):
        html = '<html><body><div class="cookie-consent">Accept cookies</div><p>Real content</p></body></html>'
        result = strip_cookie_banners(html)
        assert "Accept cookies" not in result
        assert "Real content" in result

    def test_removes_gdpr_banner(self):
        html = '<html><body><div id="gdpr-banner">GDPR notice</div><p>Article text</p></body></html>'
        result = strip_cookie_banners(html)
        assert "GDPR notice" not in result
        assert "Article text" in result

    def test_removes_onetrust(self):
        html = '<html><body><div id="onetrust-banner">Cookie settings</div><p>Content</p></body></html>'
        result = strip_cookie_banners(html)
        assert "Cookie settings" not in result

    def test_preserves_normal_content(self):
        html = '<html><body><p>Just a normal article</p></body></html>'
        result = strip_cookie_banners(html)
        assert "Just a normal article" in result
