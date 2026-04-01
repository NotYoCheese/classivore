#!/usr/bin/env python3
"""Content quality filters for collected web pages.

Applies URL blocklist, word count bounds, boilerplate detection,
unique word ratio, article vs listing detection, and truncation.
Ported from iab_forge with improvements.
"""

import hashlib
import re

# --- Boilerplate detection (applied to first 500 chars of short pages) ---

BOILERPLATE_PATTERNS = [
    re.compile(r"^.{0,50}(cookie|consent|privacy|gdpr)", re.IGNORECASE),
    re.compile(r"^.{0,50}we('ve| have) updated our (privacy|cookie)", re.IGNORECASE),
    re.compile(r"^.{0,50}(accept all|reject all|manage preferences)", re.IGNORECASE),
    re.compile(r"^.{0,50}this (site|website) uses cookies", re.IGNORECASE),
    re.compile(r"^.{0,50}(sign up|sign in|log in|register|create.{0,10}account)", re.IGNORECASE),
    re.compile(r"^.{0,100}when you register as a free member", re.IGNORECASE),
    re.compile(r"^.{0,30}(error:|404|403|access denied|page not found)", re.IGNORECASE),
    re.compile(r"^.{0,50}(enable javascript|browser.{0,20}not supported)", re.IGNORECASE),
    re.compile(r"^.{0,50}(captcha|verify you are human|are you a robot)", re.IGNORECASE),
    re.compile(r"^.{0,50}just a moment.{0,20}(cloudflare|checking)", re.IGNORECASE),
    re.compile(r"^.{0,30}(frequently asked questions|FAQ)\n", re.IGNORECASE),
]

# --- URL blocklist (marketplace, listings, search, admin, etc.) ---

URL_BLOCKLIST = [
    # Marketplace and shopping
    "marketplace.", "store.", "/marketplace/", "/shop/", "/products/", "/listing",
    "/listings/", "/buy/", "/cart/", "/checkout/",
    # Category pages and browsing
    "/category/", "/categories/", "/browse/", "/filter/", "/sort",
    # Search and results
    "/search", "/results", "/query",
    # Job postings
    "/jobs/", "/careers/search/",
    # Course listings
    "/courses/search/", "/course/",
    # Gift guides and buying guides
    "/gift-", "/gift/", "/best-", "/top-", "/buying-guide", "/product-review",
    # Policies and store pages
    "/pages/ordering-policy", "/pages/return-policy", "/pages/privacy",
    "/pages/compliance", "/pages/volume-discount", "/pages/contact",
    # E-commerce admin
    "/admin/", "/account/", "/profile/", "/dashboard",
]

# --- Content indicators (first 500 chars) ---

ARTICLE_INDICATORS = ["article", "story", "blog", "post", "author:", "published"]
LISTING_INDICATORS = ["filter", "search results", "sort by", "price:", "compare"]

# --- Cookie banner HTML selectors for pre-extraction removal ---

COOKIE_BANNER_PATTERNS = [
    re.compile(r"cookie|gdpr|consent|privacy-banner|cookie-policy", re.IGNORECASE),
    re.compile(r"onetrust|cookiebot|cookie-notice|cookie-bar", re.IGNORECASE),
]

# --- Thresholds ---

MIN_WORDS = 150
MAX_WORDS = 50000
TRUNCATE_ABOVE = 2000
TRUNCATE_TO = 1000
MIN_UNIQUE_WORD_RATIO = 0.30
MIN_PATH_DEPTH = 10


def is_url_blocked(url):
    """Check URL against blocklist patterns and minimum path depth.

    Args:
        url: URL string to check.

    Returns:
        Rejection reason string, or None if URL is acceptable.
    """
    url_lower = url.lower()

    for pattern in URL_BLOCKLIST:
        if pattern in url_lower:
            return f"url_blocklist:{pattern.strip('/')}"

    # Require minimum path depth (not just homepage/listing page)
    try:
        path = url.split("/", 3)[-1]
        if len(path) < MIN_PATH_DEPTH:
            return "url_too_shallow"
    except (IndexError, ValueError):
        return "url_parse_error"

    return None


def filter_page(text, url="", word_count=None):
    """Apply all content filters to extracted text.

    Filters applied in order (cheapest first):
    1. Word count bounds (min/max)
    2. Boilerplate detection (first 500 chars, only if <300 words)
    3. Unique word ratio (<30%)
    4. Article vs listing indicators (first 500 chars)
    5. Truncation (>2000 words → 1000)

    Args:
        text: Extracted page text.
        url: Page URL (for logging, not filtered here — use is_url_blocked).
        word_count: Pre-computed word count, or None to compute.

    Returns:
        Tuple of (filtered_text, rejection_reason).
        If filtered_text is not None, the page passed.
        If rejection_reason is not None, the page was rejected.
    """
    if not text or not text.strip():
        return (None, "empty_content")

    if word_count is None:
        word_count = len(text.split())

    # 1. Word count bounds
    if word_count < MIN_WORDS:
        return (None, f"too_short:{word_count}")

    if word_count > MAX_WORDS:
        return (None, f"too_long:{word_count}")

    # 2. Boilerplate detection (only for short pages)
    if word_count <= 300:
        text_start = text[:500]
        if any(p.search(text_start) for p in BOILERPLATE_PATTERNS):
            return (None, "boilerplate")

    # 3. Unique word ratio
    words = text.lower().split()
    unique_words = len(set(words))
    if unique_words / len(words) < MIN_UNIQUE_WORD_RATIO:
        return (None, f"low_unique_ratio:{unique_words}/{len(words)}")

    # 4. Article vs listing indicators
    text_start = text[:500].lower()
    has_article = any(ind in text_start for ind in ARTICLE_INDICATORS)
    has_listing = any(ind in text_start for ind in LISTING_INDICATORS)
    if has_listing and not has_article:
        return (None, "listing_page")

    # 5. Truncation
    if word_count > TRUNCATE_ABOVE:
        words = text.split()[:TRUNCATE_TO]
        text = " ".join(words)

    return (text, None)


def content_hash(text):
    """Compute MD5 hash of text for deduplication.

    Args:
        text: Text string.

    Returns:
        Hex digest string.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def strip_cookie_banners(html):
    """Remove cookie/GDPR banner elements from HTML before extraction.

    Uses class/id pattern matching to find and remove banner elements.
    Should be called before trafilatura or BS4 extraction.

    Args:
        html: Raw HTML string.

    Returns:
        HTML with banner elements removed.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for pattern in COOKIE_BANNER_PATTERNS:
        # Remove by class
        for element in soup.find_all(class_=pattern):
            element.decompose()
        # Remove by id
        for element in soup.find_all(id=pattern):
            element.decompose()

    return str(soup)
