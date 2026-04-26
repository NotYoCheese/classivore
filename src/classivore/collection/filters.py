#!/usr/bin/env python3
"""Content quality filters for collected web pages.

Applies two-tier URL blocklist (always-block + e-commerce-only), word count
bounds, boilerplate detection, unique word ratio, article vs listing detection,
and truncation.
"""

import hashlib
import re
from urllib.parse import urlparse

from classivore.logging_config import get_logger

logger = get_logger(__name__)

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

# --- Two-tier URL blocklist ---

# Always blocked regardless of domain
URL_BLOCKLIST_ALWAYS = [
    # E-commerce actions
    "/cart/", "/checkout/", "/buy/",
    # Navigation and browsing
    "/category/", "/categories/", "/browse/", "/filter/", "/sort",
    # Search and results
    "/search", "/results", "/query",
    # Job postings
    "/jobs/", "/careers/search/",
    # Policy and admin pages
    "/pages/ordering-policy", "/pages/return-policy", "/pages/privacy",
    "/pages/compliance", "/pages/volume-discount", "/pages/contact",
    "/admin/", "/account/", "/profile/", "/dashboard",
    # Subdomains
    "marketplace.", "store.",
    # Media galleries (low text content)
    "/photo/", "/photos/", "/pictures/", "/slides/", "/slideshow/",
    "/gallery/", "/galleries/",
    # Ad networks
    "openx.", "doubleclick.", "adnxs.",
    # Popups
    "/popup",
]

# Only blocked on e-commerce domains (editorial sites may have good content)
URL_BLOCKLIST_ECOMMERCE = [
    "/best-", "/top-", "/product-review", "/buying-guide",
    "/products/", "/shop/", "/marketplace/", "/listing", "/listings/",
    "/gift-", "/gift/",
    "/courses/search/", "/course/",
]

# Signals that a domain is e-commerce (checked against netloc)
ECOMMERCE_DOMAIN_SIGNALS = [
    "shop", "store", "buy", "deal", "price",
    "amazon", "ebay", "walmart", "etsy", "alibaba",
    "target.com", "costco", "bestbuy",
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
MIN_PATH_DEPTH = 3


def _is_ecommerce_domain(url):
    """Check if a URL's domain has e-commerce signals."""
    try:
        domain = urlparse(url).netloc.lower()
    except (AttributeError, ValueError) as e:
        logger.debug("ecommerce_domain_check_failed", url=url, error=str(e))
        return False
    return any(signal in domain for signal in ECOMMERCE_DOMAIN_SIGNALS)


def is_url_blocked(url, relaxations=None):
    """Check URL against two-tier blocklist and minimum path depth.

    Tier 1 (always block): e-commerce actions, admin, galleries, ad networks.
    Tier 2 (e-commerce domains only): /best-, /top-, /products/, etc.

    Args:
        url: URL string to check.
        relaxations: Optional dict with per-tier1 filter overrides.

    Returns:
        Rejection reason string, or None if URL is acceptable.
    """
    url_lower = url.lower()

    # Tier 1: always blocked
    for pattern in URL_BLOCKLIST_ALWAYS:
        if pattern in url_lower:
            return f"url_blocklist:{pattern.strip('/')}"

    # Tier 2: blocked only on e-commerce domains
    if _is_ecommerce_domain(url):
        for pattern in URL_BLOCKLIST_ECOMMERCE:
            if pattern in url_lower:
                return f"url_ecommerce:{pattern.strip('/')}"

    # Require minimum path depth (not just homepage)
    r = relaxations or {}
    effective_min_path_depth = r.get("min_path_depth", MIN_PATH_DEPTH)
    try:
        path = url.split("/", 3)[-1]
        if len(path) < effective_min_path_depth:
            return "url_too_shallow"
    except (IndexError, ValueError, AttributeError) as e:
        logger.debug("url_parse_failed", url=url, error=str(e))
        return "url_parse_error"

    return None


def filter_page(text, url="", word_count=None, relaxations=None):
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
        relaxations: Optional dict with per-tier1 filter overrides.

    Returns:
        Tuple of (filtered_text, rejection_reason).
        If filtered_text is not None, the page passed.
        If rejection_reason is not None, the page was rejected.
    """
    r = relaxations or {}
    effective_min_words = r.get("min_words", MIN_WORDS)
    effective_min_unique = r.get("min_unique_word_ratio", MIN_UNIQUE_WORD_RATIO)
    effective_allow_listing = r.get("allow_listing_pages", False)

    if not text or not text.strip():
        return (None, "empty_content")

    if word_count is None:
        word_count = len(text.split())

    # 1. Word count bounds
    if word_count < effective_min_words:
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
    if unique_words / len(words) < effective_min_unique:
        return (None, f"low_unique_ratio:{unique_words}/{len(words)}")

    # 4. Article vs listing indicators
    if not effective_allow_listing:
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

    # BS4's html.parser raises on malformed character refs (e.g. binary
    # garbage from a gzipped WARC interpreted as text). Skip stripping
    # rather than crashing the whole iteration.
    try:
        soup = BeautifulSoup(html, "html.parser")
    except (ValueError, AssertionError) as e:
        logger.debug("strip_cookie_banners_parse_failed", error=str(e))
        return html

    for pattern in COOKIE_BANNER_PATTERNS:
        # Remove by class
        for element in soup.find_all(class_=pattern):
            element.decompose()
        # Remove by id
        for element in soup.find_all(id=pattern):
            element.decompose()

    return str(soup)
