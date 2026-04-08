#!/usr/bin/env python3
"""Live web scraping with trafilatura and BS4 fallback.

Downloads pages via HTTP, strips cookie banners from HTML before extraction,
then extracts article text using trafilatura (precision mode). Falls back to
BeautifulSoup paragraph extraction when trafilatura returns nothing.

Rate limiting is handled by the orchestrator, not here.
"""

import random

import requests
import trafilatura
from bs4 import BeautifulSoup

from classivore.collection.filters import strip_cookie_banners
from classivore.logging_config import get_logger

logger = get_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

# Browser-like headers to avoid WAF/CDN bot detection
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

REQUEST_TIMEOUT = 20


def fetch_page(url):
    """Download a page and return raw HTML.

    Args:
        url: URL to fetch.

    Returns:
        HTML string, or None on failure or non-HTML response.
    """
    headers = {**BROWSER_HEADERS, "User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        logger.warning("fetch_failed", url=url, error=str(e))
        return None

    if resp.status_code != 200:
        logger.info("non_200_status", status_code=resp.status_code, url=url)
        return None

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type.lower():
        logger.info("non_html_content", url=url, content_type=content_type)
        return None

    return resp.text


def extract_text(html):
    """Extract article text from HTML.

    Strips cookie banners first, then uses trafilatura with favor_precision.
    Falls back to BeautifulSoup paragraph extraction.

    Args:
        html: Raw HTML string.

    Returns:
        Extracted text string, or None if extraction fails.
    """
    if not html or not html.strip():
        return None

    # Strip cookie/GDPR banners before extraction
    clean_html = strip_cookie_banners(html)

    # Primary: trafilatura
    text = trafilatura.extract(clean_html, favor_precision=True)
    if text:
        return text

    # Fallback: BeautifulSoup paragraph extraction
    return _bs4_extract(clean_html)


def _bs4_extract(html):
    """Fallback text extraction using BeautifulSoup.

    Concatenates text from <p> tags, filtering out short fragments.

    Returns:
        Extracted text string, or None if insufficient content.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 20:
            paragraphs.append(text)

    if not paragraphs:
        return None

    return "\n\n".join(paragraphs)
