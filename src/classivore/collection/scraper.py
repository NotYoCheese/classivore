#!/usr/bin/env python3
"""Live web scraping with trafilatura and BS4 fallback.

Downloads pages via HTTP, strips cookie banners from HTML before extraction,
then extracts article text using trafilatura (precision mode). Falls back to
BeautifulSoup paragraph extraction when trafilatura returns nothing.

HTTP fetches go through `curl_cffi` impersonating Chrome 131 — TLS handshake,
HTTP/2 settings, header order, User-Agent, and the full Sec-Ch-Ua / Sec-Fetch
family are all set by the impersonation profile to match a real Chrome 131 on
macOS. We pass no headers ourselves: any override risks introducing a TLS-vs-UA
or header-order mismatch that WAFs cross-check as a bot tell.

Rate limiting is handled by the orchestrator, not here.
"""

from curl_cffi import requests
import trafilatura
from bs4 import BeautifulSoup

from classivore.collection.filters import strip_cookie_banners
from classivore.logging_config import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT = 20

IMPERSONATE = "chrome131"


def _fetch_response(url, session=None):
    """Issue the GET request with a real browser TLS fingerprint.

    When `session` is None, uses module-level curl_cffi.requests to impersonate
    Chrome 131 — TLS handshake, HTTP/2 settings, header order, and the full
    default Chrome header set all match a real browser, which gets us past
    Akamai/Cloudflare WAFs that fingerprint plain `requests`.

    When `session` is provided, the caller owns the transport. We do not pass
    `impersonate=` because not every Session implementation accepts it; the
    caller is expected to have configured fingerprinting (and proxies, retries,
    timeouts, adapters, pooling) on the session itself.

    Returns the raw `requests.Response` so callers (the bench, primarily)
    can inspect status codes, headers, and body length on non-200s. The
    public `fetch_page` wraps this and returns just the HTML string.
    """
    if session is not None:
        return session.get(url, timeout=REQUEST_TIMEOUT)
    return requests.get(url, timeout=REQUEST_TIMEOUT, impersonate=IMPERSONATE)


def fetch_page(url, session=None):
    """Download a page and return raw HTML.

    Args:
        url: URL to fetch.
        session: Optional pre-configured HTTP session (e.g. `requests.Session`
            or `curl_cffi.requests.Session`). When provided, the caller owns
            all HTTP behavior — proxies, retries, timeouts, connection pooling,
            custom adapters, and TLS fingerprinting. This library does not
            apply its own impersonation profile to caller-provided sessions.
            When None (default), uses module-level `curl_cffi.requests` with
            Chrome 131 impersonation.

    Returns:
        HTML string, or None on failure or non-HTML response.
    """
    try:
        resp = _fetch_response(url, session=session)
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
    except Exception as e:
        logger.debug("bs4_parse_failed", error=str(e))
        return None

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 20:
            paragraphs.append(text)

    if not paragraphs:
        return None

    return "\n\n".join(paragraphs)
