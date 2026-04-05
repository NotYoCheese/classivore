#!/usr/bin/env python3
"""Common Crawl CDX lookup and WARC byte-range download.

Provides free content retrieval by looking up URLs in Common Crawl's CDX index
and downloading just the relevant WARC record via HTTP byte-range requests.
Falls back gracefully when URLs aren't indexed.

Handles the CDX API's endpoint format inconsistency (with/without -index suffix)
and applies exponential backoff on rate limiting (429/503).
"""

import io
import json
import time

import requests

from classivore.logging_config import get_logger

logger = get_logger(__name__)

CDX_BASE_URL = "https://index.commoncrawl.org"
WARC_BASE_URL = "https://data.commoncrawl.org"

BACKOFF_DELAYS = [2, 4, 8]
REQUEST_TIMEOUT = 30
PROBE_TIMEOUT = 5


def probe_cdx(crawl_id):
    """Quick health check on CDX index. Returns True if responsive.

    Sends a minimal query with a short timeout. Used to skip CDX
    for an entire category if the service is down or degraded.
    """
    endpoint = f"{CDX_BASE_URL}/{crawl_id}-index"
    try:
        resp = requests.get(
            endpoint,
            params={"url": "example.com", "output": "json", "limit": 1},
            timeout=PROBE_TIMEOUT,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def parse_cdx_response(text):
    """Parse NDJSON CDX response into record dicts.

    Skips HTML info pages, invalid JSON lines, and non-200 status records.
    Converts offset and length to integers.

    Returns:
        List of record dicts with url, timestamp, filename, offset, length.
    """
    if not text or text.strip().startswith("<"):
        return []

    records = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("status", "200") != "200":
            continue

        record["offset"] = int(record["offset"])
        record["length"] = int(record["length"])
        records.append(record)

    return records


def lookup_cdx(url, crawl_id, limit=5):
    """Look up a URL in Common Crawl's CDX index.

    Tries the newer endpoint format first ({crawl_id}-index), then falls back
    to the older format ({crawl_id}) on 404 or HTML info page responses.
    Applies exponential backoff on 429/503.

    Args:
        url: URL to look up.
        crawl_id: Common Crawl crawl ID (e.g. "CC-MAIN-2026-08").
        limit: Max number of records to return.

    Returns:
        List of CDX record dicts, or empty list if not found.
    """
    endpoints = [
        f"{CDX_BASE_URL}/{crawl_id}-index",
        f"{CDX_BASE_URL}/{crawl_id}",
    ]

    params = {
        "url": url,
        "output": "json",
        "limit": limit,
        "filter": "status:200",
    }

    for endpoint in endpoints:
        records = _try_cdx_endpoint(endpoint, params)
        if records is not None:
            return records

    return []


def _try_cdx_endpoint(endpoint, params):
    """Try a single CDX endpoint with backoff. Returns records or None to try next."""
    for attempt, delay in enumerate(BACKOFF_DELAYS + [None]):
        try:
            resp = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            logger.warning("cdx_request_failed", error=str(e))
            return None

        if resp.status_code == 404:
            return None

        if resp.status_code in (429, 503):
            if delay is not None:
                logger.info("cdx_rate_limited", status_code=resp.status_code, backoff_seconds=delay)
                time.sleep(delay)
                continue
            logger.warning("cdx_rate_limited_exhausted")
            return None

        if resp.status_code != 200:
            logger.warning("cdx_unexpected_status", status_code=resp.status_code)
            return None

        records = parse_cdx_response(resp.text)
        if records:
            return records

        # Got 200 but no valid records (likely HTML info page) — try next endpoint
        return None

    return None


def fetch_warc_record(record):
    """Download a single WARC record via HTTP byte-range request.

    Args:
        record: CDX record dict with filename, offset, length.

    Returns:
        Raw HTML string, or None on failure.
    """
    url = f"{WARC_BASE_URL}/{record['filename']}"
    offset = record["offset"]
    end = offset + record["length"] - 1

    try:
        resp = requests.get(
            url,
            headers={"Range": f"bytes={offset}-{end}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.warning("warc_download_failed", error=str(e))
        return None

    if resp.status_code not in (200, 206):
        logger.warning("warc_download_bad_status", status_code=resp.status_code, url=url)
        return None

    return _extract_text_from_warc(resp.content)


def _extract_text_from_warc(data):
    """Extract HTML content from a WARC record.

    Args:
        data: Raw bytes of a single WARC record (gzipped).

    Returns:
        HTML string, or None if extraction fails.
    """
    try:
        from warcio.archiveiterator import ArchiveIterator
    except ImportError:
        logger.error("warcio_not_installed")
        return None

    try:
        stream = io.BytesIO(data)
        for warc_record in ArchiveIterator(stream):
            if warc_record.rec_type in ("response", "resource"):
                content = warc_record.content_stream().read()
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return content.decode("latin-1")
    except Exception as e:
        logger.warning("warc_extraction_failed", error=str(e))

    return None
