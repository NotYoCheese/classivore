#!/usr/bin/env python3
"""Brave Search API client for URL discovery.

Searches for content matching taxonomy categories and returns discovered URLs.
Enforces 1 request per second rate limit per Brave API terms.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 15


def parse_search_results(response_data):
    """Extract search results from Brave API response.

    Args:
        response_data: Parsed JSON response from Brave Search API.

    Returns:
        List of dicts with url, title, description.
    """
    web = response_data.get("web", {})
    raw_results = web.get("results", [])

    return [
        {
            "url": r["url"],
            "title": r.get("title", ""),
            "description": r.get("description", ""),
        }
        for r in raw_results
        if "url" in r
    ]


def search_brave(query, api_key, count=10):
    """Execute a Brave Search query.

    Sleeps 1 second after each request to respect rate limits.

    Args:
        query: Search query string.
        api_key: Brave Search API subscription token.
        count: Number of results to request (max 20).

    Returns:
        List of result dicts with url, title, description. Empty list on error.
    """
    try:
        resp = requests.get(
            BRAVE_API_URL,
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params={
                "q": query,
                "count": count,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        logger.warning("Brave Search request failed: %s", e)
        time.sleep(1)
        return []

    # Always rate limit, even on errors
    time.sleep(1)

    if resp.status_code != 200:
        logger.warning("Brave Search status %d for query: %s", resp.status_code, query)
        return []

    try:
        return parse_search_results(resp.json())
    except Exception as e:
        logger.warning("Failed to parse Brave Search response: %s", e)
        return []
