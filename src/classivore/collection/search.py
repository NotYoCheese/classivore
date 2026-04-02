#!/usr/bin/env python3
"""Search provider abstraction with automatic fallback.

Supports multiple search providers (Brave, Serper) with priority-based fallback.
When a provider's quota is exhausted, seamlessly switches to the next available
provider. Each provider handles its own rate limiting and retry logic.

Return value convention for all providers:
- None: transient failure (DNS, timeout, connection) — all retries exhausted
- []: search succeeded but returned no results
- [...]: search succeeded with results
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# --- Shared constants ---

MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]
TRANSIENT_EXCEPTIONS = (requests.ConnectionError, requests.Timeout, OSError)

# --- Brave Search ---

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_REQUEST_INTERVAL = 1.1
BRAVE_MONTHLY_QUOTA_WARNING = 500


def search_brave(query, api_key, count=10):
    """Execute a Brave Search query with retry on transient errors and 429.

    Args:
        query: Search query string.
        api_key: Brave Search API subscription token.
        count: Number of results to request (max 20).

    Returns:
        List of result dicts, empty list on successful-but-empty, or None on
        transient failure after retries exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BRAVE_API_URL,
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                params={"q": query, "count": count},
                timeout=15,
            )
        except TRANSIENT_EXCEPTIONS as e:
            delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
            logger.info("Brave transient error on attempt %d, backing off %ds: %s", attempt + 1, delay, e)
            time.sleep(delay)
            continue

        if resp.status_code == 429:
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset and attempt == 0:
                try:
                    delay = min(int(reset), 10)
                except ValueError:
                    delay = BACKOFF_DELAYS[attempt]
            else:
                delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
            logger.info("Brave 429 on attempt %d, backing off %ds", attempt + 1, delay)
            time.sleep(delay)
            continue

        time.sleep(BRAVE_REQUEST_INTERVAL)
        _check_brave_quota(resp)

        if resp.status_code != 200:
            logger.warning("Brave Search status %d for query: %s", resp.status_code, query)
            return []

        try:
            return _parse_brave_results(resp.json())
        except Exception as e:
            logger.warning("Failed to parse Brave response: %s", e)
            return []

    logger.warning("Brave Search exhausted retries for query: %s", query)
    time.sleep(BRAVE_REQUEST_INTERVAL)
    return None


def _parse_brave_results(data):
    """Parse Brave Search API response."""
    web = data.get("web", {})
    return [
        {"url": r["url"], "title": r.get("title", ""), "description": r.get("description", "")}
        for r in web.get("results", [])
        if "url" in r
    ]


def _check_brave_quota(resp):
    """Log warning if Brave monthly quota is running low."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining:
        parts = [p.strip() for p in remaining.split(",")]
        if len(parts) >= 2:
            try:
                monthly = int(parts[1])
                if monthly < BRAVE_MONTHLY_QUOTA_WARNING:
                    logger.warning("Brave monthly quota low: %d remaining", monthly)
            except ValueError:
                pass


# --- Serper (Google results) ---

SERPER_API_URL = "https://google.serper.dev/search"
SERPER_REQUEST_INTERVAL = 0.5


def search_serper(query, api_key, count=10):
    """Execute a Serper search query (Google results).

    Args:
        query: Search query string.
        api_key: Serper API key.
        count: Number of results to request.

    Returns:
        List of result dicts, empty list on successful-but-empty, or None on
        transient failure after retries exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                SERPER_API_URL,
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": count},
                timeout=15,
            )
        except TRANSIENT_EXCEPTIONS as e:
            delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
            logger.info("Serper transient error on attempt %d, backing off %ds: %s", attempt + 1, delay, e)
            time.sleep(delay)
            continue

        if resp.status_code == 429:
            delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
            logger.info("Serper 429 on attempt %d, backing off %ds", attempt + 1, delay)
            time.sleep(delay)
            continue

        time.sleep(SERPER_REQUEST_INTERVAL)

        if resp.status_code != 200:
            logger.warning("Serper status %d for query: %s", resp.status_code, query)
            return []

        try:
            return _parse_serper_results(resp.json())
        except Exception as e:
            logger.warning("Failed to parse Serper response: %s", e)
            return []

    logger.warning("Serper exhausted retries for query: %s", query)
    time.sleep(SERPER_REQUEST_INTERVAL)
    return None


def _parse_serper_results(data):
    """Parse Serper API response (Google organic results)."""
    return [
        {"url": r["link"], "title": r.get("title", ""), "description": r.get("snippet", "")}
        for r in data.get("organic", [])
        if "link" in r
    ]


# --- Provider registry ---

PROVIDERS = {
    "brave": search_brave,
    "serper": search_serper,
}


# --- SearchClient with fallback ---

class SearchClient:
    """Multi-provider search client with automatic fallback.

    Tries providers in configured priority order. When a provider's quota
    is exhausted (returns None on transient failure repeatedly), marks it
    as exhausted and falls through to the next provider.
    """

    def __init__(self, provider_configs):
        """Initialize with provider configurations.

        Args:
            provider_configs: List of dicts with 'name' and 'api_key_env' keys.
                Example: [{"name": "brave", "api_key_env": "BRAVE_API_KEY"}]
        """
        self.providers = []
        for conf in provider_configs:
            name = conf["name"]
            fn = PROVIDERS.get(name)
            if not fn:
                logger.warning("Unknown search provider: %s", name)
                continue
            api_key = os.getenv(conf.get("api_key_env", ""), "")
            if not api_key:
                logger.info("No API key for %s (env: %s), skipping", name, conf.get("api_key_env"))
                continue
            self.providers.append({"name": name, "fn": fn, "api_key": api_key})

        self.exhausted = set()

    def search(self, query, count=10):
        """Search using providers in priority order.

        Returns:
            List of result dicts, empty list, or None if all providers failed.
        """
        for provider in self.providers:
            if provider["name"] in self.exhausted:
                continue

            results = provider["fn"](query, provider["api_key"], count)

            if results is None:
                logger.warning("Provider %s returned None (transient failure), marking exhausted", provider["name"])
                self.exhausted.add(provider["name"])
                continue

            return results

        # All providers exhausted or failed
        return None

    def reset_exhausted(self):
        """Reset exhausted providers (e.g. at start of a new run)."""
        self.exhausted.clear()

    @property
    def active_provider_count(self):
        """Number of providers not marked as exhausted."""
        return len([p for p in self.providers if p["name"] not in self.exhausted])

    @classmethod
    def from_config(cls, config):
        """Create SearchClient from TaxonomyConfig.

        Falls back to BRAVE_API_KEY env var if no providers configured.
        """
        provider_configs = getattr(config, "search_providers", None)
        if not provider_configs:
            # Default: try brave, then serper
            provider_configs = [
                {"name": "brave", "api_key_env": "BRAVE_API_KEY"},
                {"name": "serper", "api_key_env": "SERPER_API_KEY"},
            ]
        return cls(provider_configs)
