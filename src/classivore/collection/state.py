#!/usr/bin/env python3
"""Collection state persistence and resumability.

Tracks per-category progress (queries tried, pages collected, domain diversity)
and per-URL status (collected/failed/filtered/duplicate) to enable resume
after interrupts and prevent redundant work.

State is saved atomically via temp+rename to survive crashes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from classivore.persistence import atomic_json_save


class CollectionState:
    """Manages collection state with atomic JSON persistence."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        self.categories = {}
        self.urls = {}
        self.started_at = None
        self.last_checkpoint_at = None
        self.error_counts = {
            "search_errors": 0,
            "fetch_errors": 0,
            "filtered": 0,
            "duplicates": 0,
        }

        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.categories = data.get("categories", {})
            self.urls = data.get("urls", {})
            self.started_at = data.get("started_at")
            self.last_checkpoint_at = data.get("last_checkpoint_at")
            self.error_counts = data.get("error_counts", self.error_counts)

    def save(self):
        """Atomically save state to disk via temp+rename."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.started_at:
            self.started_at = now
        self.last_checkpoint_at = now

        data = {
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "error_counts": self.error_counts,
            "categories": self.categories,
            "urls": self.urls,
        }
        atomic_json_save(data, self.state_file, directory=self.state_dir)

    def init_category(self, name, target):
        """Initialize category tracking if not already present."""
        if name in self.categories:
            return
        self.categories[name] = {
            "target": target,
            "collected": 0,
            "queries_tried": [],
            "source_domains": {},
        }

    def is_satisfied(self, name):
        """Check if a category has met its collection target."""
        cat = self.categories.get(name)
        if not cat:
            return False
        return cat["collected"] >= cat["target"]

    def record_query(self, category, query):
        """Record a query as tried for a category."""
        queries = self.categories[category]["queries_tried"]
        if query not in queries:
            queries.append(query)

    def has_query(self, category, query):
        """Check if a query has already been tried for a category."""
        cat = self.categories.get(category)
        if not cat:
            return False
        return query in cat["queries_tried"]

    def record_url(self, url, category, status, source):
        """Record a URL's collection result.

        Args:
            url: The page URL.
            category: Category name this URL was collected for.
            status: One of 'collected', 'failed', 'filtered', 'duplicate'.
            source: One of 'commoncrawl', 'live_scrape', 'search'.
        """
        self.urls[url] = {
            "category": category,
            "status": status,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Update error counters
        if status == "failed":
            self.error_counts["fetch_errors"] += 1
        elif status == "filtered":
            self.error_counts["filtered"] += 1
        elif status == "duplicate":
            self.error_counts["duplicates"] += 1

        cat = self.categories.get(category)
        if not cat:
            return

        if status == "collected":
            cat["collected"] += 1
            domain = urlparse(url).netloc
            cat["source_domains"][domain] = cat["source_domains"].get(domain, 0) + 1

    def record_search_error(self):
        """Increment search error counter."""
        self.error_counts["search_errors"] += 1

    def is_url_known(self, url):
        """Check if a URL has already been processed."""
        return url in self.urls

    def get_domain_count(self, category, domain):
        """Get number of pages collected from a domain for a category."""
        cat = self.categories.get(category)
        if not cat:
            return 0
        return cat["source_domains"].get(domain, 0)

    def recent_urls(self, minutes=10):
        """Count URLs processed in the last N minutes.

        Args:
            minutes: Time window in minutes.

        Returns:
            Number of URLs with timestamps within the window.
        """
        now = datetime.now(timezone.utc)
        count = 0
        for entry in self.urls.values():
            ts = entry.get("timestamp")
            if not ts:
                continue
            try:
                url_time = datetime.fromisoformat(ts)
                if (now - url_time).total_seconds() <= minutes * 60:
                    count += 1
            except (ValueError, TypeError):
                continue
        return count

    def coverage_histogram(self):
        """Return category counts bucketed by collected pages.

        Returns:
            Dict with bucket labels as keys and category counts as values.
        """
        buckets = {"0": 0, "1-5": 0, "6-10": 0, "11-50": 0, "50+": 0}
        for cat in self.categories.values():
            n = cat["collected"]
            if n == 0:
                buckets["0"] += 1
            elif n <= 5:
                buckets["1-5"] += 1
            elif n <= 10:
                buckets["6-10"] += 1
            elif n <= 50:
                buckets["11-50"] += 1
            else:
                buckets["50+"] += 1
        return buckets

    def summary(self):
        """Return a summary dict of collection progress."""
        total_collected = sum(c["collected"] for c in self.categories.values())
        total_target = sum(c["target"] for c in self.categories.values())
        satisfied = sum(1 for c in self.categories.values() if c["collected"] >= c["target"])
        return {
            "total_categories": len(self.categories),
            "satisfied_categories": satisfied,
            "total_collected": total_collected,
            "total_target": total_target,
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "error_counts": dict(self.error_counts),
        }
