#!/usr/bin/env python3
"""Collection state persistence and resumability.

Tracks per-category progress (queries tried, pages collected, domain diversity)
and per-URL status (collected/failed/filtered/duplicate) to enable resume
after interrupts and prevent redundant work.

State is saved atomically via temp+rename to survive crashes.
"""

import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse


class CollectionState:
    """Manages collection state with atomic JSON persistence."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        self.categories = {}
        self.urls = {}

        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.categories = data.get("categories", {})
            self.urls = data.get("urls", {})

    def save(self):
        """Atomically save state to disk via temp+rename."""
        data = {"categories": self.categories, "urls": self.urls}
        fd, tmp_path = tempfile.mkstemp(
            dir=self.state_dir, prefix=".state_", suffix=".tmp"
        )
        try:
            with open(fd, "w") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(self.state_file)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

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
            source: One of 'commoncrawl', 'live_scrape'.
        """
        self.urls[url] = {
            "category": category,
            "status": status,
            "source": source,
        }

        cat = self.categories.get(category)
        if not cat:
            return

        if status == "collected":
            cat["collected"] += 1
            domain = urlparse(url).netloc
            cat["source_domains"][domain] = cat["source_domains"].get(domain, 0) + 1

    def is_url_known(self, url):
        """Check if a URL has already been processed."""
        return url in self.urls

    def get_domain_count(self, category, domain):
        """Get number of pages collected from a domain for a category."""
        cat = self.categories.get(category)
        if not cat:
            return 0
        return cat["source_domains"].get(domain, 0)

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
        }
