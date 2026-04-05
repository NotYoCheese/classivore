#!/usr/bin/env python3
"""Label state persistence and crash recovery.

Tracks per-page labeling status through two stages:
  unlabeled → stage1_complete → stage2_complete (or error)

Persists atomically via temp+rename to survive crashes. Stores batch IDs
for resume, tier-1 triage results for stage 2 input, and final labels.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from classivore.persistence import atomic_json_save


class LabelState:
    """Manages labeling state with atomic JSON persistence."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "label_state.json"
        self.pages = {}
        self.started_at = None
        self.last_checkpoint_at = None
        self.stage1_batch_ids = []
        self.stage2_batch_ids = []

        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.pages = data.get("pages", {})
            self.started_at = data.get("started_at")
            self.last_checkpoint_at = data.get("last_checkpoint_at")
            self.stage1_batch_ids = data.get("stage1_batch_ids", [])
            self.stage2_batch_ids = data.get("stage2_batch_ids", [])

    def save(self):
        """Atomically save state to disk via temp+rename."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.started_at:
            self.started_at = now
        self.last_checkpoint_at = now

        data = {
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "stage1_batch_ids": self.stage1_batch_ids,
            "stage2_batch_ids": self.stage2_batch_ids,
            "stats": self._compute_stats(),
            "pages": self.pages,
        }
        atomic_json_save(data, self.state_file, directory=self.state_dir)

    def init_page(self, content_hash, url):
        """Register a page for labeling if not already tracked."""
        if content_hash in self.pages:
            return
        self.pages[content_hash] = {
            "url": url,
            "status": "unlabeled",
            "tier1_categories": None,
            "labels": None,
            "reasoning": None,
            "error": None,
        }

    def complete_stage1(self, content_hash, tier1_categories):
        """Record stage 1 results and transition to stage1_complete."""
        page = self.pages[content_hash]
        page["status"] = "stage1_complete"
        page["tier1_categories"] = tier1_categories

    def complete_stage2(self, content_hash, labels, reasoning):
        """Record stage 2 results and transition to stage2_complete."""
        page = self.pages[content_hash]
        page["status"] = "stage2_complete"
        page["labels"] = labels
        page["reasoning"] = reasoning

    def mark_error(self, content_hash, error_msg):
        """Mark a page as failed."""
        page = self.pages[content_hash]
        page["status"] = "error"
        page["error"] = error_msg

    def pages_needing_stage1(self):
        """Return content hashes of pages that need stage 1 triage."""
        return [h for h, p in self.pages.items() if p["status"] == "unlabeled"]

    def pages_needing_stage2(self):
        """Return content hashes of pages that need stage 2 classification."""
        return [h for h, p in self.pages.items() if p["status"] == "stage1_complete"]

    def get_tier1_for_page(self, content_hash):
        """Get tier-1 category names for a page (from stage 1 results)."""
        page = self.pages.get(content_hash, {})
        tier1 = page.get("tier1_categories")
        if not tier1:
            return []
        return [c["name"] for c in tier1]

    def is_complete(self, content_hash):
        """Check if a page has completed all labeling stages."""
        page = self.pages.get(content_hash, {})
        return page.get("status") == "stage2_complete"

    def _compute_stats(self):
        """Compute summary statistics from page statuses."""
        counts = {"unlabeled": 0, "stage1_complete": 0, "stage2_complete": 0, "error": 0}
        for page in self.pages.values():
            status = page.get("status", "unlabeled")
            counts[status] = counts.get(status, 0) + 1
        counts["total_pages"] = len(self.pages)
        return counts

    def summary(self):
        """Return summary statistics dict."""
        return self._compute_stats()

    def summary_str(self):
        """Return formatted summary string for CLI output."""
        stats = self._compute_stats()
        lines = [
            "Labeling Status",
            "=" * 40,
            f"  Total pages:      {stats['total_pages']}",
            f"  Unlabeled:        {stats['unlabeled']}",
            f"  Stage 1 complete: {stats['stage1_complete']}",
            f"  Stage 2 complete: {stats['stage2_complete']}",
            f"  Errors:           {stats['error']}",
        ]
        if self.started_at:
            lines.append(f"  Started:          {self.started_at}")
        if self.last_checkpoint_at:
            lines.append(f"  Last update:      {self.last_checkpoint_at}")
        return "\n".join(lines)
