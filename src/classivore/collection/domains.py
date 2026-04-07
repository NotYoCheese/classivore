#!/usr/bin/env python3
"""Per-domain quality scoring and blocklist management.

Tracks success/failure rates per domain to auto-block low-quality sources
(below 20% success after 5+ attempts). Maintains a separate manual blocklist
that persists across runs.
"""

import json
from pathlib import Path

from classivore.persistence import atomic_json_save

MIN_ATTEMPTS_FOR_BLOCK = 5
MIN_SUCCESS_RATE = 0.20


class DomainTracker:
    """Tracks domain quality and manages blocklist."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.scores_file = self.state_dir / "domain_scores.json"
        self.blocklist_file = self.state_dir / "domain_blocklist.json"
        self.scores = {}
        self.blocklist = set()

        if self.scores_file.exists():
            self.scores = json.loads(self.scores_file.read_text())
        if self.blocklist_file.exists():
            self.blocklist = set(json.loads(self.blocklist_file.read_text()))

    def save(self):
        """Save scores and blocklist atomically."""
        atomic_json_save(self.scores, self.scores_file, directory=self.state_dir)
        atomic_json_save(
            sorted(self.blocklist), self.blocklist_file, directory=self.state_dir,
        )

    def record_result(self, domain, success):
        """Record a scrape attempt result for a domain."""
        if domain not in self.scores:
            self.scores[domain] = {"successes": 0, "attempts": 0}
        self.scores[domain]["attempts"] += 1
        if success:
            self.scores[domain]["successes"] += 1

    def success_rate(self, domain):
        """Get success rate for a domain, or None if no data."""
        entry = self.scores.get(domain)
        if not entry or entry["attempts"] == 0:
            return None
        return entry["successes"] / entry["attempts"]

    def is_blocked(self, domain):
        """Check if a domain is blocked (manual or auto-block)."""
        if domain in self.blocklist:
            return True
        entry = self.scores.get(domain)
        if not entry or entry["attempts"] < MIN_ATTEMPTS_FOR_BLOCK:
            return False
        return (entry["successes"] / entry["attempts"]) < MIN_SUCCESS_RATE

    def add_to_blocklist(self, domain):
        """Manually block a domain."""
        self.blocklist.add(domain)

    def remove_from_blocklist(self, domain):
        """Remove a domain from the manual blocklist."""
        self.blocklist.discard(domain)

    def audit_report(self):
        """Generate a human-readable domain quality report."""
        lines = []

        if not self.scores and not self.blocklist:
            return "No domain data recorded yet."

        if self.scores:
            sorted_domains = sorted(
                self.scores.items(),
                key=lambda x: x[1]["successes"] / max(x[1]["attempts"], 1),
            )
            lines.append("Domain Quality Report")
            lines.append("=" * 60)
            lines.append(f"{'Domain':<35} {'Rate':>6} {'Ok':>4} {'Total':>5}  Status")
            lines.append("-" * 60)

            for domain, entry in sorted_domains:
                rate = entry["successes"] / max(entry["attempts"], 1)
                status = ""
                if domain in self.blocklist:
                    status = "MANUAL BLOCK"
                elif self.is_blocked(domain):
                    status = "AUTO-BLOCKED"
                lines.append(
                    f"{domain:<35} {rate:>5.0%} {entry['successes']:>4} {entry['attempts']:>5}  {status}"
                )

        if self.blocklist:
            manual_only = self.blocklist - set(self.scores.keys())
            if manual_only:
                lines.append("")
                lines.append("Manual blocklist (no scrape data):")
                for domain in sorted(manual_only):
                    lines.append(f"  {domain}")

        return "\n".join(lines)
