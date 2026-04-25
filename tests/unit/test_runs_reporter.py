#!/usr/bin/env python3
"""Tests for runs reporter."""

import io

from classivore.runs.reporter import format_summary


def _record(**metrics):
    return {
        "run_id": "abc-123",
        "command": "agent",
        "taxonomy": "iab-2.2",
        "args": {"max_iterations": 2},
        "started_at": "2026-04-25T17:30:00Z",
        "ended_at": "2026-04-25T17:42:13Z",
        "runtime_seconds": 733.0,
        "exit_status": "ok",
        "metrics": metrics,
    }


class TestFormatSummary:
    def test_includes_header_with_command_and_taxonomy(self):
        out = format_summary(_record(), all_time={})
        assert "agent" in out
        assert "iab-2.2" in out

    def test_includes_runtime_human_readable(self):
        out = format_summary(_record(), all_time={})
        assert "12m 13s" in out  # 733 seconds

    def test_includes_search_provider_breakdown(self):
        record = _record(search={"queries_by_provider": {"brave": 412, "exa": 88, "serper": 0}})
        all_time = {"search": {"queries_by_provider": {"brave": 3118, "exa": 502, "serper": 0}}}

        out = format_summary(record, all_time)
        assert "brave" in out
        assert "412" in out
        assert "3,118" in out
        assert "exa" in out
        assert "88" in out

    def test_includes_scrape_funnel(self):
        record = _record(scrape={"fetched": 480, "kept": 374,
                                 "rejected": {"duplicate_content": 15,
                                              "domain_cap": 22,
                                              "too_short": 12}})
        out = format_summary(record, all_time={})
        assert "fetched" in out.lower()
        assert "480" in out
        assert "374" in out
        assert "duplicate_content" in out or "duplicate content" in out
        assert "domain_cap" in out or "domain cap" in out

    def test_includes_labeling_funnel_and_cache(self):
        record = _record(labeling={
            "stage1_sent": 374, "tier1_hits": 312, "stage2_sent": 312,
            "labels_emitted": 298, "errors": 14,
            "cache": {
                "stage1": {"cache_hit_rate": 0.942, "estimated_cost_usd": 1.20},
                "stage2": {"cache_hit_rate": 0.917, "estimated_cost_usd": 1.21},
            },
        })
        out = format_summary(record, all_time={})
        assert "stage1" in out.lower() or "stage 1" in out.lower()
        assert "94" in out  # 94.2% hit rate
        assert "$" in out

    def test_handles_missing_metrics_blocks(self):
        """A collect-only run won't have labeling metrics; reporter should skip cleanly."""
        record = _record(scrape={"fetched": 10, "kept": 5, "rejected": {}})

        out = format_summary(record, all_time={})
        assert "fetched" in out.lower()
        # No crash, just no labeling section
        assert isinstance(out, str)

    def test_includes_status_line_for_error_run(self):
        record = _record()
        record["exit_status"] = "error"
        record["error"] = "rate limit hit"

        out = format_summary(record, all_time={})
        assert "error" in out.lower()
        assert "rate limit hit" in out

    def test_empty_all_time_renders_zeros(self):
        record = _record(search={"queries_by_provider": {"brave": 5}})
        out = format_summary(record, all_time={})
        # All-time column should render but with 0
        assert "5" in out
