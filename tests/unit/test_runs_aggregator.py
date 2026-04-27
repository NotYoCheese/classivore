#!/usr/bin/env python3
"""Tests for runs aggregator."""

import json
from pathlib import Path

import pytest

from classivore.runs.aggregator import load_runs, sum_metrics


def _write_runs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_run(command="agent", taxonomy="iab-2.2", **metrics):
    return {
        "run_id": "x",
        "command": command,
        "taxonomy": taxonomy,
        "args": {},
        "started_at": "2026-04-25T17:30:00Z",
        "ended_at": "2026-04-25T17:35:00Z",
        "runtime_seconds": 300.0,
        "exit_status": "ok",
        "metrics": metrics,
    }


class TestLoadRuns:
    def test_returns_empty_when_file_missing(self, tmp_path):
        assert load_runs(tmp_path / "missing.jsonl") == []

    def test_returns_all_records(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        _write_runs(path, [
            _make_run(command="agent"),
            _make_run(command="collect"),
            _make_run(command="label"),
        ])

        records = load_runs(path)
        assert len(records) == 3

    def test_filters_by_command(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        _write_runs(path, [
            _make_run(command="agent"),
            _make_run(command="collect"),
            _make_run(command="agent"),
        ])

        agents = load_runs(path, command="agent")
        assert len(agents) == 2
        assert all(r["command"] == "agent" for r in agents)

    def test_filters_by_taxonomy(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        _write_runs(path, [
            _make_run(taxonomy="iab-2.2"),
            _make_run(taxonomy="iptc-media"),
            _make_run(taxonomy="iab-2.2"),
        ])

        iab = load_runs(path, taxonomy="iab-2.2")
        assert len(iab) == 2

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        path.write_text("not json\n" + json.dumps(_make_run()) + "\n")

        assert len(load_runs(path)) == 1


class TestSumMetrics:
    def test_sums_flat_int_fields(self):
        runs = [
            _make_run(scrape={"fetched": 10, "kept": 8}),
            _make_run(scrape={"fetched": 20, "kept": 15}),
        ]

        totals = sum_metrics(runs)
        assert totals["scrape"]["fetched"] == 30
        assert totals["scrape"]["kept"] == 23

    def test_sums_nested_dicts(self):
        runs = [
            _make_run(search={"queries_by_provider": {"brave": 100, "exa": 20}}),
            _make_run(search={"queries_by_provider": {"brave": 50, "serper": 30}}),
        ]

        totals = sum_metrics(runs)
        assert totals["search"]["queries_by_provider"]["brave"] == 150
        assert totals["search"]["queries_by_provider"]["exa"] == 20
        assert totals["search"]["queries_by_provider"]["serper"] == 30

    def test_handles_missing_metrics_blocks(self):
        runs = [
            _make_run(scrape={"fetched": 10}),
            _make_run(),  # no metrics at all
            _make_run(labeling={"stage1_sent": 5}),
        ]

        totals = sum_metrics(runs)
        assert totals["scrape"]["fetched"] == 10
        assert totals["labeling"]["stage1_sent"] == 5

    def test_empty_runs_returns_empty(self):
        assert sum_metrics([]) == {}

    def test_preserves_float_fields(self):
        runs = [
            _make_run(labeling={"cache": {"stage1": {"estimated_cost_usd": 1.5}}}),
            _make_run(labeling={"cache": {"stage1": {"estimated_cost_usd": 2.25}}}),
        ]

        totals = sum_metrics(runs)
        assert totals["labeling"]["cache"]["stage1"]["estimated_cost_usd"] == pytest.approx(3.75)

    def test_skips_non_numeric_fields(self):
        """String / list fields aren't summable; aggregator should ignore them."""
        runs = [
            _make_run(labeling={"errors_sample": ["a", "b"], "stage1_sent": 5}),
            _make_run(labeling={"errors_sample": ["c"], "stage1_sent": 10}),
        ]

        totals = sum_metrics(runs)
        assert totals["labeling"]["stage1_sent"] == 15
        assert "errors_sample" not in totals["labeling"]
