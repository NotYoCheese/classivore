#!/usr/bin/env python3
"""Tests for RunRecorder."""

import json
from pathlib import Path

import pytest

from classivore.runs.recorder import RunRecorder


def _read_runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRunRecorder:
    def test_appends_one_record_on_clean_exit(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with RunRecorder("agent", "iab-2.2", {"max_iterations": 2}, runs_path=runs_path) as run:
            run.metrics["search"] = {"queries_by_provider": {"brave": 10}}

        records = _read_runs(runs_path)
        assert len(records) == 1
        record = records[0]
        assert record["command"] == "agent"
        assert record["taxonomy"] == "iab-2.2"
        assert record["args"] == {"max_iterations": 2}
        assert record["exit_status"] == "ok"
        assert record["metrics"]["search"]["queries_by_provider"]["brave"] == 10
        assert "started_at" in record
        assert "ended_at" in record
        assert "runtime_seconds" in record
        assert record["runtime_seconds"] >= 0
        assert "run_id" in record

    def test_records_error_status_on_exception(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with pytest.raises(RuntimeError):
            with RunRecorder("collect", "iab-2.2", {}, runs_path=runs_path) as run:
                run.metrics["scrape"] = {"fetched": 5}
                raise RuntimeError("boom")

        records = _read_runs(runs_path)
        assert len(records) == 1
        assert records[0]["exit_status"] == "error"
        assert records[0]["error"] == "boom"
        # Partial metrics should still be recorded
        assert records[0]["metrics"]["scrape"]["fetched"] == 5

    def test_records_interrupted_status_on_keyboard_interrupt(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with pytest.raises(KeyboardInterrupt):
            with RunRecorder("label", "iab-2.2", {}, runs_path=runs_path):
                raise KeyboardInterrupt()

        records = _read_runs(runs_path)
        assert records[0]["exit_status"] == "interrupted"

    def test_appends_multiple_runs(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path):
            pass
        with RunRecorder("collect", "iab-2.2", {}, runs_path=runs_path):
            pass
        with RunRecorder("label", "iab-2.2", {}, runs_path=runs_path):
            pass

        records = _read_runs(runs_path)
        assert len(records) == 3
        assert [r["command"] for r in records] == ["agent", "collect", "label"]

    def test_run_ids_are_unique(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path):
            pass
        with RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path):
            pass

        records = _read_runs(runs_path)
        assert records[0]["run_id"] != records[1]["run_id"]

    def test_metrics_dict_starts_empty(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        with RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path) as run:
            assert run.metrics == {}

    def test_creates_parent_directory(self, tmp_path):
        runs_path = tmp_path / "deeply" / "nested" / "runs.jsonl"

        with RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path):
            pass

        assert runs_path.exists()

    def test_record_property_available_after_exit(self, tmp_path):
        runs_path = tmp_path / "runs.jsonl"

        recorder = RunRecorder("agent", "iab-2.2", {}, runs_path=runs_path)
        with recorder:
            recorder.metrics["search"] = {"queries_by_provider": {"brave": 7}}

        assert recorder.record["metrics"]["search"]["queries_by_provider"]["brave"] == 7
        assert recorder.record["exit_status"] == "ok"
