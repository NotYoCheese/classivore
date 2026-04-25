#!/usr/bin/env python3
"""Per-run telemetry and reporting.

Records every classivore command invocation as a JSONL line in
data/runs/runs.jsonl. Includes timing, exit status, and a metrics dict
populated by the command being executed.

Usage:

    with RunRecorder("agent", taxonomy="iab-2.2", args=cli_args, runs_path=p) as run:
        # ...do work, populate run.metrics["search"], etc...
        ...

    print_summary(run.record, all_time=sum_metrics(load_runs(p, command="agent")))
"""

from classivore.runs.aggregator import load_runs, sum_metrics
from classivore.runs.recorder import RunRecorder, default_runs_path
from classivore.runs.reporter import format_summary, print_summary

__all__ = [
    "RunRecorder",
    "default_runs_path",
    "load_runs",
    "sum_metrics",
    "format_summary",
    "print_summary",
]
