#!/usr/bin/env python3
"""Load and aggregate run records from runs.jsonl."""

import json
from pathlib import Path
from typing import Iterable, Optional


def load_runs(
    runs_path: Path,
    command: Optional[str] = None,
    taxonomy: Optional[str] = None,
) -> list[dict]:
    """Load all runs from runs.jsonl, optionally filtered.

    Args:
        runs_path: Path to runs.jsonl.
        command: If set, only return records with matching command.
        taxonomy: If set, only return records with matching taxonomy.

    Returns:
        List of run record dicts. Empty list if file missing.
    """
    runs_path = Path(runs_path)
    if not runs_path.exists():
        return []

    records = []
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if command and record.get("command") != command:
                continue
            if taxonomy and record.get("taxonomy") != taxonomy:
                continue
            records.append(record)
    return records


def sum_metrics(runs: Iterable[dict]) -> dict:
    """Sum all numeric metric fields across runs.

    Recursively merges nested dicts, summing numeric leaves.
    Non-numeric leaves (strings, lists, etc.) are skipped — they
    represent samples or identifiers that don't aggregate cleanly.

    Args:
        runs: Iterable of run record dicts.

    Returns:
        Dict shaped like a single run's `metrics` field, with summed values.
    """
    totals: dict = {}
    for run in runs:
        metrics = run.get("metrics") or {}
        _merge_sum(totals, metrics)
    return totals


def _merge_sum(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            sub = target.setdefault(key, {})
            if not isinstance(sub, dict):
                continue
            _merge_sum(sub, value)
        elif isinstance(value, bool):
            # bool is a subclass of int — skip explicitly so True+True=2 doesn't
            # silently accumulate
            continue
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        # else: skip non-numeric leaves
