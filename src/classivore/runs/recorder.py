#!/usr/bin/env python3
"""RunRecorder context manager — appends one record per command invocation."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_runs_path(data_dir: Path) -> Path:
    """Resolve the runs.jsonl path under the given data dir."""
    return Path(data_dir) / "runs" / "runs.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunRecorder:
    """Record a single classivore run as one JSONL line.

    Use as a context manager. Populate `metrics` during the run; on exit
    (clean, exception, or KeyboardInterrupt) the record is appended atomically.
    """

    def __init__(self, command: str, taxonomy: str, args: dict, runs_path: Path):
        self.command = command
        self.taxonomy = taxonomy
        self.args = dict(args) if args else {}
        self.runs_path = Path(runs_path)
        self.metrics: dict = {}

        self._started_monotonic: float | None = None
        self._record: dict | None = None

    def __enter__(self) -> "RunRecorder":
        self._started_monotonic = time.monotonic()
        self._record = {
            "run_id": str(uuid.uuid4()),
            "command": self.command,
            "taxonomy": self.taxonomy,
            "args": self.args,
            "started_at": _utc_now_iso(),
            "ended_at": None,
            "runtime_seconds": 0.0,
            "exit_status": "ok",
            "metrics": self.metrics,
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        runtime = time.monotonic() - (self._started_monotonic or time.monotonic())
        self._record["ended_at"] = _utc_now_iso()
        self._record["runtime_seconds"] = round(runtime, 3)
        self._record["metrics"] = self.metrics

        if exc_type is None:
            self._record["exit_status"] = "ok"
        elif issubclass(exc_type, KeyboardInterrupt):
            self._record["exit_status"] = "interrupted"
        else:
            self._record["exit_status"] = "error"
            self._record["error"] = str(exc_val) if exc_val is not None else exc_type.__name__

        self._append()
        return False  # don't suppress exceptions

    @property
    def record(self) -> dict:
        return self._record or {}

    def _append(self) -> None:
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.runs_path, "a") as f:
            f.write(json.dumps(self._record) + "\n")
