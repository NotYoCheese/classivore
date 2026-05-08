#!/usr/bin/env python3
"""Shared persistence utilities for atomic JSON saves and NDJSON I/O."""

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from classivore.logging_config import get_logger

logger = get_logger(__name__)


def atomic_json_save(data: dict, target: Path, directory: Path | None = None) -> None:
    """Write JSON atomically via temp file + rename.

    Creates a temp file in the same directory as target, writes data,
    then atomically replaces the target. If anything fails, the temp
    file is cleaned up and the original target is untouched.

    Args:
        data: Dict to serialize as JSON.
        target: Final file path.
        directory: Directory for temp file. Defaults to target's parent.
    """
    if directory is None:
        directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


@contextmanager
def atomic_writer(target: Path):
    """Yield a text-mode file handle that becomes `target` on success.

    Writes go to a hidden temp file in the same directory; on a clean
    exit the temp file replaces the target via os.replace (atomic on
    POSIX). On any exception the temp file is removed and the original
    target is left untouched. Use for streaming NDJSON or other
    line-by-line writes where atomic_json_save's dict signature does
    not fit.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=".", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            yield f
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_ndjson(path: Path) -> list[dict]:
    """Load all records from an NDJSON file.

    Skips blank lines and lines that fail JSON parsing.

    Args:
        path: Path to NDJSON file.

    Returns:
        List of parsed dicts.
    """
    return list(iter_ndjson(path))


def iter_ndjson(path: Path) -> Iterator[dict]:
    """Stream records from an NDJSON file without loading all into memory.

    Skips blank lines and lines that fail JSON parsing.

    Args:
        path: Path to NDJSON file.

    Yields:
        Parsed dicts, one per line.
    """
    if not path.exists():
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug("ndjson_line_decode_failed", path=str(path), error=str(e))
                continue


def append_ndjson(path: Path, records: list[dict]) -> None:
    """Append records to an NDJSON file.

    Creates the file and parent directories if they don't exist.

    Args:
        path: Path to NDJSON file.
        records: List of dicts to append as JSON lines.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
