#!/usr/bin/env python3
"""Tests for shared persistence utilities."""

import json

import pytest

from classivore.persistence import (
    append_ndjson,
    atomic_json_save,
    iter_ndjson,
    load_ndjson,
)


class TestAtomicJsonSave:
    """Test atomic JSON save."""

    def test_creates_file(self, tmp_path):
        target = tmp_path / "test.json"
        atomic_json_save({"key": "value"}, target)
        assert target.exists()
        assert json.loads(target.read_text()) == {"key": "value"}

    def test_overwrites_existing(self, tmp_path):
        target = tmp_path / "test.json"
        target.write_text('{"old": true}')
        atomic_json_save({"new": True}, target)
        assert json.loads(target.read_text()) == {"new": True}

    def test_no_partial_write_on_error(self, tmp_path):
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')

        # Non-serializable value should fail
        with pytest.raises(TypeError):
            atomic_json_save({"bad": object()}, target)

        # Original file should be untouched
        assert json.loads(target.read_text()) == {"original": True}

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "sub" / "dir" / "test.json"
        atomic_json_save({"nested": True}, target)
        assert target.exists()

    def test_custom_directory(self, tmp_path):
        target = tmp_path / "output" / "test.json"
        target.parent.mkdir(parents=True)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        atomic_json_save({"data": 1}, target, directory=temp_dir)
        assert target.exists()
        # No leftover temp files
        assert len(list(temp_dir.iterdir())) == 0


class TestNdjson:
    """Test NDJSON load/iter/append."""

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        assert load_ndjson(path) == []

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "missing.json"
        assert load_ndjson(path) == []

    def test_load_records(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        records = load_ndjson(path)
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        assert len(load_ndjson(path)) == 2

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"good": 1}\nnot json\n{"also_good": 2}\n')
        records = load_ndjson(path)
        assert len(records) == 2
        assert records[0] == {"good": 1}

    def test_iter_streams_without_loading_all(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = iter_ndjson(path)
        assert next(it) == {"a": 1}
        assert next(it) == {"b": 2}

    def test_append_creates_file(self, tmp_path):
        path = tmp_path / "new.json"
        append_ndjson(path, [{"x": 1}])
        assert path.exists()
        assert load_ndjson(path) == [{"x": 1}]

    def test_append_to_existing(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1}\n')
        append_ndjson(path, [{"b": 2}, {"c": 3}])
        records = load_ndjson(path)
        assert len(records) == 3

    def test_append_empty_list_is_noop(self, tmp_path):
        path = tmp_path / "data.json"
        append_ndjson(path, [])
        assert not path.exists()

    def test_append_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "data.json"
        append_ndjson(path, [{"nested": True}])
        assert path.exists()

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "roundtrip.json"
        original = [{"id": i, "name": f"item-{i}"} for i in range(10)]
        append_ndjson(path, original)
        loaded = load_ndjson(path)
        assert loaded == original
