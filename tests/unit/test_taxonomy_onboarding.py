#!/usr/bin/env python3
"""Tests for taxonomy onboarding helpers."""

import csv

from classivore.taxonomy.onboarding import normalize_taxonomy_csv


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestNormalizeTaxonomyCsv:
    def test_computes_hierarchy_columns_from_parent_id(self, tmp_path):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        _write_csv(src, ["id", "parent_id", "name"], [
            {"id": "1", "parent_id": "", "name": "Sports"},
            {"id": "2", "parent_id": "1", "name": "Soccer"},
            {"id": "3", "parent_id": "2", "name": "World Cup"},
            {"id": "4", "parent_id": "1", "name": "Tennis"},
        ])

        normalize_taxonomy_csv(src, dst)

        with open(dst, newline="") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}

        assert rows["1"]["path"] == "Sports"
        assert rows["1"]["depth"] == "1"
        assert rows["1"]["is_leaf"] == "False"
        assert rows["1"]["children_count"] == "2"

        assert rows["2"]["path"] == "Sports > Soccer"
        assert rows["2"]["depth"] == "2"
        assert rows["2"]["is_leaf"] == "False"
        assert rows["2"]["children_count"] == "1"

        assert rows["3"]["path"] == "Sports > Soccer > World Cup"
        assert rows["3"]["depth"] == "3"
        assert rows["3"]["is_leaf"] == "True"
        assert rows["3"]["children_count"] == "0"

        assert rows["4"]["is_leaf"] == "True"

    def test_preserves_existing_columns(self, tmp_path):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        _write_csv(src, ["id", "parent_id", "name", "description"], [
            {"id": "1", "parent_id": "", "name": "Sports", "description": "Athletic events"},
        ])

        normalize_taxonomy_csv(src, dst)

        with open(dst, newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["description"] == "Athletic events"
        assert rows[0]["depth"] == "1"

    def test_defaults_display_name_to_name(self, tmp_path):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        _write_csv(src, ["id", "parent_id", "name"], [
            {"id": "1", "parent_id": "", "name": "Sports"},
        ])

        normalize_taxonomy_csv(src, dst)

        with open(dst, newline="") as f:
            row = next(csv.DictReader(f))
        assert row["display_name"] == "Sports"

    def test_respects_existing_display_name(self, tmp_path):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        _write_csv(src, ["id", "parent_id", "name", "display_name"], [
            {"id": "1", "parent_id": "", "name": "Sports", "display_name": "All Sports"},
        ])

        normalize_taxonomy_csv(src, dst)

        with open(dst, newline="") as f:
            row = next(csv.DictReader(f))
        assert row["display_name"] == "All Sports"

    def test_custom_column_names(self, tmp_path):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        _write_csv(src, ["Unique ID", "Parent", "Name"], [
            {"Unique ID": "100", "Parent": "", "Name": "Root"},
            {"Unique ID": "200", "Parent": "100", "Name": "Child"},
        ])

        normalize_taxonomy_csv(
            src, dst,
            id_col="Unique ID", name_col="Name", parent_col="Parent",
        )

        with open(dst, newline="") as f:
            rows = {r["Unique ID"]: r for r in csv.DictReader(f)}

        assert rows["200"]["path"] == "Root > Child"
        assert rows["200"]["depth"] == "2"
