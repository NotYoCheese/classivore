#!/usr/bin/env python3
"""Tests for the taxonomy loader."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from classivore.taxonomy.loader import (
    build_hierarchy,
    get_children,
    get_siblings,
    load_taxonomy,
    save_enriched_taxonomy,
)

SAMPLE_CSV = """\
id,parent_id,name,display_name,path,depth,is_leaf,children_count,aliases,difficulty
1,,Automotive,Automotive: Automotive,Automotive,1,False,2,cars | vehicles | autos,easy
2,1,Auto Body Styles,Automotive: Auto Body Styles,Automotive > Auto Body Styles,2,False,3,,medium
3,2,Sedan,Automotive: Sedan,Automotive > Auto Body Styles > Sedan,3,True,0,4-door car | saloon | family car,medium
4,2,SUV,Automotive: SUV,Automotive > Auto Body Styles > SUV,3,True,0,,
5,2,Coupe,Automotive: Coupe,Automotive > Auto Body Styles > Coupe,3,True,0,2-door car | sports coupe,medium
6,1,Auto Type,Automotive: Auto Type,Automotive > Auto Type,2,False,2,,hard
7,6,Budget Cars,Automotive: Budget Cars,Automotive > Auto Type > Budget Cars,3,True,0,cheap cars | affordable cars,easy
8,6,Luxury Cars,Automotive: Luxury Cars,Automotive > Auto Type > Luxury Cars,3,True,0,premium cars | high-end vehicles,easy
"""


@pytest.fixture
def taxonomy_csv(tmp_path):
    """Write sample taxonomy CSV and return a mock config."""
    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_text(SAMPLE_CSV)

    config = MagicMock()
    config.taxonomy_file = csv_path
    config.id_column = "id"
    config.name_column = "name"
    config.parent_column = "parent_id"
    return config


class TestLoadTaxonomy:
    def test_loads_all_rows(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        assert len(cats) == 8

    def test_path_parsing(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        sedan = next(c for c in cats if c["name"] == "Sedan")
        assert sedan["path"] == ["Automotive", "Auto Body Styles", "Sedan"]

    def test_root_path(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        root = next(c for c in cats if c["name"] == "Automotive")
        assert root["path"] == ["Automotive"]

    def test_boolean_parsing(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        root = next(c for c in cats if c["name"] == "Automotive")
        sedan = next(c for c in cats if c["name"] == "Sedan")
        assert root["is_leaf"] is False
        assert sedan["is_leaf"] is True

    def test_int_parsing(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        root = next(c for c in cats if c["name"] == "Automotive")
        sedan = next(c for c in cats if c["name"] == "Sedan")
        assert root["depth"] == 1
        assert sedan["depth"] == 3
        assert root["children_count"] == 2
        assert sedan["children_count"] == 0

    def test_empty_description_default(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        assert all(c["description"] == "" for c in cats)
        assert all(c["boundaries"] == "" for c in cats)

    def test_aliases_parsing(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        root = next(c for c in cats if c["name"] == "Automotive")
        assert root["aliases"] == ["cars", "vehicles", "autos"]
        sedan = next(c for c in cats if c["name"] == "Sedan")
        assert sedan["aliases"] == ["4-door car", "saloon", "family car"]

    def test_aliases_empty(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        suv = next(c for c in cats if c["name"] == "SUV")
        assert suv["aliases"] == []

    def test_difficulty_parsing(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        root = next(c for c in cats if c["name"] == "Automotive")
        assert root["difficulty"] == "easy"
        auto_type = next(c for c in cats if c["name"] == "Auto Type")
        assert auto_type["difficulty"] == "hard"

    def test_difficulty_defaults_to_medium(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        suv = next(c for c in cats if c["name"] == "SUV")
        assert suv["difficulty"] == "medium"

    def test_missing_file_raises(self, taxonomy_csv):
        taxonomy_csv.taxonomy_file = Path("/nonexistent/taxonomy.csv")
        with pytest.raises(FileNotFoundError):
            load_taxonomy(taxonomy_csv)


class TestHierarchy:
    def test_build_hierarchy(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        hierarchy = build_hierarchy(cats)
        # Root's children: Auto Body Styles, Auto Type
        assert len(hierarchy["1"]) == 2
        # Auto Body Styles' children: Sedan, SUV, Coupe
        assert len(hierarchy["2"]) == 3

    def test_get_siblings(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        hierarchy = build_hierarchy(cats)
        sedan = next(c for c in cats if c["name"] == "Sedan")
        siblings = get_siblings(sedan, hierarchy)
        assert sorted(siblings) == ["Coupe", "SUV"]

    def test_get_siblings_root_has_none(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        hierarchy = build_hierarchy(cats)
        root = next(c for c in cats if c["name"] == "Automotive")
        siblings = get_siblings(root, hierarchy)
        assert siblings == []

    def test_get_children(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        hierarchy = build_hierarchy(cats)
        body_styles = next(c for c in cats if c["name"] == "Auto Body Styles")
        children = get_children(body_styles, hierarchy)
        assert sorted(children) == ["Coupe", "SUV", "Sedan"]

    def test_get_children_leaf_has_none(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        hierarchy = build_hierarchy(cats)
        sedan = next(c for c in cats if c["name"] == "Sedan")
        children = get_children(sedan, hierarchy)
        assert children == []


class TestSaveEnrichedTaxonomy:
    def test_roundtrip(self, taxonomy_csv):
        cats = load_taxonomy(taxonomy_csv)
        # Add descriptions to a couple
        cats[0]["description"] = "Vehicles and transportation"
        cats[0]["boundaries"] = "Distinct from Travel by focus on vehicles"

        output_path = taxonomy_csv.taxonomy_file.parent / "enriched.csv"
        save_enriched_taxonomy(cats, output_path)

        # Reload
        taxonomy_csv.taxonomy_file = output_path
        reloaded = load_taxonomy(taxonomy_csv)

        assert len(reloaded) == len(cats)
        assert reloaded[0]["description"] == "Vehicles and transportation"
        assert reloaded[0]["boundaries"] == "Distinct from Travel by focus on vehicles"
        assert reloaded[0]["aliases"] == ["cars", "vehicles", "autos"]
        assert reloaded[0]["difficulty"] == "easy"
        assert reloaded[2]["path"] == ["Automotive", "Auto Body Styles", "Sedan"]
        assert reloaded[2]["is_leaf"] is True
        assert reloaded[2]["aliases"] == ["4-door car", "saloon", "family car"]
        # SUV has empty aliases
        assert reloaded[3]["aliases"] == []
        assert reloaded[3]["difficulty"] == "medium"

    def test_creates_parent_dirs(self, tmp_path):
        output_path = tmp_path / "sub" / "dir" / "enriched.csv"
        save_enriched_taxonomy([], output_path)
        assert output_path.exists()
