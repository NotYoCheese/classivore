#!/usr/bin/env python3
"""Tests for configuration loading."""

import pytest
from pathlib import Path

from classivore.config.settings import TaxonomyConfig, load_taxonomy_config


class TestTaxonomyConfig:
    """Test taxonomy config loading from YAML."""

    def test_load_iab_config(self):
        """IAB 2.2 config loads with correct defaults."""
        config = load_taxonomy_config("iab-2.2")
        assert config.slug == "iab-2.2"
        assert config.name == "IAB Content Taxonomy"
        assert config.classification_type == "multi_label"
        assert config.max_labels == 3
        assert config.focal_alpha == 0.75
        assert config.focal_gamma == 3.5

    def test_missing_taxonomy_raises(self):
        """Loading a nonexistent taxonomy raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_taxonomy_config("nonexistent-taxonomy")

    def test_taxonomy_file_path_relative(self):
        """Taxonomy file path resolves relative to taxonomy directory."""
        config = load_taxonomy_config("iab-2.2")
        assert config.taxonomy_file.name == "taxonomy.csv"
        assert "iab_2.2" in str(config.taxonomy_file)

    def test_excluded_categories_loaded(self):
        """Excluded categories list loads from config."""
        config = load_taxonomy_config("iab-2.2")
        assert isinstance(config.excluded_categories, list)
        assert len(config.excluded_categories) > 0

    def test_enrichment_model_loaded(self):
        """Enrichment model loads from config."""
        config = load_taxonomy_config("iab-2.2")
        assert config.enrichment_model == "claude-haiku-4-5-20251001"

    def test_enrichment_max_tokens_loaded(self):
        """Enrichment max tokens loads from config."""
        config = load_taxonomy_config("iab-2.2")
        assert config.enrichment_max_tokens == 300

    def test_labeling_settings_loaded(self):
        """Labeling settings load from config."""
        config = load_taxonomy_config("iab-2.2")
        assert isinstance(config.labeling_model, str)
        assert config.stage1_max_tokens > 0
        assert config.stage2_max_tokens > 0
        assert 0 < config.tier1_confidence_threshold < 1
        assert config.labeling_temperature >= 0
        assert config.text_truncation_words > 0

    def test_excluded_tier1_categories_loaded(self):
        """Excluded tier-1 categories load from config."""
        config = load_taxonomy_config("iab-2.2")
        assert "Content Language" in config.excluded_tier1_categories
        assert "Content Source Geo" in config.excluded_tier1_categories
        assert len(config.excluded_tier1_categories) == 6

    def test_collection_settings_loaded(self):
        """Collection settings load from config."""
        config = load_taxonomy_config("iab-2.2")
        assert config.targets_by_difficulty == {"easy": 30, "medium": 50, "hard": 80, "default": 80}
        assert config.target_per_category == 80  # convenience alias = default tier
        assert config.max_queries_by_difficulty["hard"] == 30
        assert config.max_per_domain_per_category == 3
        assert config.commoncrawl_crawl_id == "CC-MAIN-2026-08"
        # Helper methods
        assert config.get_target("hard") == 80
        assert config.get_target("") == 80  # unknown difficulty → default
        assert config.use_llm_queries("hard", 0) is False  # hard starts at iteration 1
        assert config.use_llm_queries("hard", 1) is True
        assert config.use_llm_queries("easy", 0) is False
        assert config.use_llm_queries("easy", 3) is True
        assert config.query_model == "claude-haiku-4-5-20251001"


class TestDataDir:
    """Test data directory resolution."""

    def test_default_data_dir(self):
        from classivore.config.settings import get_data_dir
        assert get_data_dir() == Path("data")

    def test_override_data_dir(self):
        from classivore.config.settings import get_data_dir
        assert get_data_dir("/tmp/test") == Path("/tmp/test")
