#!/usr/bin/env python3
"""
Central configuration loader.

Reads taxonomy config.yaml and provides typed access to all settings.
Environment variables override config file values.
"""

import os
from pathlib import Path
from typing import Optional

import yaml


class TaxonomyConfig:
    """Configuration for a specific taxonomy."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.taxonomy_dir = config_path.parent

        with open(config_path, "r") as f:
            self._raw = yaml.safe_load(f)

        # Identity
        self.name: str = self._raw["name"]
        self.version: str = self._raw["version"]
        self.slug: str = self._raw["slug"]

        # File paths (relative to taxonomy dir)
        self.taxonomy_file: Path = self.taxonomy_dir / self._raw["taxonomy_file"]
        self.enriched_file: Optional[Path] = None
        if self._raw.get("enriched_file"):
            self.enriched_file = self.taxonomy_dir / self._raw["enriched_file"]

        # Column mapping
        self.id_column: str = self._raw["id_column"]
        self.name_column: str = self._raw["name_column"]
        self.parent_column: str = self._raw.get("parent_column", "parent_id")
        self.description_column: Optional[str] = self._raw.get("description_column")

        # Classification settings
        self.classification_type: str = self._raw.get("classification_type", "multi_label")
        self.max_labels: int = self._raw.get("max_labels", 3)
        self.min_confidence: float = self._raw.get("min_confidence", 0.5)

        # Training settings
        self.model_base: str = self._raw.get("model_base", "microsoft/deberta-v3-large")
        self.batch_size: int = self._raw.get("batch_size", 8)
        self.learning_rate: float = self._raw.get("learning_rate", 2e-5)
        self.max_length: int = self._raw.get("max_length", 512)
        self.num_epochs: int = self._raw.get("num_epochs", 3)

        # Focal loss
        focal = self._raw.get("focal_loss", {})
        self.focal_alpha: float = focal.get("alpha", 0.75)
        self.focal_gamma: float = focal.get("gamma", 3.5)
        self.class_weight_cap: float = self._raw.get("class_weight_cap", 7.0)

        # Enrichment
        enrichment = self._raw.get("enrichment", {})
        self.enrichment_model: str = enrichment.get("model", "claude-haiku-4-5-20251001")
        self.enrichment_max_tokens: int = enrichment.get("max_tokens_per_category", 150)

        # Collection
        collection = self._raw.get("collection", {})
        self.target_per_category: int = collection.get("target_per_category", 7)
        self.max_queries_per_category: int = collection.get("max_queries_per_category", 6)
        self.max_per_domain_per_category: int = collection.get("max_per_domain_per_category", 50)
        self.commoncrawl_crawl_id: Optional[str] = collection.get("commoncrawl_crawl_id")
        self.query_model: str = collection.get("query_model", "claude-haiku-4-5-20251001")

        # Labeling
        labeling = self._raw.get("labeling", {})
        self.labeling_model: str = labeling.get("model", "claude-haiku-4-5-20251001")
        self.stage1_max_tokens: int = labeling.get("stage1_max_tokens", 150)
        self.stage2_max_tokens: int = labeling.get("stage2_max_tokens", 300)
        self.tier1_confidence_threshold: float = labeling.get("tier1_confidence_threshold", 0.3)
        self.labeling_temperature: float = labeling.get("temperature", 0.0)
        self.text_truncation_words: int = labeling.get("text_truncation_words", 3000)
        self.excluded_tier1_categories: list = self._raw.get("excluded_tier1_categories", [])

        # Collection hints
        self.domain_hints: dict = self._raw.get("domain_hints", {})
        self.excluded_categories: list = self._raw.get("excluded_categories", [])

    def __repr__(self) -> str:
        return f"TaxonomyConfig(slug={self.slug!r}, name={self.name!r})"


def load_taxonomy_config(slug: str, taxonomies_dir: Optional[Path] = None) -> TaxonomyConfig:
    """Load taxonomy config by slug name."""
    if taxonomies_dir is None:
        taxonomies_dir = Path(__file__).parent.parent.parent.parent / "taxonomies"

    # Try slug as-is, then with hyphens→underscores (iab-2.2 → iab_2.2)
    config_path = taxonomies_dir / slug / "config.yaml"
    if not config_path.exists():
        config_path = taxonomies_dir / slug.replace("-", "_") / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Taxonomy config not found: {taxonomies_dir / slug}/config.yaml")

    return TaxonomyConfig(config_path)


def get_data_dir(override: Optional[str] = None) -> Path:
    """Get data directory from override, env var, or default."""
    if override:
        return Path(override)
    env_dir = os.getenv("CLASSIVORE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("data")


def get_models_dir(override: Optional[str] = None) -> Path:
    """Get models directory from override, env var, or default."""
    if override:
        return Path(override)
    env_dir = os.getenv("CLASSIVORE_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("models")
