#!/usr/bin/env python3
"""Artifact validation for model publishing."""

import json
from pathlib import Path

from classivore.logging_config import get_logger

logger = get_logger(__name__)

# Files required for inference — fail if any missing
REQUIRED = {
    "config.json",
    "model.safetensors",
    "label_mappings.json",
    "per_category_thresholds.json",
    "training_report.json",
    "taxonomy_metadata.json",
}

# Files included if present — warn if missing, don't fail
OPTIONAL = {
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "quality_report.json",
}


def validate_artifacts(model_path):
    """Validate model directory contains required artifacts.

    Args:
        model_path: Path to trained model directory.

    Returns:
        Dict of filename → Path for all files to publish.

    Raises:
        ValueError: If any required files are missing.
    """
    model_path = Path(model_path)
    files = {}

    # Check required files
    missing = []
    for name in REQUIRED:
        path = model_path / name
        if path.exists():
            files[name] = path
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Missing required artifacts: {', '.join(sorted(missing))}"
        )

    # Check optional files
    for name in OPTIONAL:
        path = model_path / name
        if path.exists():
            files[name] = path
        else:
            logger.warning("missing_optional_artifact", file=name)

    return files


def load_training_report(model_path):
    """Read training_report.json from model directory.

    Args:
        model_path: Path to trained model directory.

    Returns:
        Parsed training report dict.
    """
    with open(Path(model_path) / "training_report.json") as f:
        return json.load(f)
