#!/usr/bin/env python3
"""HuggingFace Hub operations for model publishing."""

import json
import re
from pathlib import Path

from huggingface_hub import (
    create_repo,
    create_tag,
    upload_folder,
)
from huggingface_hub.utils import HfHubHTTPError

from classivore.logging_config import get_logger
from classivore.publishing.artifact import load_training_report, validate_artifacts
from classivore.publishing.card import generate_model_card

logger = get_logger(__name__)

VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")

# Files excluded from upload — training artifacts and large binary analysis files
IGNORE_PATTERNS = [
    "*.npy",
    "class_weights.json",
    "confusion_pairs.json",
    "per_category_metrics.json",
    "threshold_sweep.json",
    "evaluation_report.json",
    "checkpoints/*",
]


def init_repo(repo_id, token, private=True):
    """Create a HuggingFace repo if it doesn't exist.

    Args:
        repo_id: HuggingFace repo ID (e.g. "classivore/iab22-deberta-large").
        token: HuggingFace API token.
        private: Whether repo should be private (default True).

    Raises:
        HfHubHTTPError: If token is invalid or lacks write permission.
    """
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        logger.info("repo_ready", repo_id=repo_id, private=private)
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            raise HfHubHTTPError(
                f"Authentication failed for {repo_id}. "
                "Check that your token is valid and has write permission."
            ) from e
        raise


def publish_model(model_path, repo_id, version, token, dry_run=False):
    """Publish a trained model to HuggingFace Hub.

    Args:
        model_path: Path to trained model directory.
        repo_id: HuggingFace repo ID.
        version: Semver tag (e.g. "v1.0.0").
        token: HuggingFace API token.
        dry_run: If True, print what would be uploaded without uploading.

    Returns:
        Commit URL string on success, None if dry_run.

    Raises:
        ValueError: If version format is invalid or required artifacts missing.
    """
    model_path = Path(model_path)

    # 1. Validate version format
    if not VERSION_PATTERN.match(version):
        raise ValueError(
            f"Invalid version '{version}'. Must match v<major>.<minor>.<patch> "
            "(e.g. v1.0.0)"
        )

    # 2. Validate artifacts
    files = validate_artifacts(model_path)
    logger.info("artifacts_validated", count=len(files))

    # 3. Load reports
    training_report = load_training_report(model_path)
    quality_report = None
    quality_path = model_path / "quality_report.json"
    if quality_path.exists():
        with open(quality_path) as f:
            quality_report = json.load(f)

    # 4. Generate model card
    card = generate_model_card(training_report, quality_report, repo_id, version)

    # 5. Write temporary README.md
    readme_path = model_path / "README.md"
    readme_path.write_text(card)
    logger.info("model_card_generated", path=str(readme_path))

    if dry_run:
        print(f"\nDry run — would publish to {repo_id} as {version}\n")
        print("Files to upload:")
        for name in sorted(files):
            print(f"  {name}")
        print(f"  README.md (generated)")
        print(f"\nExcluded patterns: {IGNORE_PATTERNS}")
        # Clean up README
        readme_path.unlink(missing_ok=True)
        return None

    try:
        # 6. Create repo if needed
        init_repo(repo_id, token, private=True)

        # 7. Upload
        logger.info("uploading", repo_id=repo_id, version=version)
        commit_info = upload_folder(
            repo_id=repo_id,
            folder_path=str(model_path),
            ignore_patterns=IGNORE_PATTERNS,
            token=token,
            commit_message=f"Add model version {version}",
        )

        commit_url = commit_info.commit_url
        logger.info("upload_complete", commit_url=commit_url)

        # 8. Tag — retry-safe; if HF 500s, the upload is already done
        try:
            create_tag(repo_id, tag=version, token=token, repo_type="model")
            logger.info("version_tagged", repo_id=repo_id, version=version)
        except HfHubHTTPError as e:
            logger.warning(
                "tagging_failed", repo_id=repo_id, version=version,
                error=str(e),
                hint="Upload succeeded. Retry tagging with: "
                     f"hf repo tag {repo_id} {version}",
            )
            print(
                f"\nWarning: Upload succeeded but tagging failed ({e}).\n"
                f"Retry manually: hf repo tag {repo_id} {version}"
            )

        return commit_url

    finally:
        # 9. Clean up temporary README.md
        readme_path.unlink(missing_ok=True)
