# Publishing Subsystem

## Overview

The publishing module pushes trained models to HuggingFace Hub as private repos with version tags. The companion `classivore-api` service consumes these artifacts via `snapshot_download(revision=tag)`.

## Artifact Contract

### Required (fail if missing)
- `config.json` — HuggingFace model config
- `model.safetensors` — model weights
- `label_mappings.json` — category name/ID/index mappings
- `per_category_thresholds.json` — optimized classification thresholds
- `training_report.json` — training metadata, metrics, hyperparameters

### Optional (warn if missing)
- `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`, `spm.model` — tokenizer files
- `quality_report.json` — detailed evaluation metrics

### Excluded (never uploaded)
- `*.npy` — raw probability matrices (large, analysis-only)
- `class_weights.json`, `confusion_pairs.json`, `per_category_metrics.json`, `threshold_sweep.json`, `evaluation_report.json` — training analysis artifacts
- `checkpoints/*` — intermediate training checkpoints

## Repo Naming Convention

`{org}/{taxonomy-slug}-{architecture}`

Examples:
- `classivore/iab22-deberta-large`
- `classivore/iptc-deberta-base`

## Version Tagging

Versions use semver with `v` prefix: `v1.0.0`, `v1.1.0`, etc. Each version maps to a HuggingFace git tag, enabling `snapshot_download(revision="v1.0.0")`.

## End-to-End Workflow

```bash
# 1. Train model
classivore train --taxonomy iab-2.2

# 2. Create HF repo (once)
classivore hf init --repo-id classivore/iab22-deberta-large

# 3. Preview what would be uploaded
classivore publish \
  --model-path models/iab-2.2/20260406_193556 \
  --repo-id classivore/iab22-deberta-large \
  --version v1.0.0 \
  --dry-run

# 4. Publish
classivore publish \
  --model-path models/iab-2.2/20260406_193556 \
  --repo-id classivore/iab22-deberta-large \
  --version v1.0.0
```

## How classivore-api Consumes Artifacts

The API server uses `huggingface_hub.snapshot_download()` with a specific revision tag:

```python
from huggingface_hub import snapshot_download
model_path = snapshot_download("classivore/iab22-deberta-large", revision="v1.0.0")
```

This downloads the exact artifact set defined by that version tag. The API loads `label_mappings.json` and `per_category_thresholds.json` alongside the model for inference.

## CLI Commands

### `classivore publish`
Upload a model directory and tag with a version.

### `classivore hf init`
Create a HuggingFace repo (idempotent — safe to run multiple times).

## Authentication

Set `HUGGINGFACE_TOKEN` in `.env` or pass `--token` to CLI commands. The token needs write permission to the target organization/repo.
