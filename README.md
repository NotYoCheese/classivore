# Classivore

Open-source, taxonomy-agnostic text classification pipeline. Give it any hierarchical taxonomy CSV and it builds a production multi-label classifier — from data collection through training to inference.

## What It Does

Classivore automates the full pipeline for building a text classifier on a custom taxonomy:

1. **Enrich** your taxonomy with LLM-generated descriptions, boundaries, aliases, and difficulty ratings
2. **Collect** training data from the web using search APIs (Brave, Serper) and Common Crawl
3. **Label** collected pages using hierarchical LLM classification via the Anthropic Batch API
4. **Train** a DeBERTa-v3 classifier with focal loss, per-category thresholds, and quality reporting
5. **Publish** the trained model to HuggingFace Hub for serving

The entire pipeline is driven from the command line. Each stage is resumable — interrupt and restart without losing progress.

## Quick Start

```bash
# Install
git clone https://github.com/NotYoCheese/classivore.git
cd classivore
python -m venv venv && source venv/bin/activate
pip install -e .

# Initialize a new taxonomy
classivore init --csv your_taxonomy.csv --name "My Taxonomy" --version "1.0" --slug my-tax

# Or use an existing taxonomy and run the pipeline step by step
classivore enrich --taxonomy my-tax
classivore collect --taxonomy my-tax
classivore label --taxonomy my-tax
classivore train --taxonomy my-tax

# Run inference
classivore classify --text "Article about machine learning trends..."
classivore classify --file articles.json --output predictions.json
classivore classify --interactive
```

## Pipeline Stages

### `classivore init`

Onboard a new taxonomy from a CSV file. Validates the CSV structure, generates a `config.yaml` with sensible defaults, optionally runs LLM enrichment and domain hint generation, and prints an onboarding report with collection cost estimates.

### `classivore enrich`

Generate descriptions, boundaries, aliases, and difficulty ratings for each taxonomy category using the Anthropic Batch API. These fields improve search query quality and help the labeling stage make better decisions.

### `classivore collect`

Discover and scrape web pages for training data. Uses search APIs (Brave, Serper) with automatic fallback, Common Crawl CDX for historical pages, and content quality filters. Collection targets are tiered by category difficulty — hard categories get more pages to compensate for scarce editorial content.

### `classivore label`

Classify collected pages using a two-stage hierarchical LLM approach:
- **Stage 1**: Tier-1 triage identifies which top-level categories apply (cheap, broad pass)
- **Stage 2**: Subtree classification within selected tier-1s (detailed, with chain-of-thought)

Uses the Anthropic Batch API for 50% cost reduction. Crash-recoverable — pages at each stage are checkpointed.

### `classivore train`

Fine-tune DeBERTa-v3-large for multi-label classification. Features:
- Weighted focal loss for extreme class imbalance
- Confidence-weighted training (legacy labels discounted)
- Per-category threshold optimization (+5% F1 macro over global threshold)
- Comprehensive quality report with per-category metrics, confusion pairs, and overfitting detection

### `classivore classify`

Run inference using a trained model. Supports single text, batch JSON/NDJSON, and interactive mode. Long documents are automatically chunked with a sliding window. Auto-discovers the most recent trained model.

### `classivore agent`

Automated collect-label-evaluate loop. Analyzes coverage gaps, collects pages for the weakest categories, labels them, and repeats until targets are met or budget is exhausted.

### `classivore publish`

Push a trained model to a private HuggingFace Hub repo with version tagging. The published artifact is self-contained — includes model weights, tokenizer, thresholds, label mappings, and taxonomy metadata (paths, IDs). No taxonomy CSV needed at serve time.

## Other Commands

| Command | Description |
|---------|-------------|
| `classivore taxonomy` | Show taxonomy stats, coverage gaps, and exclusions |
| `classivore validate` | Run data quality checks via label-lens |
| `classivore hints` | Generate domain hints for tier-1 categories |
| `classivore hf init` | Create a private HuggingFace repo |

## Taxonomy Configuration

Each taxonomy lives in `taxonomies/<slug>/` with a `config.yaml` that controls everything: collection targets by difficulty, query budgets, LLM models, filter relaxations, domain hints, and category exclusions. See `taxonomies/` for examples.

## Self-Contained Inference

The `Classifier` class at `classivore.inference.Classifier` is designed for production use. It has zero classivore internal dependencies — only torch, transformers, numpy, and json. Load a model directory and get predictions:

```python
from classivore.inference import Classifier

classifier = Classifier("models/my-tax/20260408_162922")
results = classifier.predict("Article text here...")
# [{"name": "Category", "id": "42", "path": ["Parent", "Category"], "confidence": 0.93}]
```

The companion [classivore-api](https://github.com/NotYoCheese/classivore-api) repo uses this for serving.

## Project Structure

```
src/classivore/
  cli/            Command-line interface
  config/         Settings loader and defaults
  taxonomy/       CSV loader, enricher, onboarding
  collection/     Search, scraping, filters, state
  labeling/       Two-stage LLM labeling pipeline
  training/       DeBERTa trainer, focal loss, thresholds, evaluation
  inference/      Self-contained Classifier for production inference
  publishing/     HuggingFace Hub publishing
  agent/          Automated collect-label-evaluate loop
  validation/     Data quality checks
```

## Requirements

- Python >= 3.11
- Anthropic API key (for enrichment and labeling)
- Search API key: Brave and/or Serper (for collection)
- GPU recommended for training (RTX 4090: ~45 min for 30K pages)

## License

MIT
