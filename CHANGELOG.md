# Changelog

All notable changes to this project will be documented in this file.

## [1.2.2] - 2026-04-24

### Inference

- Fix `IndexError: index out of range in self` when classifying long input
  with RoBERTa-family models (RoBERTa, XLM-RoBERTa, CamemBERT, Longformer,
  XMOD). These models offset `position_ids` by `pad_token_id + 1`, so a
  config of `max_position_embeddings=514` only addresses 512 input tokens.
  `Classifier` now detects the offset from `config.model_type` and clamps
  `max_length` accordingly. BERT, DeBERTa, and other non-RoBERTa-family
  models are unaffected.

## [1.2.1] - 2026-04-21

### Inference

- Force eager weight loading (`low_cpu_mem_usage=False`) in `Classifier` to
  avoid a meta-tensor race under thread-pool concurrency. Previously, multiple
  threads loading the same model could leave some parameters on the meta
  device, causing `.to(device)` to raise
  `NotImplementedError: Cannot copy out of meta tensor; no data!` — observed
  in the classivore-api container on linux/amd64. No accuracy change.

## [1.2.0] - 2026-04-21

### Inference

- `Classifier` now dispatches the output activation based on the model's
  `problem_type` in `config.json`:
  - `multi_label_classification` → sigmoid (existing behavior)
  - `single_label_classification` → softmax (probabilities sum to 1 per row)
  - `regression` → raises `NotImplementedError`
  - missing / null → defaults to multi-label (preserves legacy behavior)
- Default per-category threshold floor is now `0.0` for single-label models
  (so softmax results aren't silently suppressed by the multi-label `0.5` default).
  Multi-label models still default to `0.5`. Per-category threshold files
  always take precedence when present.
- Allows dropping in off-the-shelf HuggingFace single-label classifiers
  (e.g. sentiment, toxicity) alongside classivore-trained multi-label models.

### Collection

- Added Exa AI as a third search provider with two roles:
  - Neural/semantic search fallback when Brave and Serper are exhausted
  - Scrape fallback via Exa `/contents` when live scraping is blocked (WAFs, etc.)
- Exa search results include prefetched page text, so the collector skips the
  scrape step entirely for those URLs.

### Docs

- Expanded README with a full "API Keys & External Services" section
  documenting every external service (Anthropic, Brave, Serper, Exa,
  HuggingFace, Common Crawl), which commands use each one, where to get keys,
  and which features are enabled by each.

## [1.0.0] - 2026-04-06

First release. Full pipeline from taxonomy enrichment to trained classifier.

### Pipeline

- **Enrich** — LLM-generated descriptions and boundaries for taxonomy categories
  via Anthropic Batch API
- **Collect** — Web scraping with Brave/Serper search, Common Crawl CDX lookup,
  content filtering, domain quality tracking, circuit breaker
- **Label** — Two-stage hierarchical LLM labeling (tier-1 triage + subtree
  classification) via Anthropic Batch API with crash recovery
- **Agent** — Automated collect-label-evaluate loop with gap prioritization,
  per-category targets, and stop conditions
- **Validate** — Data quality checks via label-lens integration
- **Train** — DeBERTa-v3-large fine-tuning with weighted focal loss,
  confidence-weighted training, per-category threshold optimization, and
  comprehensive quality reporting

### Infrastructure

- Structured logging via structlog with JSON output
- Atomic JSON persistence for all state files
- Typed exception hierarchy
- Shared data models (dataclasses)
- 460+ unit tests

### Training Results (IAB 2.2)

- 27,102 training pages across 605 leaf categories
- F1 micro: 0.68 (val), 0.65 (test)
- F1 macro: 0.65 (val), 0.62 (test)
- Per-category threshold optimization: +5% F1 macro
- Training time: 43 minutes on RTX 4090

### Key Technical Decisions

- Focal loss alpha=0.75, gamma=3.5 (validated over 25+ training iterations)
- Class weight cap at 7.0 without normalization
- Legacy label confidence discount (0.75 weight)
- Non-leaf labels excluded from training
- One target, one source of truth (labels.json) for the agent loop
