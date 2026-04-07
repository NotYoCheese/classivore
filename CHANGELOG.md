# Changelog

All notable changes to this project will be documented in this file.

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
