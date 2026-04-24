# Labeling Subsystem

## Modules

- `src/classivore/labeling/__init__.py` — Two-stage orchestrator (`run_labeling`). Submits Anthropic Batch API jobs, polls for completion, and writes final labels.
- `src/classivore/labeling/prompts.py` — Prompt builders for stage 1 (tier-1 triage) and stage 2 (subtree classification). Includes inline category descriptions + boundaries from the enriched taxonomy.
- `src/classivore/labeling/parser.py` — Parses batch responses, validates categories against the taxonomy, filters by confidence.
- `src/classivore/labeling/state.py` — `LabelState` persistence: tracks per-page stage progress, supports crash recovery.

## Two-Stage Hierarchical Labeling

Content is labeled in two Anthropic Batch API passes:

1. **Stage 1 — Tier-1 triage.** A single prompt per page lists all tier-1 categories; the LLM picks the relevant top-level subtrees (and/or the "no content category applies" option for pages excluded at tier-1, like generic homepages).
2. **Stage 2 — Subtree classification.** For each selected tier-1 from stage 1, a scoped prompt offers only that subtree's categories and asks for specific leaf labels.

This keeps the per-call category list small enough for reliable LLM selection even on 600+ category taxonomies, and lets the pipeline skip irrelevant subtrees rather than paying the cost of evaluating them all per page.

## Prompt Structure

Each prompt (stage 1 and stage 2) includes:
1. Task description and output format
2. The relevant slice of the taxonomy, with enriched descriptions and boundaries inline per category
3. The content to classify (text + URL + title)
4. Instructions for chain-of-thought reasoning

## LLM Response Schema (per stage)

```json
{
    "reasoning": "2-3 sentence analysis...",
    "categories": [
        {"category": "Automotive: Green Vehicles", "confidence": 0.95}
    ]
}
```

## Provider

Uses the Anthropic Messages Batches API directly (50% cost reduction vs. standard API). Configure via:

- `ANTHROPIC_API_KEY` — read by the `anthropic` SDK
- `labeling.model` in `taxonomies/<slug>/config.yaml` — model override (defaults to Haiku)

## Batch & Crash Recovery

- Batches are submitted and polled; each page is tracked in `LabelState` with its stage (stage1/stage2) status and batch ID.
- On restart, in-flight batches are resumed where possible; completed pages are skipped.
- Raw batch responses are archived as `stage{1,2}_raw_<batch_id>.jsonl` for debugging.

## Output

Final labels are written as NDJSON to `data/labels/<taxonomy-slug>/labels.json`, one page per line:

```json
{"url": "...", "content_hash": "...", "categories": ["Category Name 1", "Category Name 2"]}
```

Only stage-2-complete pages are emitted.

## Tests

- `tests/unit/test_labeling_orchestrator.py` — end-to-end `run_labeling` flow (mocked Anthropic client)
- `tests/unit/test_labeling_parser.py` — response parsing, category validation, confidence filtering
- `tests/unit/test_labeling_prompts.py` — stage 1 + stage 2 prompt construction
- `tests/unit/test_labeling_state.py` — state transitions, crash recovery, persistence
