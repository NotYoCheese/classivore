# Labeling Subsystem

## Modules

- `src/classivore/labeling/labeler.py` — Multi-provider LLM labeler. Supports Anthropic API, AWS Bedrock. Handles checkpointing, resume, batch processing.
- `src/classivore/labeling/prompt_builder.py` — Builds taxonomy-aware prompts. Injects all categories with enriched descriptions. Handles few-shot examples.

## Prompt Structure

The prompt includes:
1. Task description and output format
2. Full taxonomy with descriptions (from enriched taxonomy)
3. Few-shot examples (loaded from existing labeled data or curated file)
4. The content to classify (text + URL + title)
5. Instructions for chain-of-thought reasoning

Key difference from iab_forge: descriptions are included inline with each category in the taxonomy section. This eliminates ambiguity for the LLM.

## LLM Response Schema

```json
{
    "reasoning": "2-3 sentence analysis...",
    "categories": [
        {"category": "Automotive: Green Vehicles", "category_id": "22", "confidence": 0.95}
    ]
}
```

## Provider Support

Configured via environment variables:
- `CLASSIVORE_API_KEY` or `ANTHROPIC_API_KEY` — Anthropic direct
- `AWS_REGION` + AWS credentials — Bedrock
- Provider selection: `--provider anthropic|bedrock` CLI flag or config.yaml

## Checkpointing

- Saves every 10 labeled pages to output file
- On restart, loads existing labeled pages and skips by URL
- Stores `raw_llm_response` for debugging

## Few-Shot Examples

- Auto-loaded from existing labeled data (highest confidence, diverse categories)
- Can be overridden with curated file: `data/few_shot_examples.json`
- 3-5 examples included in prompt

## Tests

- `tests/unit/test_prompt_builder.py` — test prompt generation with various taxonomy configs
- `tests/unit/test_labeler.py` — test response parsing, validation, checkpoint/resume (mock LLM)
