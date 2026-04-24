# Inference Subsystem

## Design Principle

The Classifier is self-contained. Everything needed for text → structured predictions lives in the model directory. No taxonomy CSV, no config.yaml, no classivore internal imports. classivore-api does `snapshot_download(revision="v1.0.0")` and serves from that directory.

## Module

`src/classivore/inference/classifier.py` — standalone, depends only on torch, transformers, numpy, json.

## Usage

```python
from classivore.inference import Classifier

clf = Classifier("models/iab-2.2/20260406_193556")
results = clf.predict("Tesla announces new Model Y with improved range...")
# [{"name": "Electric Vehicles", "id": "42", "path": ["Automotive", ...], "confidence": 0.93}]

# Batch prediction
all_results = clf.predict_batch(["text1", "text2", ...])

# Long documents are automatically chunked with sliding window
```

## Model Directory Artifacts

Required for inference:
- `model.safetensors` + `config.json` — model weights
- `tokenizer.json` + `tokenizer_config.json` + `spm.model` — tokenizer
- `label_mappings.json` — index ↔ category name/ID mappings
- `per_category_thresholds.json` — optimized per-category confidence thresholds
- `taxonomy_metadata.json` — category paths, depth, is_leaf (generated at training time)

## Prediction Pipeline

1. Tokenize input text
2. If tokens exceed the model's usable input length: sliding window chunking with 128-token overlap
3. Batched forward pass through the model
4. Activation dispatched from `config.json`'s `problem_type`:
   - `multi_label_classification` → sigmoid (independent per-class probs). Default when field is missing.
   - `single_label_classification` → softmax (probs sum to 1 per row)
   - `regression` → raises `NotImplementedError`
5. If chunked: aggregate probabilities across chunks (max or mean)
6. Apply per-category thresholds (vectorized). Default floor is `0.5` for multi-label and `0.0` for single-label; `per_category_thresholds.json` always overrides when present.
7. Build result dicts with name, id, path, confidence
8. Sort by confidence descending

### Usable input length

`max_length` is derived from `config.max_position_embeddings`, clamped against `tokenizer.model_max_length` when it carries a real value. RoBERTa-family models (`roberta`, `xlm-roberta`, `camembert`, `longformer`, `xmod`) offset `position_ids` by `pad_token_id + 1`, so a config of `514` only permits `512` input tokens — `Classifier` detects this from `config.model_type` and subtracts the offset. BERT, DeBERTa, and other non-RoBERTa-family models use the full configured length.

## Device Selection

Auto-detects: CUDA > MPS > CPU. FP16 only on CUDA. Inlined in classifier.py (no training module dependency).

## CLI

```
classivore classify --text "Article about sedans..."
classivore classify --file input.json --output results.json
classivore classify --interactive
classivore classify --model-dir models/iab-2.2/20260406_193556 --text "..."
```

Auto-discovers most recent model in `models/{slug}/` if `--model-dir` not specified.

## Tests

- `tests/unit/test_cli.py` — CLI classify routing with mocked Classifier
