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
2. If tokens exceed `max_position_embeddings` (512): sliding window chunking with 128-token overlap
3. Batched forward pass through DeBERTa
4. Sigmoid activation on logits
5. If chunked: aggregate probabilities across chunks (max or mean)
6. Apply per-category thresholds (vectorized)
7. Build result dicts with name, id, path, confidence
8. Sort by confidence descending

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
