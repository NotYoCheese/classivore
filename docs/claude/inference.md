# Inference Subsystem

## Modules

- `src/classivore/inference/classifier.py` — Load trained model, apply per-category thresholds, return predictions with confidence scores and descriptions.

## Usage

```python
from classivore.inference.classifier import Classifier

clf = Classifier("models/iab-2.2")
results = clf.predict("Tesla announces new Model Y...")
# [{"id": "22", "name": "Automotive: Green Vehicles",
#   "confidence": 0.94, "description": "Electric, hybrid..."}]

# Batch prediction
results = clf.predict_batch(["text1", "text2", ...])
```

## Loading

The Classifier loads from a model directory containing:
- Model weights + tokenizer
- `per_category_thresholds.json` (falls back to 0.5 global if missing)
- `taxonomy_enriched.csv` (for returning descriptions with predictions)
- `label2id.json` / `id2label.json`

## Prediction Pipeline

1. Tokenize input text (max_length from model config)
2. Forward pass through DeBERTa
3. Sigmoid activation on logits (leaf nodes only)
4. Apply per-category thresholds
5. Filter by min_confidence (from taxonomy config)
6. Sort by confidence descending
7. Limit to max_labels (from taxonomy config)
8. For each predicted leaf, walk up taxonomy tree to build full ancestor path
9. Attach category description from enriched taxonomy

Note: The model only predicts leaf nodes. The full path (including parent/ancestor
nodes) is derived from the taxonomy hierarchy in post-processing. Multiple
predictions can come from completely different branches of the taxonomy.

## Performance

- CPU: 50-100 texts/second (sufficient for API serving)
- MPS: 100-200 texts/second
- CUDA: 200-500 texts/second
- Model size: ~1.7GB (DeBERTa-v3-large)

## Tests

- `tests/unit/test_classifier.py` — test prediction pipeline, threshold application, description attachment
