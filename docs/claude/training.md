# Training Subsystem

## Modules

- `src/classivore/training/trainer.py` — DeBERTa fine-tuning. Multi-label with focal loss, class weighting, early stopping.
- `src/classivore/training/thresholds.py` — Per-category threshold optimization on validation set.

## Training Configuration

All hyperparameters come from taxonomy `config.yaml`:

```yaml
model_base: "microsoft/deberta-v3-large"
batch_size: 8
learning_rate: 2e-5
max_length: 512
num_epochs: 3
focal_loss:
  alpha: 0.75
  gamma: 3.5
class_weight_cap: 7.0
```

## Training Process

1. Load reviewed data, filter rejected pages
2. Encode labels with MultiLabelBinarizer (n_samples × n_categories)
3. Split 70/20/10 train/val/test
4. Fine-tune DeBERTa with focal loss + class weights
5. Early stopping on validation F1
6. Save model + tokenizer + label mappings

## Per-Category Thresholds

After training, optimize sigmoid threshold independently for each category:
- Default: 0.5 global threshold
- Optimized: per-category thresholds tuned on validation set
- Typical improvement: +10-15% F1 Macro

## Model Artifacts

Saved to `models/<taxonomy-slug>/`:
- Model weights + tokenizer (HuggingFace format)
- `label2id.json`, `id2label.json`
- `per_category_thresholds.json`
- `taxonomy_enriched.csv` (bundled for inference)
- `training_metrics.json`

## Device Auto-Detection

CUDA GPU > Apple Silicon (MPS) > CPU. Configured automatically.

## Tests

- `tests/unit/test_trainer.py` — test data preparation, label encoding, loss computation (small mock data)
- `tests/unit/test_thresholds.py` — test threshold optimization logic
