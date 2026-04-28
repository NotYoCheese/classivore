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
3. Split 70/20/10 train/val/test with multi-label iterative stratification (Sechidis et al. 2011) via `iterative-stratification` — keeps per-label representation proportional across all three splits so per-category F1 stays meaningful for tail categories
4. Fine-tune DeBERTa with focal loss + class weights. Training sets `problem_type="multi_label_classification"` in the saved `config.json`; `Classifier` reads this at load time to choose the output activation.
5. Early stopping on validation F1
6. Save model + tokenizer + artifacts (see below)

## Per-Category Thresholds

After training, optimize sigmoid threshold independently for each category:
- Classifier applies a per-category threshold vector at inference time; the default floor is `0.5` for multi-label models and `0.0` for single-label models, and entries in `per_category_thresholds.json` override the default whenever present.
- Optimized: per-category thresholds tuned on validation set
- Typical improvement: +10-15% F1 Macro

## Model Artifacts

Saved to `models/<taxonomy-slug>/<timestamp>/`:
- HuggingFace model directory (weights, `config.json`, tokenizer files) via `Trainer.save_model` + `tokenizer.save_pretrained`
- `label_mappings.json` — `index_to_name` + `index_to_id`
- `taxonomy_metadata.json` — per-category `path`, `depth`, `is_leaf`, `parent_name` (required by `Classifier` and the publishing artifact contract)
- `per_category_thresholds.json` — optimized per-category thresholds
- `training_report.json` — hyperparameters + metrics + artifact inventory
- `quality_report.json` — detailed per-category evaluation metrics

## Device Auto-Detection

CUDA GPU > Apple Silicon (MPS) > CPU. Configured automatically.

## Tests

- `tests/unit/test_trainer.py` — test data preparation, label encoding, loss computation (small mock data)
- `tests/unit/test_thresholds.py` — test threshold optimization logic
