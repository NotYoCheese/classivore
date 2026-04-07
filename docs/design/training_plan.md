# Training Module Implementation Plan

## Lessons from Previous Work

These findings come from 25+ training iterations on the same taxonomy (IAB 2.2,
698 categories) that produced a model achieving 0.6556 F1 micro. The lessons
are taxonomy-agnostic and should inform the classivore implementation.

### What Matters Most (in order)

1. **Data quality > data quantity.** Targeted minority-class examples improved
   F1 by 147x. Beyond ~15K pages, adding random data shows diminishing or
   negative returns. The agent's gap-prioritization strategy already addresses this.

2. **Class weight normalization destroys class weights.** Normalizing weights
   after capping (`weights / weights.mean()`) reduces all weights to ~1.0,
   completely negating the purpose. This single bug fix improved F1 by 82%.
   **Never normalize weights after capping.**

3. **Focal loss breaks through the class-weight ceiling.** Class weights alone
   plateau around 0.42 F1. Adding focal loss (alpha=0.75, gamma=3.5) pushed
   to 0.51 — a 21% improvement. The two techniques compound.

4. **Model size matters more than data size at scale.** DeBERTa-v3-large
   (434M params) improved F1 by 28.5% over base (184M) on the same data.
   Adding 27.6% more data to the base model improved F1 by only 0.18%.

5. **Per-category thresholds are free performance.** Optimizing a threshold
   per category (instead of one global threshold) improved F1 macro by 12.76%
   with zero inference latency cost.

6. **Optimal threshold shifts with architecture.** Base model optimal: 0.50.
   Large model optimal: 0.55. Always re-run threshold optimization after
   any training change.

### Parameters to Carry Forward

| Parameter | Value | Notes |
|-----------|-------|-------|
| Focal alpha | 0.75 | Higher than paper default (0.25). Tested 20 combinations. |
| Focal gamma | 3.5 | Aggressive hard-example focus. Optimal for 698-class imbalance. |
| Class weight cap | 7.0 | Without normalization. |
| Learning rate | 2e-5 | Standard fine-tuning rate. |
| Batch size | 8 | For large model on 24GB GPU. |
| Epochs | 3 | With early stopping, patience 2. |
| Max length | 512 | Tokens (~1,800 words). |
| Global threshold | 0.55 | Starting point; per-category optimization on top. |
| Train/val/test split | 70/20/10 | Stratified by category. |

### What to Avoid

- Weight normalization after capping
- Random data collection without targeting weak categories
- Fixed global threshold (always use per-category)
- Class weight cap below 5.0 (too conservative for 698 categories)
- Focal gamma below 3.0 (too weak for this imbalance level)

---

## Architecture

### Module Structure

```
src/classivore/training/
├── __init__.py         # re-exports train_model
├── trainer.py          # Training orchestration
├── dataset.py          # Dataset class, data loading, splits
├── loss.py             # Weighted focal loss
├── thresholds.py       # Per-category threshold optimization
└── evaluate.py         # Metrics, evaluation, reporting
```

### Data Flow

```
label_state.json + corpus/pages.json
        ↓
    dataset.py: join text + labels (with confidence), split train/val/test
        ↓
    trainer.py: fine-tune DeBERTa with focal loss
        ↓
    thresholds.py: optimize per-category thresholds on val set
        ↓
    evaluate.py: final metrics on test set
        ↓
    models/{slug}/{timestamp}/
        ├── model.safetensors
        ├── config.json
        ├── tokenizer files
        ├── label_mappings.json
        ├── per_category_thresholds.json
        └── training_report.json
```

### Key Design Decisions

**1. Confidence-weighted training**

We now have real confidence scores on every label. Use them:
- Labels with confidence >= 0.8: full weight
- Labels with confidence 0.5-0.8: reduced weight (scale linearly)
- Labels below min_confidence: excluded (already filtered at labeling time)

This is a soft-label approach — the model learns to be uncertain where the
labeler was uncertain.

**2. Text preparation**

Concatenate available fields: `title + " " + text`. Truncate to max_length
tokens. No chunking at training time — the model learns from the first 512
tokens, which covers ~1,800 words (median article length in our corpus).

Chunking is an inference-time concern, not training-time.

**3. Category ID ordering**

Sort categories by integer ID, not string. Previous implementation had a bug
where category "10" sorted after "1" but before "2" in string sort, causing
label index mismatches. Use `sorted(categories, key=lambda c: int(c["id"]))`.

**4. Label mappings**

Save a `label_mappings.json` alongside the model:
```json
{
  "id_to_index": {"1": 0, "2": 1, ...},
  "index_to_name": {"0": "Automotive", "1": "Sedan", ...},
  "index_to_id": {"0": "1", "1": "2", ...}
}
```

This is the contract between training and inference. The model outputs
698 logits; this file maps them to category names.

**5. Device selection**

Auto-detect: CUDA > MPS (Apple Silicon) > CPU. Use FP16 on CUDA only
(MPS doesn't support FP16 training reliably). Report device at startup
so the user knows what they're training on.

**6. Output directory**

```
models/{taxonomy_slug}/{timestamp}/
```

Each training run gets its own timestamped directory. No overwriting.
The `training_report.json` captures all hyperparameters, metrics, data
stats, and timing so runs are reproducible and comparable.

---

## Module Details

### `dataset.py`

**Responsibilities:**
- Load corpus pages (NDJSON) and labels (from label_state.json for confidence)
- Join by content_hash
- Build label matrix (num_samples × num_categories) with confidence values
- Stratified train/val/test split
- HuggingFace Dataset class with tokenization

**Data loading — two files, one join:**

The authoritative source for training data is `label_state.json`, not
`labels.json`. label_state.json has per-label confidence scores and the
`reasoning` field needed to identify legacy vs current-pipeline labels.
labels.json is a simplified output for collection seeding and does not
contain confidence.

- **Text source:** `data/corpus/pages.json` (NDJSON, keyed by content_hash)
- **Label source:** `data/labels/{slug}/label_state.json` (has labels with
  confidence, reasoning, and pipeline provenance)
- **Join:** on content_hash. Pages in corpus but not in label_state are
  unlabeled (skipped). Pages in label_state but not in corpus have no text
  (skipped with warning).
- **Filter:** Only pages with `status == "stage2_complete"` and at least one label
- **Exclude:** Non-leaf category labels (see below)

**Split strategy:**
- Stratified by most common category per page (for multi-label, use the
  highest-confidence label as the stratification key)
- 70% train, 20% validation, 10% test
- Save split assignments so they're reproducible

**Non-leaf labels: excluded from training.**

Non-leaf labels like "Automotive" (parent of "Sedan", "SUV", etc.) are
excluded from the label matrix entirely. The model only predicts leaf
categories. Non-leaf labels indicate the labeler couldn't decide which
leaf to pick — training on them teaches the model to be vague. At
inference time, callers can aggregate leaf predictions up to parents
if needed.

### `loss.py`

**Weighted focal loss implementation:**

```python
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.5, class_weights=None):
        ...

    def forward(self, logits, targets, confidence_weights=None):
        # BCE focal loss with optional per-sample confidence weighting
        ...
```

**Class weight computation:**
```python
weight = log(total_samples / (positive_count + 1.0))
weight = min(weight, class_weight_cap)  # Cap at 7.0
# NO NORMALIZATION
```

**Confidence weighting:**
- `confidence_weights` is a (batch, num_classes) tensor
- Multiplied element-wise with the loss
- When `confidence_weights=None`, defaults to all-ones (no weighting)
- Labels with real pipeline confidence (e.g. 0.92) get that value as weight
- Legacy labels (restored from previous project with confidence 1.0) get a
  fixed discount: `LEGACY_CONFIDENCE_WEIGHT = 0.75`. These labels were
  generated by a weaker pipeline with higher non-leaf rates and no
  enriched descriptions — they shouldn't get full trust despite having
  confidence 1.0 in the data. The dataset module applies this discount
  when loading labels that originated from the legacy source.
- Labels from the current pipeline with confidence 1.0 are genuinely
  high-confidence and keep full weight.

**How to distinguish legacy vs current pipeline labels:**
Labels in `label_state.json` where `reasoning == "seeded from existing labels"`
are legacy imports. All others are from the current pipeline. The dataset
module checks this field and applies the legacy discount accordingly.

### `trainer.py`

**Training orchestration:**

```python
def train_model(config, data_dir, output_dir=None, device=None):
    """Train a multi-label classifier.

    Args:
        config: TaxonomyConfig instance.
        data_dir: Path to data directory.
        output_dir: Override output directory.
        device: Override device selection.

    Returns:
        Dict with training results and model path.
    """
```

**Uses HuggingFace Trainer** with custom loss function (passed via
`compute_loss` override or custom Trainer subclass). This gives us:
- Automatic gradient accumulation
- Mixed precision (FP16)
- Logging integration
- Checkpoint saving
- Early stopping

**Training report saved to `training_report.json`:**
```json
{
  "taxonomy": "iab-2.2",
  "timestamp": "2026-04-07T...",
  "model_base": "microsoft/deberta-v3-large",
  "num_categories": 698,
  "num_train": 18000,
  "num_val": 5200,
  "num_test": 2600,
  "hyperparameters": { ... },
  "metrics": {
    "val_f1_micro": 0.65,
    "val_f1_macro": 0.55,
    "test_f1_micro": 0.64,
    "test_f1_macro": 0.54
  },
  "training_time_seconds": 7200,
  "device": "cuda:0 (NVIDIA A10)"
}
```

### `thresholds.py`

**Per-category threshold optimization:**

After training, run threshold optimization on the validation set:
1. Get model predictions (raw logits → sigmoid probabilities) for all val pages
2. For each category, sweep thresholds [0.3, 0.35, ..., 0.70] and find the
   one that maximizes F1 for that category
3. Categories with < 5 val samples: use global optimal threshold
4. Save as `per_category_thresholds.json`

This is a post-training step, not part of the training loop.

### `evaluate.py`

**Final evaluation on held-out test set:**

- Apply per-category thresholds
- Compute: F1 micro, F1 macro, precision, recall
- Per-category breakdown (worst 20, best 20)
- Confusion analysis: most common misclassifications
- Save to `training_report.json`

---

## CLI Integration

```python
def _register_train(subparsers):
    p = subparsers.add_parser("train", help="Train classification model")
    _add_common_args(p)
    p.add_argument("--model-base", default=None,
                   help="Override base model (default: from config)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--device", default=None,
                   help="Force device (cuda, mps, cpu)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show data stats and config without training")
```

`classivore train --taxonomy iab-2.2` runs the full pipeline:
load data → train → optimize thresholds → evaluate → save model.

`classivore train --taxonomy iab-2.2 --dry-run` shows:
- Total pages, train/val/test split sizes
- Number of leaf categories with labels
- Class distribution histogram (buckets by sample count)
- Categories below minimum sample threshold (< 20 training examples) — listed by name
- Estimated training time based on sample count × model size × epochs
  (calibrated constant from previous runs: ~1.2 sec/sample/epoch on A10)
- Legacy vs current-pipeline label counts
- Confidence score distribution summary

---

## Implementation Order

1. **`loss.py`** + tests — Focal loss with class weights and confidence weighting
2. **`dataset.py`** + tests — Data loading, joining, splitting, tokenization
3. **`trainer.py`** + tests — Training loop with HuggingFace Trainer
4. **`thresholds.py`** + tests — Per-category threshold optimization
5. **`evaluate.py`** + tests — Metrics and reporting
6. **CLI wiring** — `_cmd_train` and `_register_train`

Each step is independently testable. Steps 1-2 can be tested on CPU
with tiny data. Steps 3-5 need a GPU for realistic testing but can be
unit-tested with mocks.

---

## Training Environment (Out of Scope)

Per discussion, the infrastructure to provision, configure, and tear down
a GPU training environment is **not part of this project**. The training
module assumes:

- Python 3.13 with PyTorch and transformers installed
- A GPU with >= 24GB VRAM (for large model) or >= 12GB (for base model)
- The classivore package installed (`pip install -e .`)
- Access to the data directory

The training host setup (Hetzner/RunPod/Lambda provisioning, dependency
installation, data transfer) lives outside the classivore repo — either
as a separate mini-project or a .gitignored `infra/` folder.

---

## Cost Estimates

**Training time** (DeBERTa-v3-large, 29K samples, 3 epochs):
- A10 (24GB): ~3-4 hours
- A100 (40GB): ~1-2 hours
- RTX 4090 (24GB): ~2-3 hours

**Compute cost:**
- RunPod A10 @ $0.40/hr × 4 hours = ~$1.60
- Threshold optimization: ~15 min additional
- Total: ~$2 per training run

**Iteration cost** (if re-training after collecting more data):
- Same ~$2 per run
- Expect 3-5 runs to tune and validate
- Total project training budget: ~$10-15

---

## Expected Baseline Performance

Based on previous work with same taxonomy and similar data volume:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| F1 micro | 0.63-0.68 | Global threshold 0.55 |
| F1 macro | 0.53-0.58 | Before per-category thresholds |
| F1 macro (per-cat) | 0.65-0.72 | After per-category thresholds |
| Precision | 0.65-0.70 | |
| Recall | 0.62-0.67 | |

We have more data (29K vs 20K) and better label quality (improved prompts,
lower non-leaf rate). Performance should meet or exceed these ranges.

---

## Closed Questions

1. **Non-leaf label handling** — **Option A: exclude.** Non-leaf labels like
   "Automotive" tell you almost nothing about which leaf category applies.
   Propagating to children (Option C) would introduce massive label noise —
   a page labeled "Automotive" could be about any of 30+ subcategories.
   The model only predicts leaf categories. Callers can aggregate up to
   parents at inference time if needed.

3. **DeBERTa-v3-base vs large** — **Go straight to large.** Previous work
   showed large improved F1 by 28.5% over base, while adding 27% more data
   to base improved F1 by 0.18%. The answer is clear. If GPU VRAM is
   insufficient (< 24GB), fall back to base with a warning.

## Open Questions

2. **Confidence weighting curve** — Should confidence scale linearly or use a
   different curve (e.g. square, threshold at 0.7)? Linear is the simplest
   starting point. Can revisit after first training run if calibration
   analysis suggests a different curve.

4. **Chunking at training time** — Previous work only chunked at inference.
   Should we try training on multiple chunks per document for long articles?
   Deferred — not needed for first training run.

## Out of Scope (Acknowledged)

**Inference module** — The `label_mappings.json` and `per_category_thresholds.json`
contracts are defined here. The inference module (`classivore classify`) that
loads these artifacts and runs predictions is a separate implementation task.
It will need to handle: model loading, tokenization, sigmoid thresholds,
per-category threshold application, and long-document chunking with weighted
aggregation.
