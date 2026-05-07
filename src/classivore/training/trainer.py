#!/usr/bin/env python3
"""Training orchestration for multi-label classification.

Fine-tunes DeBERTa-v3 with weighted focal loss for taxonomy classification.
Uses HuggingFace Trainer with custom loss function, early stopping, and
automatic device selection.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from classivore.logging_config import get_logger
from classivore.persistence import atomic_json_save
from classivore.training.dataset import (
    ClassificationDataset,
    load_training_data,
    split_data,
)
from classivore.training.loss import WeightedFocalLoss, compute_class_weights

logger = get_logger(__name__)


def select_device(override=None):
    """Auto-detect best available device.

    Args:
        override: Force a specific device ("cuda", "mps", "cpu").

    Returns:
        Tuple of (device string, use_fp16 bool).
    """
    if override:
        use_fp16 = override == "cuda"
        return override, use_fp16

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("device_selected", device="cuda", name=name)
        return "cuda", True
    elif torch.backends.mps.is_available():
        logger.info("device_selected", device="mps", name="Apple Silicon")
        return "mps", False  # MPS doesn't support FP16 training reliably
    else:
        logger.info("device_selected", device="cpu", name="CPU")
        return "cpu", False


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer with custom focal loss."""

    def __init__(self, focal_loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        confidence_weights = inputs.pop("confidence_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss_fn(logits, labels, confidence_weights)

        return (loss, outputs) if return_outputs else loss


def train_model(config, data_dir, output_dir=None, device=None,
                epochs=None, batch_size=None, dry_run=False):
    """Train a multi-label classifier.

    Args:
        config: TaxonomyConfig instance.
        data_dir: Path to data directory.
        output_dir: Override output directory.
        device: Override device selection ("cuda", "mps", "cpu").
        epochs: Override number of epochs.
        batch_size: Override batch size.
        dry_run: Show data stats without training.

    Returns:
        Dict with training results and model path.
    """
    data_dir = Path(data_dir)
    model_base = config.model_base

    # Load and prepare data
    data = load_training_data(config, data_dir)
    stats = data["stats"]

    if dry_run:
        _print_dry_run(data, config)
        return {"dry_run": True, "stats": stats}

    # Device selection
    device_name, use_fp16 = select_device(device)

    # Training parameters
    num_epochs = epochs or config.num_epochs
    bs = batch_size or config.batch_size
    lr = config.learning_rate
    max_length = config.max_length
    num_classes = stats["num_classes"]

    logger.info(
        "training_start",
        model=model_base,
        device=device_name,
        samples=stats["total_pages"],
        classes=num_classes,
        epochs=num_epochs,
        batch_size=bs,
        fp16=use_fp16,
    )

    # Split data
    splits = split_data(data)
    logger.info(
        "data_split",
        train=len(splits["train"]["texts"]),
        val=len(splits["val"]["texts"]),
        test=len(splits["test"]["texts"]),
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_base,
        num_labels=num_classes,
        problem_type="multi_label_classification",
    )

    # Build datasets
    train_ds = ClassificationDataset(
        splits["train"]["texts"], splits["train"]["label_matrix"],
        splits["train"]["confidence_matrix"], tokenizer, max_length,
    )
    val_ds = ClassificationDataset(
        splits["val"]["texts"], splits["val"]["label_matrix"],
        splits["val"]["confidence_matrix"], tokenizer, max_length,
    )

    # Compute class weights from training data
    train_label_counts = splits["train"]["label_matrix"].sum(axis=0)
    class_weights = compute_class_weights(
        train_label_counts,
        total_samples=len(splits["train"]["texts"]),
        cap=config.class_weight_cap,
    ).to(device_name)

    # Build loss function
    loss_fn = WeightedFocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        class_weights=class_weights,
    ).to(device_name)

    # Output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("models") / config.slug / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments. use_cpu honors device="cpu" overrides on Macs;
    # without it HF Trainer auto-selects MPS for the model while our
    # class_weights/loss_fn stay on CPU, producing a device-mismatch error
    # at the first forward pass.
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr,
        fp16=use_fp16,
        use_cpu=(device_name == "cpu"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    # Custom trainer with focal loss
    trainer = FocalLossTrainer(
        focal_loss_fn=loss_fn,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_make_compute_metrics(num_classes),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    logger.info("training_complete", time_seconds=int(training_time))

    # Evaluate on validation set
    val_metrics = trainer.evaluate()

    # Save model and artifacts
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label mappings
    categories = _load_categories(config)
    name_to_id = {c["name"]: c["id"] for c in categories}
    label_mappings = {
        "id_to_index": {
            name_to_id[name]: i
            for i, name in enumerate(data["label_names"])
            if name in name_to_id
        },
        "index_to_name": {str(i): name for i, name in enumerate(data["label_names"])},
        "index_to_id": {
            str(i): name_to_id[name]
            for i, name in enumerate(data["label_names"])
            if name in name_to_id
        },
    }
    atomic_json_save(label_mappings, output_dir / "label_mappings.json")

    # Save taxonomy metadata for self-contained inference
    cat_by_name = {c["name"]: c for c in categories}
    taxonomy_metadata = {
        "taxonomy_slug": config.slug,
        "taxonomy_version": config.version,
        "categories": {},
    }
    for name in data["label_names"]:
        cat = cat_by_name.get(name)
        if cat:
            taxonomy_metadata["categories"][name] = {
                "id": cat["id"],
                "path": cat["path"],
                "depth": cat["depth"],
                "is_leaf": cat["is_leaf"],
                "parent_name": cat["path"][-2] if len(cat["path"]) >= 2 else None,
            }
    atomic_json_save(taxonomy_metadata, output_dir / "taxonomy_metadata.json")

    # Save training report
    report = {
        "taxonomy": config.slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_base": model_base,
        "num_categories": num_classes,
        "num_train": len(splits["train"]["texts"]),
        "num_val": len(splits["val"]["texts"]),
        "num_test": len(splits["test"]["texts"]),
        "hyperparameters": {
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": num_epochs,
            "max_length": max_length,
            "focal_alpha": config.focal_alpha,
            "focal_gamma": config.focal_gamma,
            "class_weight_cap": config.class_weight_cap,
        },
        "metrics": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "training_time_seconds": int(training_time),
        "device": device_name,
        "data_stats": {
            "total_pages": stats["total_pages"],
            "legacy_pages": stats["legacy_pages"],
            "pipeline_pages": stats["pipeline_pages"],
        },
    }
    atomic_json_save(report, output_dir / "training_report.json")

    logger.info("model_saved", path=str(output_dir))

    # ── Quality report ──
    from classivore.training.evaluate import (
        build_quality_report,
        get_predictions,
        print_quality_report,
        save_quality_report,
    )
    from classivore.training.thresholds import (
        optimize_global_threshold,
        optimize_per_category_thresholds,
        save_thresholds,
    )

    logger.info("running_quality_evaluation")

    # Get predictions on val and test
    val_probs = get_predictions(
        model, tokenizer, splits["val"]["texts"],
        batch_size=bs * 2, max_length=max_length, device=device_name,
    )
    test_probs = get_predictions(
        model, tokenizer, splits["test"]["texts"],
        batch_size=bs * 2, max_length=max_length, device=device_name,
    )

    # Save raw probabilities for future analysis
    np.save(output_dir / "val_probs.npy", val_probs)
    np.save(output_dir / "test_probs.npy", test_probs)

    # Optimize thresholds on val set
    val_labels = splits["val"]["label_matrix"]
    global_t, _ = optimize_global_threshold(val_probs, val_labels)
    per_cat_thresholds = optimize_per_category_thresholds(
        val_probs, val_labels, data["label_names"], global_threshold=global_t,
    )
    save_thresholds(per_cat_thresholds, output_dir)

    # Compute train F1 for overfitting detection
    train_probs = get_predictions(
        model, tokenizer, splits["train"]["texts"][:2000],  # Sample for speed
        batch_size=bs * 2, max_length=max_length, device=device_name,
    )
    train_preds = (train_probs > 0.5).astype(int)
    train_labels_sample = splits["train"]["label_matrix"][:2000]
    from sklearn.metrics import f1_score as _f1
    train_f1 = float(_f1(train_labels_sample, train_preds, average="micro", zero_division=0))

    # Build and save quality report
    quality_report = build_quality_report(
        test_probs=test_probs,
        test_labels=splits["test"]["label_matrix"],
        label_names=data["label_names"],
        categories=categories,
        thresholds=per_cat_thresholds,
        global_threshold=global_t,
        model_path=output_dir,
        taxonomy_slug=config.slug,
        train_f1_micro=train_f1,
        val_f1_micro=val_metrics.get("eval_f1_micro"),
    )
    save_quality_report(quality_report, output_dir)
    print_quality_report(quality_report)

    return {
        "model_path": str(output_dir),
        "metrics": val_metrics,
        "training_time": int(training_time),
        "report": report,
        "quality_report": quality_report,
    }


def _make_compute_metrics(num_classes):
    """Create a metrics computation function for HF Trainer."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs > 0.5).astype(int)

        return {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        }

    return compute_metrics


def _load_categories(config):
    """Load taxonomy categories (cached helper)."""
    from classivore.taxonomy.loader import load_taxonomy
    return load_taxonomy(config)


def _print_dry_run(data, config):
    """Print training data summary for dry run."""
    stats = data["stats"]
    splits = split_data(data)

    print(f"\nTraining Dry Run: {config.slug}")
    print("=" * 60)
    print(f"  Model:           {config.model_base}")
    print(f"  Total pages:     {stats['total_pages']}")
    print(f"    Legacy:        {stats['legacy_pages']}")
    print(f"    Pipeline:      {stats['pipeline_pages']}")
    print(f"  Leaf categories: {stats['num_classes']}")
    print(f"    With labels:   {stats['categories_with_labels']}")
    print(f"    Below 20:      {stats['categories_below_20']}")
    print(f"  Samples/class:   min={stats['min_samples']}, "
          f"median={stats['median_samples']}, max={stats['max_samples']}")
    print()
    print(f"  Train split:     {len(splits['train']['texts'])}")
    print(f"  Val split:       {len(splits['val']['texts'])}")
    print(f"  Test split:      {len(splits['test']['texts'])}")
    print()
    print(f"  Hyperparameters:")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Batch size:    {config.batch_size}")
    print(f"    Epochs:        {config.num_epochs}")
    print(f"    Max length:    {config.max_length}")
    print(f"    Focal alpha:   {config.focal_alpha}")
    print(f"    Focal gamma:   {config.focal_gamma}")
    print(f"    Weight cap:    {config.class_weight_cap}")

    # Estimated training time
    est_seconds = len(splits["train"]["texts"]) * config.num_epochs * 1.2 / config.batch_size
    est_minutes = est_seconds / 60
    if est_minutes > 60:
        print(f"  Est. time (A10): ~{est_minutes / 60:.1f} hours")
    else:
        print(f"  Est. time (A10): ~{est_minutes:.0f} minutes")

    # Categories below threshold
    label_counts = stats["label_counts"]
    label_names = data["label_names"]
    below_20 = [(label_names[i], int(label_counts[i]))
                for i in range(len(label_names)) if label_counts[i] < 20]
    if below_20:
        below_20.sort(key=lambda x: x[1])
        print(f"\n  Categories below 20 training samples ({len(below_20)}):")
        for name, count in below_20[:15]:
            print(f"    {count:>4}  {name}")
        if len(below_20) > 15:
            print(f"    ... and {len(below_20) - 15} more")
