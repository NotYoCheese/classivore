#!/usr/bin/env python3
"""Model evaluation and reporting.

Runs final evaluation on the held-out test set with per-category thresholds.
Produces detailed metrics including per-category F1, worst performers, and
confusion analysis.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from classivore.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(model, tokenizer, test_data, label_names, thresholds=None,
                   global_threshold=0.55, batch_size=16, device="cpu",
                   max_length=512):
    """Evaluate model on test data with optional per-category thresholds.

    Args:
        model: Trained HuggingFace model.
        tokenizer: Tokenizer instance.
        test_data: Dict with "texts" and "label_matrix".
        label_names: List of category names.
        thresholds: Optional dict of per-category thresholds.
        global_threshold: Fallback threshold. Default 0.55.
        batch_size: Inference batch size.
        device: Device string.
        max_length: Max token length.

    Returns:
        Dict with evaluation metrics and per-category breakdown.
    """
    model.eval()
    model.to(device)

    texts = test_data["texts"]
    labels = test_data["label_matrix"]
    num_classes = labels.shape[1]

    # Get predictions
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    # Apply thresholds
    if thresholds:
        threshold_vec = np.array([
            thresholds.get(label_names[i], global_threshold)
            for i in range(num_classes)
        ])
        preds = (all_probs > threshold_vec).astype(int)
    else:
        preds = (all_probs > global_threshold).astype(int)

    # Global metrics
    metrics = {
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "threshold_type": "per_category" if thresholds else "global",
        "global_threshold": global_threshold,
        "avg_predictions_per_sample": float(preds.sum(axis=1).mean()),
    }

    # Per-category metrics
    per_category = []
    for i in range(num_classes):
        support = int(labels[:, i].sum())
        if support == 0:
            continue
        cat_f1 = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
        per_category.append({
            "name": label_names[i],
            "f1": cat_f1,
            "support": support,
            "threshold": float(threshold_vec[i]) if thresholds else global_threshold,
        })

    per_category.sort(key=lambda x: x["f1"])
    metrics["worst_categories"] = per_category[:20]
    metrics["best_categories"] = per_category[-20:][::-1]
    metrics["per_category"] = per_category

    logger.info(
        "evaluation_complete",
        f1_micro=f"{metrics['f1_micro']:.4f}",
        f1_macro=f"{metrics['f1_macro']:.4f}",
        precision=f"{metrics['precision_micro']:.4f}",
        recall=f"{metrics['recall_micro']:.4f}",
    )

    return metrics


def save_evaluation(metrics, output_dir):
    """Save evaluation metrics to the model directory.

    Args:
        metrics: Dict from evaluate_model.
        output_dir: Model output directory.
    """
    path = Path(output_dir) / "evaluation_report.json"
    # Remove large per_category list from main report (save separately)
    per_cat = metrics.pop("per_category", [])

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    if per_cat:
        per_cat_path = Path(output_dir) / "per_category_metrics.json"
        with open(per_cat_path, "w") as f:
            json.dump(per_cat, f, indent=2)

    # Restore for caller
    metrics["per_category"] = per_cat

    logger.info("evaluation_saved", path=str(path))


def print_evaluation(metrics):
    """Print evaluation results to console."""
    print(f"\nEvaluation Results")
    print("=" * 60)
    print(f"  F1 Micro:     {metrics['f1_micro']:.4f}")
    print(f"  F1 Macro:     {metrics['f1_macro']:.4f}")
    print(f"  Precision:    {metrics['precision_micro']:.4f}")
    print(f"  Recall:       {metrics['recall_micro']:.4f}")
    print(f"  Avg preds/sample: {metrics['avg_predictions_per_sample']:.2f}")
    print(f"  Threshold:    {metrics['threshold_type']} "
          f"(global={metrics['global_threshold']})")

    if metrics.get("worst_categories"):
        print(f"\n  Worst 10 categories:")
        for cat in metrics["worst_categories"][:10]:
            print(f"    {cat['f1']:.3f}  {cat['name']:<40} ({cat['support']} samples)")

    if metrics.get("best_categories"):
        print(f"\n  Best 10 categories:")
        for cat in metrics["best_categories"][:10]:
            print(f"    {cat['f1']:.3f}  {cat['name']:<40} ({cat['support']} samples)")
