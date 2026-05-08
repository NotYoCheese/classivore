#!/usr/bin/env python3
"""Comprehensive model evaluation and quality reporting.

Produces a quality_report.json alongside the model after every training run.
Taxonomy-agnostic — all category names and hierarchy come from TaxonomyConfig.

Metrics computed on the held-out test set with per-category thresholds:
- Global multi-label metrics (F1, precision, recall, hamming, subset accuracy, ranking)
- Prediction distribution (zero/over-prediction rates, histogram)
- Per-category F1/precision/recall breakdown
- Tier-1 rollup (aggregate leaf F1 by top-level category)
- Overfitting detection (train/val/test gap)
- Confusion pairs (most common misclassifications)
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)

from classivore.config.defaults import MIN_SAMPLES_FOR_OPTIMIZATION
from classivore.logging_config import get_logger
from classivore.persistence import atomic_json_save

logger = get_logger(__name__)


def get_predictions(model, tokenizer, texts, batch_size=16, max_length=512,
                    device="cpu"):
    """Run inference and return sigmoid probabilities.

    Args:
        model: Trained HuggingFace model.
        tokenizer: Tokenizer instance.
        texts: List of text strings.
        batch_size: Inference batch size.
        max_length: Max token length.
        device: Device string.

    Returns:
        Numpy array of shape (num_samples, num_classes) with sigmoid probabilities.
    """
    model.eval()
    model.to(device)

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoding = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def build_quality_report(
    test_probs,
    test_labels,
    label_names,
    categories,
    thresholds,
    global_threshold,
    model_path,
    taxonomy_slug,
    train_f1_micro=None,
    val_f1_micro=None,
):
    """Build the comprehensive quality report.

    Args:
        test_probs: Sigmoid probabilities, shape (num_samples, num_classes).
        test_labels: Binary label matrix, shape (num_samples, num_classes).
        label_names: List of leaf category names (one per column).
        categories: Full list of category dicts from load_taxonomy (for hierarchy).
        thresholds: Dict of per-category thresholds.
        global_threshold: Fallback threshold value.
        model_path: Path to model directory (for report metadata).
        taxonomy_slug: Taxonomy identifier.
        train_f1_micro: Optional training set F1 micro (for overfitting detection).
        val_f1_micro: Optional validation set F1 micro.

    Returns:
        Dict containing the full quality report.
    """
    num_classes = test_probs.shape[1]

    # Apply per-category thresholds
    threshold_vec = np.array([
        thresholds.get(label_names[i], global_threshold)
        for i in range(num_classes)
    ])
    preds = (test_probs > threshold_vec).astype(int)

    report = {
        "taxonomy": taxonomy_slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "num_test_samples": len(test_labels),
        "num_categories": num_classes,
    }

    # ── Global multi-label metrics ──
    report["global_metrics"] = _compute_global_metrics(
        test_probs, test_labels, preds, global_threshold,
    )

    # ── Prediction distribution ──
    report["prediction_distribution"] = _compute_prediction_distribution(preds)

    # ── Overfitting detection ──
    test_f1 = report["global_metrics"]["f1_micro"]
    report["overfitting"] = _compute_overfitting(
        train_f1_micro, val_f1_micro, test_f1,
    )

    # ── Per-category breakdown ──
    per_cat, zero_samples, low_confidence = _compute_per_category(
        test_labels, preds, label_names, threshold_vec,
    )
    report["per_category"] = per_cat
    report["flagged"] = {
        "zero_test_samples": zero_samples,
        "low_confidence_eval": low_confidence,
    }

    # ── Tier-1 rollup ──
    report["tier1_rollup"] = _compute_tier1_rollup(
        per_cat, label_names, categories,
    )

    # ── Confusion pairs ──
    report["confusion_pairs"] = _compute_confusion_pairs(
        test_labels, preds, label_names, top_n=100,
    )

    return report


def _compute_global_metrics(probs, labels, preds, global_threshold):
    """Compute all global multi-label metrics."""
    metrics = {
        "f1_micro": round(float(f1_score(labels, preds, average="micro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "precision_micro": round(float(precision_score(labels, preds, average="micro", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(labels, preds, average="macro", zero_division=0)), 4),
        "recall_micro": round(float(recall_score(labels, preds, average="micro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(labels, preds, average="macro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(labels, preds)), 6),
        "subset_accuracy": round(float(accuracy_score(labels, preds)), 4),
        "global_threshold": global_threshold,
    }

    # Ranking metrics require probabilities, not predictions.
    # coverage_error and label_ranking_average_precision need at least one
    # true label per sample — filter to samples with labels.
    has_labels = labels.sum(axis=1) > 0
    if has_labels.sum() > 0:
        metrics["coverage_error"] = round(float(
            coverage_error(labels[has_labels], probs[has_labels])
        ), 2)
        metrics["label_ranking_avg_precision"] = round(float(
            label_ranking_average_precision_score(labels[has_labels], probs[has_labels])
        ), 4)
    else:
        metrics["coverage_error"] = None
        metrics["label_ranking_avg_precision"] = None

    return metrics


def _compute_prediction_distribution(preds):
    """Compute prediction count distribution."""
    pred_counts = preds.sum(axis=1)
    n = len(pred_counts)

    histogram = {
        "0": int((pred_counts == 0).sum()),
        "1": int((pred_counts == 1).sum()),
        "2": int((pred_counts == 2).sum()),
        "3": int((pred_counts == 3).sum()),
        "4": int((pred_counts == 4).sum()),
        "5+": int((pred_counts >= 5).sum()),
    }

    return {
        "mean_predictions_per_sample": round(float(pred_counts.mean()), 2),
        "zero_prediction_pct": round(float((pred_counts == 0).sum() / n), 4) if n > 0 else 0,
        "over_prediction_pct": round(float((pred_counts >= 5).sum() / n), 4) if n > 0 else 0,
        "histogram": histogram,
    }


def _compute_overfitting(train_f1, val_f1, test_f1):
    """Detect overfitting from train/val/test gap."""
    result = {
        "train_f1_micro": round(train_f1, 4) if train_f1 is not None else None,
        "val_f1_micro": round(val_f1, 4) if val_f1 is not None else None,
        "test_f1_micro": round(test_f1, 4),
        "gap_warning": False,
    }

    if train_f1 is not None:
        gap = train_f1 - test_f1
        result["train_test_gap"] = round(gap, 4)
        if gap > 0.10:
            result["gap_warning"] = True

    return result


def _compute_per_category(labels, preds, label_names, threshold_vec):
    """Compute per-category F1, precision, recall."""
    num_classes = labels.shape[1]
    per_cat = []
    zero_samples = []
    low_confidence = []

    for i in range(num_classes):
        support = int(labels[:, i].sum())
        name = label_names[i]

        if support == 0:
            zero_samples.append(name)
            continue

        tp = int((preds[:, i] * labels[:, i]).sum())
        fp = int((preds[:, i] * (1 - labels[:, i])).sum())
        fn = int(((1 - preds[:, i]) * labels[:, i]).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        entry = {
            "name": name,
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "test_samples": support,
            "threshold": round(float(threshold_vec[i]), 3),
            "low_confidence": support < MIN_SAMPLES_FOR_OPTIMIZATION,
        }
        per_cat.append(entry)

        if support < MIN_SAMPLES_FOR_OPTIMIZATION:
            low_confidence.append(name)

    per_cat.sort(key=lambda x: x["f1"])
    return per_cat, zero_samples, low_confidence


def _compute_tier1_rollup(per_cat, label_names, categories):
    """Aggregate leaf F1 by tier-1 category using the taxonomy hierarchy."""
    # Build name → tier1 lookup from taxonomy
    name_to_tier1 = {}
    for cat in categories:
        if cat["is_leaf"] and cat.get("path"):
            name_to_tier1[cat["name"]] = cat["path"][0]

    # Group per-category F1 by tier1
    tier1_f1s = {}
    for entry in per_cat:
        tier1 = name_to_tier1.get(entry["name"])
        if not tier1:
            continue
        if tier1 not in tier1_f1s:
            tier1_f1s[tier1] = []
        tier1_f1s[tier1].append(entry["f1"])

    # Compute macro average per tier1
    rollup = []
    for tier1, f1s in tier1_f1s.items():
        rollup.append({
            "name": tier1,
            "leaf_f1_macro": round(float(np.mean(f1s)), 4),
            "num_leaves": len(f1s),
            "min_leaf_f1": round(float(min(f1s)), 4),
            "max_leaf_f1": round(float(max(f1s)), 4),
        })

    rollup.sort(key=lambda x: x["leaf_f1_macro"])
    return rollup


def _compute_confusion_pairs(labels, preds, label_names, top_n=100):
    """Find most common misclassification pairs."""
    confusions = Counter()
    for idx in range(len(labels)):
        true_cats = set(np.where(labels[idx] == 1)[0])
        pred_cats = set(np.where(preds[idx] == 1)[0])

        for fp in pred_cats - true_cats:
            for tc in true_cats:
                confusions[(label_names[fp], label_names[tc])] += 1

        for fn in true_cats - pred_cats:
            for pc in pred_cats:
                confusions[(label_names[pc], label_names[fn])] += 1

    return [
        {"predicted": pred, "actual": true, "count": count}
        for (pred, true), count in confusions.most_common(top_n)
    ]


def save_quality_report(report, output_dir):
    """Save quality report to model directory.

    Args:
        report: Dict from build_quality_report.
        output_dir: Model output directory.
    """
    path = Path(output_dir) / "quality_report.json"
    atomic_json_save(report, path)
    logger.info("quality_report_saved", path=str(path))


def print_quality_report(report):
    """Print human-readable quality summary to stdout."""
    gm = report["global_metrics"]
    pd = report["prediction_distribution"]
    of = report["overfitting"]
    flagged = report["flagged"]

    print(f"\n{'='*60}")
    print(f"Quality Report: {report['taxonomy']}")
    print(f"{'='*60}")

    # Global metrics
    print(f"\n  Global Metrics ({report['num_test_samples']} test samples)")
    print(f"  {'-'*50}")
    print(f"    F1 micro:          {gm['f1_micro']:.4f}")
    print(f"    F1 macro:          {gm['f1_macro']:.4f}")
    print(f"    Precision micro:   {gm['precision_micro']:.4f}")
    print(f"    Recall micro:      {gm['recall_micro']:.4f}")
    print(f"    Hamming loss:      {gm['hamming_loss']:.6f}")
    print(f"    Subset accuracy:   {gm['subset_accuracy']:.4f}")
    if gm.get("coverage_error") is not None:
        print(f"    Coverage error:    {gm['coverage_error']:.2f}")
    if gm.get("label_ranking_avg_precision") is not None:
        print(f"    Ranking avg prec:  {gm['label_ranking_avg_precision']:.4f}")

    # Prediction distribution
    print(f"\n  Prediction Distribution")
    print(f"  {'-'*50}")
    print(f"    Mean preds/sample: {pd['mean_predictions_per_sample']:.2f}")
    print(f"    Zero predictions:  {pd['zero_prediction_pct']:.1%}")
    print(f"    Over-prediction:   {pd['over_prediction_pct']:.1%}")
    hist = pd["histogram"]
    print(f"    Histogram:  0={hist['0']}  1={hist['1']}  2={hist['2']}  "
          f"3={hist['3']}  4={hist['4']}  5+={hist['5+']}")

    # Overfitting
    print(f"\n  Overfitting Detection")
    print(f"  {'-'*50}")
    if of.get("train_f1_micro") is not None:
        print(f"    Train F1:  {of['train_f1_micro']:.4f}")
    if of.get("val_f1_micro") is not None:
        print(f"    Val F1:    {of['val_f1_micro']:.4f}")
    print(f"    Test F1:   {of['test_f1_micro']:.4f}")
    if of.get("train_test_gap") is not None:
        print(f"    Gap:       {of['train_test_gap']:.4f}", end="")
        if of["gap_warning"]:
            print("  *** WARNING: possible overfitting ***")
        else:
            print()

    # Tier-1 rollup
    tier1 = report.get("tier1_rollup", [])
    if tier1:
        print(f"\n  Tier-1 Rollup ({len(tier1)} top-level categories)")
        print(f"  {'-'*50}")
        print(f"    Worst 5:")
        for t in tier1[:5]:
            print(f"      {t['leaf_f1_macro']:.3f}  {t['name']:<30} "
                  f"({t['num_leaves']} leaves, min={t['min_leaf_f1']:.3f})")
        print(f"    Best 5:")
        for t in tier1[-5:][::-1]:
            print(f"      {t['leaf_f1_macro']:.3f}  {t['name']:<30} "
                  f"({t['num_leaves']} leaves)")

    # Per-category worst/best
    per_cat = report.get("per_category", [])
    if per_cat:
        print(f"\n  Worst 20 Categories")
        print(f"  {'-'*50}")
        for c in per_cat[:20]:
            flag = " *" if c["low_confidence"] else ""
            print(f"    {c['f1']:.3f}  {c['name']:<40} "
                  f"(n={c['test_samples']}{flag})")

        print(f"\n  Best 20 Categories")
        print(f"  {'-'*50}")
        for c in per_cat[-20:][::-1]:
            print(f"    {c['f1']:.3f}  {c['name']:<40} "
                  f"(n={c['test_samples']})")

    # Flagged
    if flagged.get("zero_test_samples"):
        print(f"\n  Categories with zero test samples: {len(flagged['zero_test_samples'])}")
    if flagged.get("low_confidence_eval"):
        print(f"  Categories with < 5 test samples:  {len(flagged['low_confidence_eval'])}")

    # Confusion pairs
    pairs = report.get("confusion_pairs", [])
    if pairs:
        print(f"\n  Top 10 Confusion Pairs")
        print(f"  {'-'*50}")
        for p in pairs[:10]:
            print(f"    {p['count']:4d}  {p['predicted']:<30} ↔ {p['actual']}")

    print()
