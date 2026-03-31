#!/usr/bin/env python3
"""Run data quality validation using label-lens and taxonomy-aware checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from label_lens import (
        analyze_distribution,
        find_exact_duplicates,
        find_near_duplicates,
        generate_report,
        score_label_noise,
    )

    HAS_LABEL_LENS = True
except ImportError:
    HAS_LABEL_LENS = False

from classivore.validation.loader import load_labeled_data, load_scraped_data


def validate_labeled(
    data_dir: Path,
    taxonomy_slug: str,
    taxonomy_categories: list[str] | None = None,
    skip_noise: bool = False,
) -> dict[str, Any]:
    """Run full quality validation on labeled data.

    Args:
        data_dir: Path to the data directory.
        taxonomy_slug: Taxonomy slug (e.g. "iab-2.2").
        taxonomy_categories: Known leaf categories from taxonomy. Enables
            coverage and unknown-label checks.
        skip_noise: Skip label noise scoring (slow on large datasets).

    Returns:
        Dict with label-lens report plus taxonomy-specific findings.
    """
    if not HAS_LABEL_LENS:
        raise ImportError(
            "label-lens is required for validation. "
            "Install with: uv pip install label-lens"
        )

    df = load_labeled_data(data_dir, taxonomy_slug)

    # Prepare for label-lens (expects 'text' and 'label' columns)
    ll_df = df[["text", "label"]].copy()

    # Run label-lens analyses
    dist_result = analyze_distribution(ll_df)
    dup_result = find_exact_duplicates(ll_df)
    near_dup_result = find_near_duplicates(ll_df)

    if skip_noise or len(ll_df) < 50:
        noise_result = {
            "total_samples": len(ll_df),
            "num_suspects": 0,
            "suspect_percentage": 0.0,
            "threshold_confidence": 0.0,
            "suspects": [],
            "all_confidences": [],
        }
    else:
        noise_result = score_label_noise(ll_df)

    report = generate_report(dist_result, dup_result, near_dup_result, noise_result)

    # Add taxonomy-aware checks
    taxonomy_findings = _check_taxonomy(df, taxonomy_categories)
    report["findings"].extend(taxonomy_findings["findings"])
    report["recommendations"].extend(taxonomy_findings["recommendations"])

    # Add confidence distribution summary
    confidence_findings = _check_confidence(df)
    report["findings"].extend(confidence_findings["findings"])
    report["recommendations"].extend(confidence_findings["recommendations"])

    # Recalculate overall severity with new findings
    severities = [f["severity"] for f in report["findings"]]
    if "Critical" in severities:
        report["overall_severity"] = "Critical"
    elif "Warning" in severities:
        report["overall_severity"] = "Warning"

    report["taxonomy_slug"] = taxonomy_slug
    report["total_pages"] = df["url"].nunique()
    report["total_labels"] = len(df)

    return report


def validate_scraped(data_dir: Path) -> dict[str, Any]:
    """Run basic quality checks on scraped (unlabeled) corpus data.

    Returns:
        Dict with findings about corpus quality.
    """
    df = load_scraped_data(data_dir)
    findings = []
    recommendations = []

    # Duplicate URLs
    dup_urls = df["url"].duplicated().sum()
    if dup_urls > 0:
        findings.append({
            "category": "Corpus",
            "severity": "Warning",
            "message": f"{dup_urls} duplicate URLs found in corpus.",
        })
        recommendations.append("Remove duplicate URLs to avoid wasted labeling budget.")

    # Content hash duplicates (same text, different URLs)
    if "content_hash" in df.columns:
        hash_dups = df["content_hash"].duplicated().sum()
        if hash_dups > 0:
            findings.append({
                "category": "Corpus",
                "severity": "Warning",
                "message": (
                    f"{hash_dups} content duplicates (same text, different URLs)."
                ),
            })
            recommendations.append(
                "Deduplicate by content hash before labeling to save costs."
            )

    # Word count distribution
    if "word_count" in df.columns:
        short = (df["word_count"] < 100).sum()
        if short > 0:
            pct = round(short / len(df) * 100, 1)
            findings.append({
                "category": "Corpus",
                "severity": "Warning" if pct > 10 else "Info",
                "message": f"{short} pages ({pct}%) have fewer than 100 words.",
            })
            if pct > 10:
                recommendations.append(
                    "Short pages may not contain enough signal for classification. "
                    "Review and consider removing pages under 100 words."
                )

    # Total corpus size
    findings.append({
        "category": "Corpus",
        "severity": "Info",
        "message": f"Corpus contains {len(df)} pages.",
    })

    severities = [f["severity"] for f in findings]
    overall = "Critical" if "Critical" in severities else (
        "Warning" if "Warning" in severities else "Info"
    )

    return {
        "overall_severity": overall,
        "findings": findings,
        "recommendations": recommendations,
        "total_pages": len(df),
    }


def _check_taxonomy(
    df: pd.DataFrame, taxonomy_categories: list[str] | None
) -> dict[str, Any]:
    """Taxonomy-aware validation checks."""
    findings = []
    recommendations = []

    if taxonomy_categories is None:
        return {"findings": findings, "recommendations": recommendations}

    known = set(taxonomy_categories)
    used = set(df["label"].unique())

    # Unknown labels (in data but not in taxonomy)
    unknown = used - known
    if unknown:
        findings.append({
            "category": "Taxonomy",
            "severity": "Critical",
            "message": f"{len(unknown)} labels not in taxonomy: {sorted(unknown)[:5]}{'...' if len(unknown) > 5 else ''}",
        })
        recommendations.append(
            "Labels not in the taxonomy indicate labeling errors or a stale taxonomy. "
            "Review and correct these labels."
        )

    # Missing coverage (in taxonomy but no data)
    uncovered = known - used
    if uncovered:
        coverage_pct = round((1 - len(uncovered) / len(known)) * 100, 1)
        severity = "Critical" if coverage_pct < 50 else (
            "Warning" if coverage_pct < 80 else "Info"
        )
        findings.append({
            "category": "Taxonomy",
            "severity": severity,
            "message": (
                f"{len(uncovered)} of {len(known)} categories have no training data "
                f"({coverage_pct}% coverage)."
            ),
        })
        if coverage_pct < 80:
            recommendations.append(
                "Run `classivore agent` to collect data for uncovered categories: "
                f"{sorted(uncovered)[:5]}{'...' if len(uncovered) > 5 else ''}"
            )

    # Thin categories (fewer than 10 samples)
    label_counts = df["label"].value_counts()
    thin = label_counts[label_counts < 10]
    if len(thin) > 0:
        findings.append({
            "category": "Taxonomy",
            "severity": "Warning",
            "message": f"{len(thin)} categories have fewer than 10 labeled samples.",
        })
        recommendations.append(
            "Categories with very few samples will train poorly. "
            "Collect more data or consider excluding them."
        )

    return {"findings": findings, "recommendations": recommendations}


def _check_confidence(df: pd.DataFrame) -> dict[str, Any]:
    """Check LLM confidence score distribution."""
    findings = []
    recommendations = []

    if "confidence" not in df.columns:
        return {"findings": findings, "recommendations": recommendations}

    mean_conf = df["confidence"].mean()
    low_conf = (df["confidence"] < 0.5).sum()
    low_pct = round(low_conf / len(df) * 100, 1)

    if low_pct > 20:
        findings.append({
            "category": "Confidence",
            "severity": "Warning",
            "message": (
                f"{low_conf} labels ({low_pct}%) have confidence below 0.5 "
                f"(mean: {mean_conf:.2f})."
            ),
        })
        recommendations.append(
            "High proportion of low-confidence labels suggests ambiguous categories "
            "or poor prompt quality. Review low-confidence samples and consider relabeling."
        )
    else:
        findings.append({
            "category": "Confidence",
            "severity": "Info",
            "message": f"Mean LLM confidence: {mean_conf:.2f} ({low_pct}% below 0.5).",
        })

    return {"findings": findings, "recommendations": recommendations}
