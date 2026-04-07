#!/usr/bin/env python3
"""Model card generation for HuggingFace Hub."""


def generate_model_card(training_report, quality_report, repo_id, version):
    """Generate a README.md string for HuggingFace model repo.

    Args:
        training_report: Parsed training_report.json dict.
        quality_report: Parsed quality_report.json dict, or None.
        repo_id: HuggingFace repo ID (e.g. "classivore/iab22-deberta-large").
        version: Version tag (e.g. "v1.0.0").

    Returns:
        Model card string (Markdown).
    """
    metrics = training_report.get("metrics", {})
    hyperparams = training_report.get("hyperparameters", {})

    sections = []

    # Header
    sections.append(f"# {repo_id}\n")
    sections.append(f"**Version:** {version}\n")
    sections.append(f"**Base model:** {training_report.get('model_base', 'unknown')}\n")
    sections.append(
        f"**Taxonomy:** {training_report.get('taxonomy', 'unknown')} "
        f"({training_report.get('num_categories', '?')} categories)\n"
    )

    # Metrics
    sections.append("## Metrics\n")
    metric_lines = []
    for key in ("eval_f1_micro", "eval_f1_macro", "eval_precision_micro", "eval_recall_micro"):
        if key in metrics:
            label = key.replace("eval_", "").replace("_", " ").title()
            metric_lines.append(f"| {label} | {metrics[key]:.4f} |")
    if metric_lines:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.extend(metric_lines)
        sections.append("")

    # Quality report extras
    if quality_report:
        global_t = quality_report.get("global_threshold")
        global_metrics = quality_report.get("global_metrics", {})
        test_f1 = global_metrics.get("f1_micro")
        if global_t is not None:
            sections.append(f"**Global threshold:** {global_t}")
        if test_f1 is not None:
            sections.append(f"**Test F1 micro:** {test_f1:.4f}")
        sections.append("")

    # Training metadata
    sections.append("## Training\n")
    sections.append(f"- **Timestamp:** {training_report.get('timestamp', 'unknown')}")
    sections.append(f"- **Device:** {training_report.get('device', 'unknown')}")
    time_s = training_report.get("training_time_seconds")
    if time_s is not None:
        minutes = time_s // 60
        sections.append(f"- **Training time:** {minutes}m {time_s % 60}s")
    sections.append("")

    # Hyperparameters
    if hyperparams:
        sections.append("## Hyperparameters\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for key, value in hyperparams.items():
            label = key.replace("_", " ").title()
            sections.append(f"| {label} | {value} |")
        sections.append("")

    # Usage
    sections.append("## Usage\n")
    sections.append(
        "Designed for use with classivore-api. "
        "Use `classivore.inference.Classifier` to load.\n"
    )

    # License
    sections.append("## License\n")
    sections.append("MIT\n")

    return "\n".join(sections)
