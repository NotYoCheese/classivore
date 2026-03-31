#!/usr/bin/env python3
"""Format validation results for terminal output."""

from __future__ import annotations

from typing import Any

SEVERITY_ICONS = {
    "Critical": "✗",
    "Warning": "!",
    "Info": "·",
}

SEVERITY_COLORS = {
    "Critical": "\033[91m",  # red
    "Warning": "\033[93m",   # yellow
    "Info": "\033[90m",      # gray
}

RESET = "\033[0m"
BOLD = "\033[1m"


def format_report(report: dict[str, Any], use_color: bool = True) -> str:
    """Format a validation report for terminal display.

    Args:
        report: Report dict from validate_labeled or validate_scraped.
        use_color: Whether to use ANSI color codes.

    Returns:
        Formatted string for printing.
    """
    lines = []

    # Header
    severity = report["overall_severity"]
    if use_color:
        color = SEVERITY_COLORS.get(severity, "")
        lines.append(f"\n{BOLD}Data Quality Report{RESET}")
        if "taxonomy_slug" in report:
            lines.append(f"Taxonomy: {report['taxonomy_slug']}")
        lines.append(f"Overall: {color}{severity}{RESET}")
    else:
        lines.append("\nData Quality Report")
        if "taxonomy_slug" in report:
            lines.append(f"Taxonomy: {report['taxonomy_slug']}")
        lines.append(f"Overall: {severity}")

    # Summary stats
    lines.append("")
    if "total_pages" in report:
        lines.append(f"  Pages: {report['total_pages']}")
    if "total_labels" in report:
        lines.append(f"  Labels: {report['total_labels']}")
    if "summary" in report:
        s = report["summary"]
        lines.append(f"  Classes: {s.get('num_classes', 'N/A')}")
        lines.append(f"  Imbalance ratio: {s.get('imbalance_ratio', 'N/A')}:1")
        lines.append(f"  Exact duplicates: {s.get('exact_duplicates', 0)}")
        lines.append(f"  Near duplicates: {s.get('near_duplicates', 0)}")
        lines.append(f"  Suspect mislabels: {s.get('suspect_mislabels', 0)}")

    # Findings
    lines.append("")
    lines.append(f"{BOLD}Findings{RESET}" if use_color else "Findings")
    lines.append("-" * 60)

    for finding in report.get("findings", []):
        sev = finding["severity"]
        icon = SEVERITY_ICONS.get(sev, " ")
        msg = finding["message"]
        cat = finding.get("category", "")
        prefix = f"[{cat}]" if cat else ""

        if use_color:
            color = SEVERITY_COLORS.get(sev, "")
            lines.append(f"  {color}{icon}{RESET} {prefix} {msg}")
        else:
            lines.append(f"  {icon} {prefix} {msg}")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        lines.append("")
        lines.append(f"{BOLD}Recommendations{RESET}" if use_color else "Recommendations")
        lines.append("-" * 60)
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")

    lines.append("")
    return "\n".join(lines)
