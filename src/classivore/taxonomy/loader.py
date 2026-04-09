#!/usr/bin/env python3
"""Load taxonomy from CSV and provide hierarchy navigation."""

import csv
from pathlib import Path


def load_taxonomy(config):
    """Load taxonomy CSV into a list of category dicts.

    Args:
        config: TaxonomyConfig with taxonomy_file path and column mappings.

    Returns:
        List of category dicts in the internal format.
    """
    path = config.taxonomy_file
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    categories = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories.append({
                "id": row[config.id_column],
                "name": row[config.name_column],
                "display_name": row.get("display_name", row[config.name_column]),
                "parent_id": row.get(config.parent_column, ""),
                "path": row.get("path", row[config.name_column]).split(" > "),
                "depth": int(row.get("depth", 1)),
                "is_leaf": row.get("is_leaf", "True").strip().lower() == "true",
                "children_count": int(row.get("children_count", 0)),
                "description": row.get("description", ""),
                "boundaries": row.get("boundaries", ""),
                "aliases": [
                    a.strip() for a in row.get("aliases", "").split("|")
                    if a.strip()
                ] if row.get("aliases", "").strip() else [],
                "difficulty": row.get("difficulty", "").strip() or "medium",
            })

    return categories


def build_hierarchy(categories):
    """Build a parent_id → children mapping.

    Args:
        categories: List of category dicts.

    Returns:
        Dict mapping parent_id to list of child category dicts.
    """
    hierarchy = {}
    for cat in categories:
        parent_id = cat["parent_id"]
        if parent_id not in hierarchy:
            hierarchy[parent_id] = []
        hierarchy[parent_id].append(cat)
    return hierarchy


def get_siblings(category, hierarchy):
    """Get names of sibling categories (same parent, excluding self).

    Args:
        category: A category dict.
        hierarchy: Dict from build_hierarchy.

    Returns:
        List of sibling category names.
    """
    parent_id = category["parent_id"]
    siblings = hierarchy.get(parent_id, [])
    return [s["name"] for s in siblings if s["id"] != category["id"]]


def get_children(category, hierarchy):
    """Get names of direct children.

    Args:
        category: A category dict.
        hierarchy: Dict from build_hierarchy.

    Returns:
        List of child category names.
    """
    children = hierarchy.get(category["id"], [])
    return [c["name"] for c in children]


def save_enriched_taxonomy(categories, output_path):
    """Write enriched taxonomy to CSV with description and boundaries columns.

    Args:
        categories: List of category dicts.
        output_path: Path to write the CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "parent_id", "name", "display_name", "path",
        "depth", "is_leaf", "children_count", "description", "boundaries",
        "aliases", "difficulty",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cat in categories:
            row = {k: cat.get(k, "") for k in fieldnames}
            # Convert path list back to string
            if isinstance(row["path"], list):
                row["path"] = " > ".join(row["path"])
            # Convert bool back to string
            if isinstance(row["is_leaf"], bool):
                row["is_leaf"] = str(row["is_leaf"])
            # Convert aliases list to pipe-delimited string
            if isinstance(row.get("aliases"), list):
                row["aliases"] = " | ".join(row["aliases"])
            writer.writerow(row)
