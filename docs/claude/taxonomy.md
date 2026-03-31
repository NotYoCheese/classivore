# Taxonomy Subsystem

## Modules

- `src/classivore/taxonomy/loader.py` — Load taxonomy from CSV/TSV/JSON. Normalize to internal format.
- `src/classivore/taxonomy/validator.py` — Validate hierarchy integrity, check for orphans/cycles, verify required columns.
- `src/classivore/taxonomy/enricher.py` — Generate descriptions for categories that lack them using LLM. Uses sibling/parent context for disambiguation.

## Internal Taxonomy Format

All loaders normalize to this structure:

```python
{
    "id": "22",
    "name": "Green Vehicles",
    "display_name": "Automotive: Green Vehicles",  # Tier1: LowestTier
    "parent_id": "16",
    "path": ["Automotive", "Auto Type", "Green Vehicles"],
    "depth": 3,
    "is_leaf": True,
    "children_count": 0,
    "description": "Electric, hybrid, and alternative fuel vehicles...",
    "boundaries": "Distinguished from Performance Cars by focus on efficiency..."
}
```

## Hierarchy Strategy: Leaf-Only with Inferred Ancestors

The model predicts ONLY leaf nodes (`is_leaf: True`). Parent/ancestor nodes are
inferred from the taxonomy tree in post-processing. This is the industry standard
and gives the best balance of model simplicity and output richness.

- **Training:** Only leaf nodes are label targets. Parent nodes are never predicted directly.
- **Inference:** Model outputs leaf predictions. The inference layer walks up the tree
  to build the full path for each prediction.
- **Multi-label:** Predictions can come from any branch of the taxonomy. There is NO
  constraint that multiple labels must share a common ancestor. An article about car
  aerodynamics can legitimately be tagged "Automotive: Sedan" AND "Science: Physics."
- **API response:** Returns full path per prediction so customers can filter at any tier.
- **taxonomy.csv stores all nodes** (parents + leaves) for hierarchy lookups, enrichment
  context, and path construction. The `is_leaf` column distinguishes classifiable
  categories from structural parent nodes.
- **Configurable:** For taxonomies that allow labeling at any level (not just leaves),
  set `leaf_only: false` in config.yaml. This is the exception.

## Config Schema (config.yaml)

Each taxonomy directory has a `config.yaml`. See `taxonomies/iab_2.2/config.yaml` for a complete example. Required fields:

- `name`, `version`, `slug` — identity
- `taxonomy_file` — path to raw taxonomy file (relative to taxonomy dir)
- `id_column`, `name_column`, `parent_column` — column mapping
- `classification_type` — `multi_label` or `single_label`
- `max_labels` — max categories per prediction (e.g., 3)

Optional fields:
- `description_column` — if taxonomy already has descriptions
- `enriched_file` — output path for enriched taxonomy
- `domain_hints` — dict of category → domains for data collection
- `excluded_categories` — list of categories to skip in expansion
- All training hyperparameters (model_base, batch_size, focal_loss, etc.)

## Enrichment Process

The enricher sends batches of categories to the LLM with this context per category:
- Category name and full hierarchy path
- Sibling categories (same parent) — critical for boundary generation
- Children categories (if any) — to understand scope

Output per category: one definition sentence + one boundary sentence.

Cost: ~$0.50-2.00 per taxonomy using Haiku.

## Tests

- `tests/unit/test_taxonomy_loader.py` — test loading from each format
- `tests/unit/test_taxonomy_validator.py` — test hierarchy validation, error detection
- `tests/unit/test_taxonomy_enricher.py` — test prompt generation, response parsing (mock LLM)
