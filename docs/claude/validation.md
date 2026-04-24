# Validation Subsystem

## Modules

- `src/classivore/validation/loader.py` — Load corpus + labels into a flat DataFrame for analysis.
- `src/classivore/validation/runner.py` — Orchestrate label-lens analyses and taxonomy-aware checks.
- `src/classivore/validation/formatter.py` — Format reports for terminal output.

## External Dependency: label-lens

Validation uses [label-lens](../../label-lens) as a library for general-purpose data quality analysis:
- Class distribution analysis (imbalance ratio, effective classes, long-tail detection)
- Exact and near-duplicate detection
- Label noise scoring via cross-validated confidence

Install with: `uv pip install label-lens` or `uv pip install classivore[validate]`

## Classivore-Specific Checks (on top of label-lens)

- **Taxonomy coverage**: Which taxonomy categories have no training data?
- **Unknown labels**: Labels in data that don't match any taxonomy category.
- **Thin categories**: Categories with fewer than 10 samples.
- **LLM confidence distribution**: Flags high proportion of low-confidence labels.

## Data Flow

1. `loader.py` joins `data/corpus/pages.json` with `data/labels/<taxonomy-slug>/labels.json` (preferred) or the legacy `data/labels/<taxonomy-slug>.json` path
2. Multi-label entries are exploded into one row per (text, label) pair
3. label-lens runs distribution, duplicate, and noise analyses on the flat DataFrame
4. Taxonomy-aware checks compare labels against known categories from `taxonomies/<slug>/`
5. All findings are aggregated into a single report with severity ratings

## CLI Usage

```bash
# Validate labeled data
classivore validate --labeled --taxonomy iab-2.2

# Validate scraped corpus (no labels needed)
classivore validate --taxonomy iab-2.2

# Skip noise scoring for faster results
classivore validate --labeled --skip-noise

# Disable color output
classivore validate --labeled --no-color
```

## Tests

- `tests/unit/test_validation.py` — loader, runner, taxonomy checks, confidence checks, formatter
