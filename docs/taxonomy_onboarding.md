# Taxonomy Onboarding Guide

How to add a new taxonomy to classivore and build a classifier from scratch.

## Prerequisites

- classivore installed (`pip install -e .`)
- Anthropic API key in `.env` (`CLASSIVORE_API_KEY` or `ANTHROPIC_API_KEY`)
- Brave Search API key in `.env` (`BRAVE_API_KEY`)
- Optional: Serper API key (`SERPER_API_KEY`) for search fallback

## Step 1: Prepare the Taxonomy

Create a directory under `taxonomies/` with:

```
taxonomies/
└── my_taxonomy/
    ├── config.yaml
    └── taxonomy.csv
```

### taxonomy.csv

A CSV with one row per category. Required columns:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier (string or integer) |
| `name` | Category name (used in labels and model output) |
| `parent_id` | ID of parent category (empty for root nodes) |

Optional columns:
- `description` — Will be populated by enrichment if blank
- `boundaries` — Will be populated by enrichment if blank
- `is_leaf` — `true`/`false` (computed automatically if missing)

The CSV can have any additional columns — classivore only reads what's mapped in config.yaml.

### config.yaml

Copy from `taxonomies/iab_2.2/config.yaml` and modify. Key sections:

```yaml
# Identity
name: "My Taxonomy"
version: "1.0"
slug: "my-taxonomy"

# Column mapping (must match your CSV headers)
taxonomy_file: "taxonomy.csv"
id_column: "id"
name_column: "name"
parent_column: "parent_id"

# Classification settings — tune after initial labeling
classification_type: "multi_label"   # or "single_label"
max_labels: 3                        # max categories per page
min_confidence: 0.5                  # label confidence threshold

# Enrichment — generates descriptions and boundaries
enrichment:
  model: "claude-haiku-4-5-20251001"
  max_tokens_per_category: 150

# Collection — web scraping targets
collection:
  target_per_category: 50            # labeled pages per category goal
  max_queries_per_category: 40       # upper bound on search queries
  max_per_domain_per_category: 50    # domain diversity cap
  commoncrawl_crawl_id: "CC-MAIN-2026-12"  # use latest crawl
  query_model: "claude-haiku-4-5-20251001"

# Labeling — two-stage hierarchical classification
labeling:
  model: "claude-haiku-4-5-20251001"
  stage1_max_tokens: 300
  stage2_max_tokens: 500
  tier1_confidence_threshold: 0.3
  temperature: 0.0
  text_truncation_words: 3000

# Training
model_base: "microsoft/deberta-v3-large"
batch_size: 8
learning_rate: 2.0e-5
max_length: 512
num_epochs: 3
focal_loss:
  alpha: 0.75
  gamma: 3.5
```

### Decisions you need to make

| Setting | Guidance |
|---------|----------|
| `classification_type` | `multi_label` if a page can belong to multiple categories, `single_label` if mutually exclusive |
| `max_labels` | 1 for single-label; 2-5 for multi-label depending on taxonomy breadth |
| `target_per_category` | Start with 50. Increase to 100+ for taxonomies with many similar categories |
| `min_confidence` | 0.5 is a safe default. Lower to 0.3 if your taxonomy has many overlapping categories |
| `excluded_categories` | List categories that are impossible or impractical to find content for |
| `excluded_tier1_categories` | Metadata categories that aren't content labels (e.g. "Content Language", "Content Source") |
| `domain_hints` | Known-good domains per tier-1 category — improves collection quality |

### Excluded categories

Add to config.yaml:

```yaml
# Metadata tier-1 categories (not content labels)
excluded_tier1_categories:
  - "Content Language"
  - "Content Source"

# Specific categories to skip (too niche, no findable content)
excluded_categories:
  - "Music: Soft AC Music"
  - "Music: Adult Album Alternative"
```

### Domain hints

Optional but valuable. Map tier-1 categories to known-good domains:

```yaml
domain_hints:
  Automotive:
    - "caranddriver.com"
    - "motortrend.com"
  Technology:
    - "techcrunch.com"
    - "arstechnica.com"
```

These are passed to LLM query generation for site-scoped searches.

## Step 2: Enrich the Taxonomy

Generate descriptions and boundary definitions for each category using LLM:

```bash
classivore enrich --taxonomy my-taxonomy --dry-run    # preview
classivore enrich --taxonomy my-taxonomy              # run enrichment
classivore enrich --taxonomy my-taxonomy --review      # interactive review
```

This creates `taxonomy_enriched.csv` alongside the original. All subsequent
commands use the enriched version automatically.

**Cost estimate:** ~$0.01 per category (Haiku batch). A 500-category taxonomy
costs ~$5.

**Why this matters:** Enriched descriptions and boundaries are used by:
- Template query generation (keyword extraction from descriptions)
- LLM query generation (passed as context)
- Labeling prompts (helps the LLM distinguish between similar categories)

## Step 3: Initial Collection

Run a small collection pass to build the initial corpus:

```bash
# Preview what will be collected
classivore collect --taxonomy my-taxonomy --queries-only -v

# Collect pages (shared corpus — benefits all taxonomies)
classivore collect --taxonomy my-taxonomy -v

# Check progress
classivore collect --taxonomy my-taxonomy --status
```

The collection pipeline generates **25-40 search queries per category** covering
multiple intent types. These are generated from the enriched taxonomy using
templates organized by search intent:

### Query template design

Templates are organized into intent categories to maximize content diversity:

**Informational:** `what is {name}`, `{name} explained`, `understanding {name}`,
`{name} overview`, `{name} for beginners`, `introduction to {name}`,
`history of {name}`

**Commercial/Comparison:** `best {name} options {year}`, `{name} reviews {year}`,
`{name} vs alternatives`, `top {name} recommendations`,
`{name} comparison guide`

**How-to/Practical:** `how to choose {name}`, `{name} tips and advice`,
`{name} best practices`, `{name} guide {year}`,
`getting started with {name}`, `{name} mistakes to avoid`,
`{name} checklist`

**News/Trends:** `{name} trends {year}`, `{name} news {year}`,
`future of {name}`, `{name} industry report {year}`,
`{name} statistics {year}`

**Definitional/Authoritative:** `{name} definition`, `types of {name}`,
`{name} categories explained`

**Long-tail/Specific:** `{name} case studies`, `{name} examples`,
`{name} research`, `{name} problems and solutions`

Additional templates use keywords extracted from the category's `description`
and `boundaries` fields to vary concept vocabulary beyond just the category
name. Tier-1 ancestor context is included to scope queries for deep categories.

All `{year}` tokens resolve to the current year at runtime.

### LLM query generation (iteration 2+)

When template queries are exhausted (typically after the first agent iteration),
the system falls back to LLM-generated queries. These are custom-crafted per
category using:

- Category name, description, and boundary definitions
- Sibling categories (to avoid overlap)
- All previously tried queries (to avoid repetition)
- Domain hints from config (enables site-scoped queries)
- How many more pages are needed

The LLM generates 5 queries per call, designed to find content the templates
missed by varying vocabulary, angle, and specificity. This costs ~$0.001 per
category (single Haiku call, not batch).

## Step 4: Label the Corpus

Run two-stage hierarchical labeling on all unlabeled corpus pages:

```bash
# Preview labeling scope
classivore label --taxonomy my-taxonomy --dry-run

# Run labeling (uses Anthropic Batch API — 50% discount)
classivore label --taxonomy my-taxonomy -v

# Check progress
classivore label --taxonomy my-taxonomy --status
```

**Two-stage approach:**
1. **Stage 1 (Tier-1 triage):** "Which top-level categories apply?" — cheap,
   filters irrelevant subtrees
2. **Stage 2 (Subtree classification):** "Which specific categories?" — only
   sends relevant subtrees, uses chain-of-thought reasoning

**Cost estimate:** ~$0.90 per 100 pages (Haiku batch, both stages).
A 25,000-page corpus costs ~$225.

**Important:** The labeling module seeds state from existing `labels.json`,
so previously labeled pages (including legacy imports) are not re-labeled.

## Step 5: Run the Agent Loop

The agent automates the collect→label→evaluate cycle, prioritizing categories
with the fewest labeled pages:

```bash
# See current coverage
classivore agent --taxonomy my-taxonomy --dry-run --target 50

# Run 3 iterations targeting 50 labels per category
classivore agent --taxonomy my-taxonomy --target 50 --max-iterations 3 -v

# Check agent history
classivore agent --taxonomy my-taxonomy --status
```

**Agent behavior:**
- Iteration 0: Uses template queries (free)
- Iteration 1+: Uses hybrid strategy (templates first, LLM queries when exhausted)
- Focuses collection on the worst gaps (fewest labeled pages first)
- Stops when: all categories at target, max iterations reached, or yield drops to zero

**Cost per iteration:**
- Collection: Free (template queries) + ~$0.50-2.00 (LLM queries on later iterations)
- Labeling: ~$0.90/100 new pages
- A typical iteration collecting ~1000 pages costs ~$10-15 total

## Step 6: Validate

Check label quality and distribution:

```bash
# Validate labeled data
classivore validate --taxonomy my-taxonomy --labeled

# Validate raw scraped data
classivore validate
```

Look for:
- **Coverage gaps** — Categories with zero or very few labels
- **Class imbalance** — Some categories dominating others
- **Unknown labels** — Category names in labels not found in taxonomy
- **Low confidence** — Many labels below the confidence threshold

## Step 7: Train

```bash
classivore train --taxonomy my-taxonomy
```

See `docs/claude/training.md` for the full training pipeline, artifacts produced, and how inference consumes them.

## Cost Summary

For a 500-category taxonomy targeting 50 labeled pages per category (25,000 total):

| Step | Cost |
|------|------|
| Enrichment | ~$5 |
| Collection | Free (search API within free tier) |
| Labeling | ~$225 |
| Agent LLM queries (3 iterations) | ~$5 |
| **Total** | **~$235** |

Costs scale linearly with category count and target pages per category.
Using Sonnet instead of Haiku for labeling would cost ~4x more.

## Reusing the Shared Corpus

The corpus (`data/corpus/pages.json`) is shared across all taxonomies.
When you add a second taxonomy:

1. Existing pages are immediately available for labeling — no re-scraping
2. The agent only collects pages for categories not well-represented
3. New pages benefit all future taxonomies

This is the key cost optimization: you scrape once and label many times.

## Troubleshooting

### Template queries exhausted quickly
The system generates 25-40 templates per category. If you need more, the
agent's second iteration uses LLM-generated queries automatically.

### Common Crawl CDX errors
The CDX index server occasionally goes down. Collection falls back to live
scraping automatically. Check status at https://index.commoncrawl.org/collinfo.json

### Search rate limiting
Brave free tier allows 15,000 queries/month (1 req/sec). The circuit breaker
pauses for 60s after 5 consecutive failures. Add a Serper API key
(`SERPER_API_KEY`) as a fallback provider.

### Labeling cost too high
- Lower `target_per_category` (30 is a reasonable minimum for training)
- Use `--stage 1` to run only the cheap triage pass first
- Check `--dry-run` before committing to a batch

### Categories with no findable content
Add them to `excluded_categories` in config.yaml. The agent will skip them.
