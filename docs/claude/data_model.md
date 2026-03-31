# Data Model

## Shared Corpus Architecture

Scraped content is a shared resource across all taxonomies. You scrape once and label
many times. This saves significant collection cost and time when adding new taxonomies.

```
data/
├── corpus/
│   └── pages.json              # Shared scraped content (all taxonomies)
├── labels/
│   ├── iab-2.2.json            # IAB 2.2 labels (references corpus by URL)
│   ├── iptc-media.json         # IPTC labels for same corpus pages
│   └── google-product.json     # Etc.
└── reviewed/
    ├── iab-2.2.json            # Human-reviewed IAB labels
    └── iptc-media.json         # Human-reviewed IPTC labels
```

## Corpus Page Schema

```json
{
    "url": "https://example.com/article",
    "title": "Article Title",
    "text": "Extracted article text...",
    "word_count": 450,
    "content_hash": "a1b2c3d4...",
    "collected_at": "2026-03-31T12:00:00Z"
}
```

- `content_hash`: MD5 of text, used for dedup across URLs
- `collected_at`: ISO timestamp of when the page was scraped
- No taxonomy-specific fields — the corpus is taxonomy-agnostic

## Label Schema (per taxonomy)

```json
{
    "url": "https://example.com/article",
    "categories": [
        {"category": "Automotive: Green Vehicles", "category_id": "22", "confidence": 0.94}
    ],
    "reasoning": "This article discusses electric vehicle technology...",
    "labeled_at": "2026-03-31T14:00:00Z",
    "review_status": "unreviewed|accepted|edited|rejected"
}
```

Labels reference corpus pages by URL. The training step joins labels with corpus text.

## Workflow for New Taxonomy

1. **Reuse corpus** — existing pages are immediately available for labeling
2. **Label** — run `classivore label --taxonomy iptc-media` against the shared corpus
3. **Fill gaps** — run `classivore agent --taxonomy iptc-media` to find content for
   categories not well-represented in the existing corpus
4. **New pages go into shared corpus** — benefiting all future taxonomies

## Deduplication

Applied at collection time AND at migration:
- **URL dedup**: exact URL match
- **Content dedup**: MD5 hash of text (catches mirrors, CDN variants, redirects)

## Batch Labeling (Anthropic Batch API)

All labeling uses the Anthropic Message Batches API for 50% cost reduction:

- Submit up to 10,000 labeling requests per batch
- Each request is one page → one set of category labels
- Batches typically complete in <1 hour
- `custom_id` maps to corpus URL for result matching

Flow:
1. Build batch requests (one per unlabeled page)
2. Submit batch via `client.messages.batches.create()`
3. Poll for completion via `client.messages.batches.retrieve()`
4. Stream results via `client.messages.batches.results()`
5. Parse categories from each response, save to labels file

Cost comparison for 20,000 pages (Haiku):
- Standard API: ~$20
- Batch API: ~$10 (50% off)

## Tests

- `tests/unit/test_data_model.py` — test corpus loading, label joining, dedup logic
