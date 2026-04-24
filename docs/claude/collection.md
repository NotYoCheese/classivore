# Collection Subsystem

## Modules

- `src/classivore/collection/__init__.py` — Orchestrator. Iterates search providers, retrieves content through the fallback chain, applies filters, and appends to the shared corpus.
- `src/classivore/collection/search.py` — `SearchClient` with a configurable provider fallback chain: Brave → Serper → Exa. Also exposes Exa's `/contents` endpoint as a scrape fallback.
- `src/classivore/collection/commoncrawl.py` — CDX index lookup for exact URLs + WARC byte-range download. Used as a free retrieval layer before live scraping.
- `src/classivore/collection/scraper.py` — Live web scraping with trafilatura.
- `src/classivore/collection/filters.py` — Content dedup (MD5 hash), boilerplate detection (regex patterns), word count bounds, listing page detection.

## Data Flow

1. Agent or CLI requests URLs per category via `SearchClient` (Brave → Serper → Exa fallback).
2. For each result URL, retrieval walks a 4-stage fallback:
   1. **Prefetched text** — if the search result already carries page text (Exa's `search_and_contents`), skip scraping entirely and record `source="exa"`.
   2. **Common Crawl CDX** — if the URL is indexed, download just that WARC record.
   3. **Live scrape** — trafilatura against the live URL with browser-style headers.
   4. **Exa `/contents`** — when live scraping is blocked (WAFs, bot detection), fetch the page text via Exa's content endpoint. Also recorded as `source="exa"`.
3. All extracted text passes through `filters.py`.
4. Output: appended to `data/corpus/pages.json` (shared across taxonomies).

## Scraped Page Schema

```json
{
    "url": "https://example.com/article",
    "title": "Article Title",
    "text": "Extracted article text...",
    "word_count": 450,
    "source": "commoncrawl|live_scrape|exa",
    "collected_at": "2026-03-31T12:00:00Z"
}
```

## Content Filters

Applied at collection time (not retroactively):
- **Content-hash dedup:** MD5 of text, rejects identical content from different URLs
- **Boilerplate detection:** Regex patterns matching cookie walls, registration gates, error pages. Only applied to pages <300 words.
- **Word count bounds:** 150 minimum, 50,000 maximum. Truncate >2,000 to 1,000.
- **Unique word ratio:** Reject if <30% unique words (listings/repetitive content)

## Common Crawl

- CDX API supports exact URL lookup (not just wildcards)
- WARC download uses HTTP byte-range requests — downloads only the specific record
- Rate limiting: 0.5s between CDX queries, 0.5s between WARC downloads
- Crawl ID configured in taxonomy config.yaml, defaults to latest

## Tests

- `tests/unit/test_collection_search.py` — provider dispatch, fallback chain, Exa search + `/contents` paths
- `tests/unit/test_collection_orchestrator.py` — prefetched-text fast path, Exa scrape fallback, `source` field recording
- `tests/unit/test_collection_filters.py` — dedup, boilerplate detection, word count
- `tests/unit/test_collection_commoncrawl.py` — CDX response parsing, URL construction (mock HTTP)
