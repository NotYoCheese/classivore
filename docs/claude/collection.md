# Collection Subsystem

## Modules

- `src/classivore/collection/commoncrawl.py` — CDX index lookup for exact URLs + WARC byte-range download. Used as free retrieval layer before live scraping.
- `src/classivore/collection/scraper.py` — Live web scraping with trafilatura. Fallback when Common Crawl doesn't have the URL.
- `src/classivore/collection/filters.py` — Content dedup (MD5 hash), boilerplate detection (regex patterns), word count bounds, listing page detection.

## Data Flow

1. Agent or CLI provides URLs (from Brave Search or domain patterns)
2. For each URL: check Common Crawl CDX → if found, download WARC record
3. URLs not in CC → live scrape with trafilatura
4. All extracted text passes through filters.py
5. Output: `data/scraped_pages.json`

## Scraped Page Schema

```json
{
    "url": "https://example.com/article",
    "title": "Article Title",
    "text": "Extracted article text...",
    "word_count": 450,
    "source": "commoncrawl|live_scrape",
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

- `tests/unit/test_filters.py` — test dedup, boilerplate detection, word count
- `tests/unit/test_commoncrawl.py` — test CDX response parsing, URL construction (mock HTTP)
- `tests/integration/test_scraper.py` — test live scraping against known URLs
