# Data Expansion Agent

## Modules

- `src/classivore/agent/agent.py` — LangGraph workflow: analyze → search → scrape → label → check progress
- `src/classivore/agent/strategies.py` — Search strategy per taxonomy (template queries, LLM queries)

## Workflow

```
analyze_categories → should_continue? → search_for_content → scrape_content → label_content → check_progress → loop or end
```

## Key Behaviors

- Loads full taxonomy to find categories with 0 examples (not just categories already in labeled data)
- Filters search URLs against existing scraped + labeled URLs
- Checks Common Crawl before live scraping
- Passes --data_dir through to all subprocess calls
- Uses template queries (free) for first attempt, LLM for retries
- Tracks per-category examples added and target category hit rate
- Configurable via taxonomy config.yaml (target_count, max_search_attempts, excluded_categories)

## Search Strategy

Defined in strategies.py, selected by taxonomy config:

- `template` — Simple keyword queries from category name + description. Free.
- `llm` — Claude generates creative queries. ~$0.01-0.03 per query.
- `hybrid` — Template first, LLM on retry. Default.

## Dependencies

- LangGraph for workflow orchestration
- Brave Search API for URL discovery (BRAVE_API_KEY env var)
- Common Crawl for free retrieval
- iab-scrape / iab-label CLI commands (or classivore equivalents)

## Tests

- `tests/unit/test_agent.py` — test state transitions, category analysis, URL filtering (mock search/scrape)
- `tests/unit/test_strategies.py` — test query generation for various category types
