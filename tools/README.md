# tools/

Operational scripts that don't ship with the package. Run from the repo
root with the project venv active.

## scraper_bench.py

Live-scrape benchmark for evaluating the classivore web scraper. Produces
JSONL records (one per URL) so different runs can be sliced and compared on
the `label` field.

This is a research instrument, not part of the pipeline. It is **read-only**
against `DomainTracker` — it never calls `record_result()` or `save()`, so a
bench run can't mutate production domain quality scores.

### Quick start

```bash
source venv/bin/activate

# Smoke test, 10 URLs, two-host concurrency
python tools/scraper_bench.py run \
    --urls-file tools/fixtures/urls_top200.txt \
    --output tools/results/baseline_smoke.jsonl \
    --label baseline_smoke \
    --limit 10 \
    --concurrency 4 \
    --per-domain-max 2

# Full baseline against the 200-URL fixture
python tools/scraper_bench.py run \
    --urls-file tools/fixtures/urls_top200.txt \
    --output tools/results/baseline_local_$(date +%Y%m%d).jsonl \
    --label baseline_local \
    --concurrency 8 \
    --per-domain-max 2
```

### Optional: respect production domain blocklist

Pass `--domain-state-dir data/collection` to skip URLs whose host is
auto-blocked or manually blocklisted. The bench still records a row with
`outcome=domain_blocked` so the skip is visible in the JSONL.

### Subcommands

| Subcommand | Purpose |
|------------|---------|
| `run`      | Fetch URLs and emit JSONL records. |
| `stub`     | Print a notebook-entry skeleton to stdout. Use this when results came from a remote machine (Hetzner, residential proxy, etc.) and you just want to log them in `docs/scraper-notebook.md`. |

### JSONL record shape

| Field | Type | Notes |
|-------|------|-------|
| `ts` | str | ISO 8601 UTC. |
| `label` | str | Whatever you passed to `--label`. |
| `url` | str | |
| `host` | str | Lowercased hostname. |
| `outcome` | str | One of: `ok`, `empty_extraction`, `blocked`, `http_error`, `timeout`, `connection_error`, `domain_blocked`, `exception`. |
| `http_status` | int \| null | Response status; null if request never completed. |
| `bytes_received` | int \| null | `len(response.content)`. |
| `extracted_chars` | int \| null | `len(extract_text(html))`. |
| `fetch_ms` | float \| null | Wall time in fetch. |
| `extract_ms` | float \| null | Wall time in extract. |
| `total_ms` | float | Wall time across the URL. |
| `block_markers` | list[str] | Markers that fired (e.g. `just_a_moment`). |
| `fetch_failed_reason` | str \| null | Short tag (`timeout`, `http_403`, `domain_quality_blocked`, `upstream_429_aborted_run`, …). |
| `would_have_used_exa` | bool | True if production would have fallen through to Exa `/contents`. |
| `error` | str \| null | First 200 chars of `str(exc)`. |

### Block-marker heuristic

A response is labeled `blocked` when a marker fires *and* the body is small
(< 5KB), or when the body is so small (< 2KB) that there's nothing to
extract regardless. Marker hits in long bodies (> 50KB) are ignored — long
articles legitimately mention "captcha" or "access denied". The full
extraction is still attempted and `extracted_chars` is recorded; only the
`outcome` label is affected by the block decision.

### Per-host throttling

`--per-domain-max` caps concurrent fetches per host (default 2) on top of
the global `--concurrency` cap. After a host returns HTTP 429, all
remaining URLs for that host this run are short-circuited with
`outcome=http_error, fetch_failed_reason=upstream_429_aborted_run`.

### Headline metric

Live-scrape success = `outcome == "ok"`, which means fetch returned 200,
no block markers tripped the heuristic, and `extract_text` produced
≥ 100 chars. Anything below that threshold is `empty_extraction` — useful
to surface separately because it's where extractor (trafilatura/BS4)
improvements would help, vs. block_markers/http_error which are upstream
problems.

## TODO

- If a real circuit breaker is added later (state machine, faster recovery,
  reaction to 429 floods, half-open probes), grow the `outcome` enum at
  that time rather than retrofitting `domain_blocked`.
