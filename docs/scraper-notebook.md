# Scraper improvement notebook

A research log for the live-scrape pipeline. Each entry records what was
tried, the measured impact against the bench, and the decision that
followed (ship, drop, defer). Append-only — keep failed experiments in
place so we don't reinvestigate them later.

The bench tool that produces the metrics referenced below is
`tools/scraper_bench.py`. Raw JSONL outputs land in `tools/results/`
(gitignored except for `.gitkeep`).

## Format

Each entry follows this template. Copy it (or use `python tools/scraper_bench.py stub --label <name> ...`) when adding a new one.

```
## <label> — YYYY-MM-DD

**Environment:** local / Hetzner / residential proxy / curl_cffi / etc.
**URL set:** path or description (e.g. tools/fixtures/urls_top200.txt)
**Concurrency / per-domain:** N / M

### Metrics
- success rate (ok / total): X / Y = Z%
- median fetch_ms: …
- p95 fetch_ms: …
- blocked: …
- empty_extraction: …
- http_error breakdown: 403=…, 429=…, 5xx=…

### Observations
- bullets

### Hypothesis tested
- what we believed before the run
- what the run says about that belief

### Next step
- ship / drop / defer — and why
```

---

## Entries

<!-- Newest entries on top. Append above this line. -->

_(no runs yet — the first baseline goes here)_
