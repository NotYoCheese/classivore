# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.4.0] - 2026-05-08

### Reliability

- `batch.submit_batch` retries 3× with exponential backoff (1s, 2s, 4s)
  on `APIConnectionError`, `RateLimitError`, and 5xx `APIStatusError`.
  4xx other than 429 propagate immediately. Previously a single transient
  failure on submit would abort a long enrichment run.
- `batch.poll_until_complete` now bounds wall-clock time at 24h via
  `time.monotonic()` and raises `BatchPollTimeoutError(batch_id,
  elapsed_seconds)` referencing `label_state.json` for resume. Replaces
  the unbounded poll loop that could hang forever on a stuck batch.
- `TaxonomyConfig` validates required fields at load time
  (`name`, `version`, `slug`, `taxonomy_file`, `id_column`, `name_column`)
  and raises `ValueError` naming the offending config path and missing
  field. Catches typos in `taxonomies/*/config.yaml` before they crash
  mid-run.
- Atomic JSON writes across `training/` and `labeling/` via
  `persistence.atomic_json_save` and the new `persistence.atomic_writer`
  context manager. Partial files can no longer corrupt artifacts on
  crash mid-write (label state, training reports, threshold files,
  taxonomy metadata).

### CLI

- Drop the unimplemented `classivore serve` command. Was a stub that
  printed a TODO; the API server lives in the `classivore-api` companion
  repo.
- Fix `classivore enrich --review` to actually accept input. Previously
  printed each enrichment without prompting, making the flag a no-op.
- Split `cli/main.py` (1230 → 549 lines) by extracting classify, init,
  taxonomy, enrich, agent, and hints into `cli/runners/`. Heavy imports
  (torch, transformers) stay deferred per command. No user-visible
  surface change.
- Consolidate 11 copies of the "load enriched taxonomy if exists"
  pattern into `taxonomy.loader.apply_enriched_if_present` and
  `enriched_taxonomy_path` helpers.

### Training

- `train_model` now passes `use_cpu=(device == "cpu")` to
  `TrainingArguments` so `device="cpu"` is honored on Macs. Without it,
  HF Trainer auto-selected MPS for the model while `class_weights` and
  `loss_fn` stayed on CPU, breaking the first forward pass.

### Tests

- New `tests/unit/test_trainer.py` runs `train_model` on a 50-row toy
  taxonomy with `deberta-v3-xsmall`, asserting the full inference
  artifact set lands on disk. Skips cleanly when the model is uncached
  and offline.
- New `tests/unit/test_batch.py` covers the retry and timeout paths
  added to `batch.submit_batch` and `batch.poll_until_complete`.

### Collection

- `fetch_page` now accepts an optional `session` parameter for caller-controlled
  HTTP behavior (proxies, retries, timeouts, connection pooling, custom
  adapters, TLS fingerprinting). When `session=None` (default), behavior is
  unchanged — module-level `curl_cffi.requests` with Chrome 131 impersonation.
  Proxy management has moved out of this library: classivore exposes scraping
  primitives, the operator (e.g. `classivore-api`) configures the transport.
- Replace `requests` with `curl_cffi` impersonating Chrome 131 in the live
  scraper. TLS handshake, HTTP/2 settings, and header order now match a real
  browser, getting us past Akamai/Cloudflare WAFs that fingerprint plain
  `requests`. Bench measured **+39.3 pp** live-scrape success on a frozen
  252-URL fixture (37.3% → 76.6%); 31 of 39 previously-403 hosts recovered.
  See `docs/scraper-notebook.md` for the full diff. New dependency:
  `curl_cffi>=0.7.0`.
- Drop the manual `BROWSER_HEADERS` dict and `USER_AGENTS` rotation from
  `scraper.py`. Both pre-dated the curl_cffi swap and were overriding
  curl_cffi's auto-injected Chrome 131 header set — including replacing the
  matching macOS UA with Linux/Windows strings on 2/3 of requests, which
  WAFs cross-check against the macOS-shaped TLS handshake. Bench: **+1.6 pp**
  local (76.6% → 78.2%) and **+2.3 pp** Hetzner (68.3% → 70.6%); 403s
  drop 8→3 local and 28→24 Hetzner. Recovers etsy, inc.com, barrons on
  both machines plus hm/reddit/threads/thekitchn locally and
  walmart/kohls/sportingnews/smittenkitchen on Hetzner.

### Labeling

- Add `--limit N` flag to `classivore label` for cost-bounded sampling. Caps
  each stage at N pages this run; subsequent runs naturally pick up where
  the previous run left off via the existing label state. Useful for
  baselining cost on a subset before committing to the full corpus, or for
  spreading a large labeling job across multiple smaller batches.

### Taxonomy

- Fix `classivore init` to compute `path`, `depth`, `is_leaf`, and
  `children_count` from the `parent_id` graph and write a normalized
  `taxonomy.csv`. Previously `init` did `shutil.copy2` of the input CSV
  verbatim, so taxonomies onboarded from a raw three-column source (id,
  parent_id, name) ended up with every category looking like a depth-1
  leaf to the loader, breaking enrichment, collection, and training.

### Tools

- Add `tools/scraper_bench.py` — a live-scrape benchmark for evaluating
  scraper improvements across environments (transport options, header
  strategies, network conditions, etc.) against a frozen 200-URL fixture. Emits one JSONL record per URL with
  outcome, timings, block markers, and a `would_have_used_exa` flag so
  runs can be sliced and compared on the `--label` field. Read-only
  against `DomainTracker` (it never calls `record_result()` or `save()`).
- Add `docs/scraper-notebook.md` — append-only research log so failed
  experiments stay documented and don't get re-investigated.
- Refactor `scraper.fetch_page` to delegate the actual HTTP call to a new
  `_fetch_response(url)` helper. The bench reuses the helper to inspect
  status codes, content-type, and response bodies on non-200s while
  keeping `fetch_page`'s public behavior identical.

## [1.3.0] - 2026-04-28

### Training

- Fix `split_data` to actually stratify train/val/test by label. Previous
  versions computed a stratification key but never used it, falling back to
  a plain random shuffle. This left thin-tail categories with 0–2 test
  samples, making per-category F1 statistically meaningless. Now uses
  multi-label iterative stratification (Sechidis et al. 2011) via
  `iterative-stratification`. New dependency: `iterative-stratification>=0.1.7`.

### Labeling

- Add opt-in prompt caching for stage-1 and stage-2 system prompts via
  `cache_control` blocks with 1-hour TTL. Disabled by default — caching
  only pays off when reads-per-write exceeds break-even and the cached
  block clears the model's minimum cacheable token threshold (2,048 for
  Haiku 4.5). Toggle via `labeling.prompt_cache` config or
  `--prompt-cache / --no-prompt-cache` CLI flag.
- Token-usage accounting tracks `cache_creation_input_tokens` and
  `cache_read_input_tokens` per batch with aggregated cache hit rate.

### Collection

- Add Exa AI as a third search provider with neural/semantic fallback when
  Brave and Serper are exhausted, plus scrape fallback via Exa `/contents`
  when live scraping is blocked. Exa results include prefetched page text,
  letting the collector skip scraping for those URLs.
- Fall through the search cascade on Serper 400 responses (returned when
  credits are exhausted — Serper does not use 401/402/403 for quota).
- Drop literal parent-name injection from collection query templates;
  parent context now flows through descriptions instead.

### Agent

- Add per-run statistics tracking with NDJSON history under `data/agent_runs/`.
- Stop iteration-counting only includes successful zero-yield iterations;
  errored iterations no longer falsely trigger the stop condition.
- Surface labeling/training metrics on failure and audit silent exception
  swallowing in the agent loop.

## [1.2.2] - 2026-04-24

### Inference

- Fix `IndexError: index out of range in self` when classifying long input
  with RoBERTa-family models (RoBERTa, XLM-RoBERTa, CamemBERT, Longformer,
  XMOD). These models offset `position_ids` by `pad_token_id + 1`, so a
  config of `max_position_embeddings=514` only addresses 512 input tokens.
  `Classifier` now detects the offset from `config.model_type` and clamps
  `max_length` accordingly. BERT, DeBERTa, and other non-RoBERTa-family
  models are unaffected.

## [1.2.1] - 2026-04-21

### Inference

- Force eager weight loading (`low_cpu_mem_usage=False`) in `Classifier` to
  avoid a meta-tensor race under thread-pool concurrency. Previously, multiple
  threads loading the same model could leave some parameters on the meta
  device, causing `.to(device)` to raise
  `NotImplementedError: Cannot copy out of meta tensor; no data!` — observed
  in the classivore-api container on linux/amd64. No accuracy change.

## [1.2.0] - 2026-04-21

### Inference

- `Classifier` now dispatches the output activation based on the model's
  `problem_type` in `config.json`:
  - `multi_label_classification` → sigmoid (existing behavior)
  - `single_label_classification` → softmax (probabilities sum to 1 per row)
  - `regression` → raises `NotImplementedError`
  - missing / null → defaults to multi-label (preserves legacy behavior)
- Default per-category threshold floor is now `0.0` for single-label models
  (so softmax results aren't silently suppressed by the multi-label `0.5` default).
  Multi-label models still default to `0.5`. Per-category threshold files
  always take precedence when present.
- Allows dropping in off-the-shelf HuggingFace single-label classifiers
  (e.g. sentiment, toxicity) alongside classivore-trained multi-label models.

### Collection

- Added Exa AI as a third search provider with two roles:
  - Neural/semantic search fallback when Brave and Serper are exhausted
  - Scrape fallback via Exa `/contents` when live scraping is blocked (WAFs, etc.)
- Exa search results include prefetched page text, so the collector skips the
  scrape step entirely for those URLs.

### Docs

- Expanded README with a full "API Keys & External Services" section
  documenting every external service (Anthropic, Brave, Serper, Exa,
  HuggingFace, Common Crawl), which commands use each one, where to get keys,
  and which features are enabled by each.

## [1.0.0] - 2026-04-06

First release. Full pipeline from taxonomy enrichment to trained classifier.

### Pipeline

- **Enrich** — LLM-generated descriptions and boundaries for taxonomy categories
  via Anthropic Batch API
- **Collect** — Web scraping with Brave/Serper search, Common Crawl CDX lookup,
  content filtering, domain quality tracking, circuit breaker
- **Label** — Two-stage hierarchical LLM labeling (tier-1 triage + subtree
  classification) via Anthropic Batch API with crash recovery
- **Agent** — Automated collect-label-evaluate loop with gap prioritization,
  per-category targets, and stop conditions
- **Validate** — Data quality checks via label-lens integration
- **Train** — DeBERTa-v3-large fine-tuning with weighted focal loss,
  confidence-weighted training, per-category threshold optimization, and
  comprehensive quality reporting

### Infrastructure

- Structured logging via structlog with JSON output
- Atomic JSON persistence for all state files
- Typed exception hierarchy
- Shared data models (dataclasses)
- 460+ unit tests

### Training Results (IAB 2.2)

- 27,102 training pages across 605 leaf categories
- F1 micro: 0.68 (val), 0.65 (test)
- F1 macro: 0.65 (val), 0.62 (test)
- Per-category threshold optimization: +5% F1 macro
- Training time: 43 minutes on RTX 4090

### Key Technical Decisions

- Focal loss alpha=0.75, gamma=3.5 (validated over 25+ training iterations)
- Class weight cap at 7.0 without normalization
- Legacy label confidence discount (0.75 weight)
- Non-leaf labels excluded from training
- One target, one source of truth (labels.json) for the agent loop
