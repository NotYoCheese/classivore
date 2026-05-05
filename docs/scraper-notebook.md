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

## baseline_hetzner — 2026-05-05

**Environment:** Hetzner Cloud (Falkenstein), bare `python3 tools/scraper_bench.py run` over the same scraper config as local. IP `178.156.229.253/32` (datacenter ASN).
**URL set:** `tools/fixtures/urls_top200.txt` (252 URLs, identical to baseline_local).
**Concurrency / per-domain:** 8 / 2.

### Metrics
- success rate (`ok` / total): **99 / 252 = 39.3%** (vs 37.3% local — +2 pp).
- median fetch_ms: **291** (vs 178 local), p95: **1422**.
- median extract_ms: **261** (vs 46 local), p95: **3135** (vs 826 local).
- outcomes: `ok` 99, `empty_extraction` 96, `http_error` 56, `blocked` 1, `timeout` 0.
- HTTP status: 200×196, 202×4, 401×2, **402×11**, 403×32, 406×1, **429×6**.

### Side-by-side vs baseline_local

| | Local M1 | Hetzner | Delta |
|---|---:|---:|---:|
| ok | 94 | **99** | +5 |
| empty_extraction | 105 | **96** | -9 |
| http_error | 48 | 56 | +8 |
| timeout | 5 | **0** | -5 |
| 200 responses | 200 | 196 | -4 |
| 403 | 39 | 32 | -7 |
| 402 | 0 | **11** | +11 |
| 429 | 1 | 6 | +5 |
| same-URL agreement | — | — | **77%** |

- **22 hosts 403 in both runs.** These are fingerprint/header-determined blocks — neither IP type gets through. This is the curl_cffi target list.
- **17 hosts 403 only on local.** Mostly Dotdash Meredith properties (people.com, ew.com, allrecipes, seriouseats, foodandwine, marthastewart, realsimple, thespruce, health, mayoclinic, …). On Hetzner those same properties return **HTTP 402** instead — same Akamai gatekeeper, different status by network class. Net effect is identical (no body), so 402 is just 403's twin under a different ASN.
- **10 hosts 403 only on Hetzner.** reddit, economist, gap, oldnavy, tmz, deviantart, giphy, fodors, niemanlab, rome2rio. These ASN-deny-list datacenter ranges; M1 from a residential connection sails through.
- **Hetzner is CPU-bound on extract.** trafilatura/lxml is 5× slower at p95 (3.1 s vs 0.8 s) on a Hetzner shared core.
- **One segfault on the first run** (lxml-shaped, mid-batch around 140/252) didn't repeat on the second run. Bench's per-record `flush()` meant no records were lost. Filed as noise.

### Hypothesis tested
- *Belief:* "Datacenter IP egress will materially change 403 count — datacenters are denylisted everywhere."
- *Result:* Largely refuted. 403 count actually *dropped* (39 → 32) on Hetzner, but new failure modes (402, 429) brought `http_error` higher. **77% of URLs produce identical outcomes regardless of IP.** The remaining 23% flips in both directions: 21 URLs went from not-ok → ok on Hetzner; 16 went the other way. **There is no clean IP-reputation winner — different services run different rules.**

### Next step
- **curl_cffi gets prioritized.** With IP-rotation off the table as a one-shot fix, TLS fingerprint mimicry against the 22-host "fingerprint-determined" overlap set is the highest-leverage move. Worth a spike against just those 22 hosts (small enough to verify by hand).
- **The 11 Dotdash 402s and 17 Dotdash 403s are the same block** — fold them into one entry in any future per-host analysis.
- **Empty extraction is still the biggest single failure mode** (96 + 105 across both runs). The third-extractor experiment (readability-lxml or jusText) is still the highest single-improvement-by-LOC bet.
- **Don't run Hetzner for routine bench iterations.** It's slower, identical to within 2 pp on success rate, and doesn't isolate anything we couldn't measure locally. Reserve it for "does this fix work cross-egress" verification on specific changes.

---

## baseline_local — 2026-05-04

**Environment:** local M1 Max MacBook, plain `requests` with rotating UA + browser headers (current production scraper, unmodified).
**URL set:** `tools/fixtures/urls_top200.txt` (252 URLs — file has more than the 200 the header advertises; left as-is to match the load-test fixture exactly).
**Concurrency / per-domain:** 8 / 2.

### Metrics
- success rate (`ok` / total): **94 / 252 = 37.3%**
- median fetch_ms: **178**, p95: **1230**, max: **20153**
- median extract_ms: **46**, p95: **826**
- outcomes:
  - `ok` 94 (37.3%)
  - `empty_extraction` 105 (41.7%)
  - `http_error` 48 (19.0%)
  - `timeout` 5 (2.0%)
- HTTP status breakdown: 200×200, 202×5, 401×2, 403×39, 429×1
- block markers seen: `captcha_present` 41× (35 of those still extracted to `ok` — long article bodies that *mention* "captcha"; 6 fell to `empty_extraction`), `access_denied` 1×.

### Observations
- **Empty extraction is the dominant failure mode, not blocks.** 105 of 252 URLs fetched HTTP 200 with substantial bodies (median **71 KB**, max 3.8 MB) but produced < 100 chars after trafilatura → BS4 fallback. Only 6 of those 105 had any block marker. These are real pages — they're just JS-rendered, paywalled, or structured in ways the extractors can't handle.
- **403s are concentrated on known anti-bot domains.** The 39 forbidden hosts are a who's-who: Bloomberg, FT, Politico, Etsy, Adidas, Quora, Medium, Tripadvisor, Mayo Clinic, Bon Appetit-tier recipe sites, Bauer/Meredith lifestyle sites. Likely Akamai/Cloudflare bot-management against the bare `requests` TLS fingerprint.
- **The block-marker heuristic behaved correctly.** 41 `captcha_present` hits, but 35 of those produced ≥ 100 chars of text and were correctly labeled `ok`. The size guard (don't trust markers in bodies > 50 KB) is doing its job — these are long articles that incidentally mention "captcha".
- **Tail latency is meaningful.** p95 fetch is 1.2 s and max is 20 s (we have a 20 s timeout). The 5 timeouts are all hitting the ceiling.
- **Throttling held.** No 429s besides one, no host produced a 429-aborted-run cascade.

### Hypothesis tested
- *Belief:* "Most failures are bot-detection 403s; mimicking TLS fingerprint with curl_cffi would close the gap."
- *Result:* Even if curl_cffi recovered every single 403, success would only go from 37% → ~53%. The bigger lever is the 105 empty-extraction pages — those are extractor-shaped, not network-shaped.

### Next step
- **Defer** curl_cffi as the first improvement. It addresses ≤ 19 pp of headroom.
- **Try** a third extraction fallback (readability-lxml or jusText) before BS4. If it claws back even half the empty-extraction tail, that's +20 pp on success rate at the cost of one dependency.
- **Investigate** whether the empty-extraction pages are JS-rendered (a quick check: do their bodies contain `<noscript>` or empty `<main>` shells?) — if so, headless browser is the real answer for that subset, not better extractors.
- Open question for a later run: does residential-IP egress (Hetzner/proxy) move the 403 count, or is it purely TLS fingerprint? Hetzner-IP run with the same `requests` config would isolate that.


