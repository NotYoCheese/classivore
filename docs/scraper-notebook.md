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

## header_bundles_skipped — 2026-05-05

**Status:** experiment discarded pre-emptively. **No bench runs performed.**

**Hypothesis to test:** real Chrome sends a coordinated bundle of HTTP headers (`Sec-Ch-Ua`, `Sec-Ch-Ua-Mobile`, `Sec-Ch-Ua-Platform`, `Sec-Fetch-Dest/Mode/Site/User`, `Upgrade-Insecure-Requests`, plus consistent `Accept`/`Accept-Language`/`Accept-Encoding`). A scraper sending only `User-Agent` looks bot-like even with a perfect TLS fingerprint. Rotating internally-consistent header bundles per request might recover a few more URLs.

**Pre-condition check (run before the experiment):** does `curl_cffi` already set these automatically when `impersonate="chrome131"` is configured?

```python
from curl_cffi import requests
resp = requests.get("https://httpbin.org/headers", impersonate="chrome131")
# echoed headers contain everything in scope:
#   Accept: text/html,application/xhtml+xml,...,application/signed-exchange;v=b3;q=0.7
#   Accept-Encoding: gzip, deflate, br, zstd
#   Accept-Language: en-US,en;q=0.9
#   Priority: u=0, i
#   Sec-Ch-Ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"
#   Sec-Ch-Ua-Mobile: ?0
#   Sec-Ch-Ua-Platform: "macOS"
#   Sec-Fetch-Dest: document
#   Sec-Fetch-Mode: navigate
#   Sec-Fetch-Site: none
#   Sec-Fetch-User: ?1
#   Upgrade-Insecure-Requests: 1
#   User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ... Chrome/131.0.0.0 Safari/537.36
```

### Result
**Every header named in the experiment scope is already injected by curl_cffi** when `impersonate="chrome131"` is set, plus `Priority: u=0, i` (HTTP/2 priority frame) which the experiment hadn't called out. The chrome131 impersonation profile is internally consistent by construction (Sec-Ch-Ua brand list matches the major version in User-Agent matches the platform claim in Sec-Ch-Ua-Platform).

### Verdict
**Discard pre-emptively** per the experiment's verdict criteria. Manually constructed `HEADER_BUNDLES` would at best duplicate what curl_cffi already does, and at worst drift from chrome131's actual on-the-wire header set as Chrome itself updates over time (curl_cffi's profile gets re-scraped from real Chrome periodically; a hardcoded dict in `headers.py` would not).

No code change. No bench runs.

### Incidental finding — header overrides may be undermining impersonation

The header dump showed that the scraper's existing `BROWSER_HEADERS` dict and `USER_AGENTS` rotation — both pre-dating the curl_cffi swap — are **explicitly overriding** curl_cffi's auto-injected headers in three ways that plausibly hurt impersonation fidelity:

1. **UA/TLS mismatch on ~2/3 of requests.** curl_cffi's TLS handshake fingerprints as "Chrome 131 on macOS"; the `USER_AGENTS` rotation replaces curl_cffi's matching UA with Windows NT or Linux x86_64 strings 2 times out of 3. WAFs cross-check TLS-OS vs UA-OS as a tell.
2. **Weaker `Accept` / `Accept-Encoding`.** Our hardcoded values miss `image/apng`, `application/signed-exchange;v=b3;q=0.7`, and `zstd` — all things real Chrome 131 sends.
3. **Adds `DNT: 1`.** Chrome 131 doesn't send DNT by default; only the small subset of users who toggled "Do Not Track" do. Sending it is itself a fingerprint.

The +39 pp curl_cffi win held *despite* these mismatches. Cleaning them up is its own experiment with its own one-axis change ("remove explicit headers and UA rotation, let curl_cffi's chrome131 profile stand"). Filed as a follow-up. **Not bundled with this entry** — the experiment-spec rule of "one axis only" applies to follow-ups too.

### Next step
- Greenlight a separate experiment: drop `BROWSER_HEADERS` and `USER_AGENTS` rotation from `scraper.py`, let curl_cffi's chrome131 profile own the entire header set. Quantitative target: small (1–3 pp) since the +39 pp lift already happened despite the override; the question is whether residual http_errors carry a UA/TLS-mismatch signal we can clean up.

---

## curl_cffi_hetzner — 2026-05-05

**Environment:** Hetzner Cloud (Falkenstein, datacenter ASN), `curl_cffi.requests` impersonating `chrome131`. Same code as `curl_cffi_local`; only egress IP differs.
**URL set:** `tools/fixtures/urls_top200.txt` (252 URLs, identical to all prior runs).
**Concurrency / per-domain:** 8 / 2.

### Metrics
- success rate: **172 / 252 = 68.3%** (vs `baseline_hetzner` 39.3% — **+29.0 pp**; vs `curl_cffi_local` 76.6% — **-8.3 pp**).
- median fetch_ms: 276, p95 1423.
- median extract_ms: 1184, p95 3806 (Hetzner CPU is slower; extract is the dominant cost on this hardware).
- outcomes: `ok` 172, `empty_extraction` 36, `http_error` 40, `blocked` 3, `timeout` 1.
- HTTP status: 200×211, 202×3, 401×3, **403×28**, **429×6**.

### The 2×2 — fingerprint × IP

|                | baseline | curl_cffi | row Δ |
|----------------|---------:|----------:|------:|
| **local M1**   |    37.3% | **76.6%** | +39.3 |
| **Hetzner DC** |    39.3% |     68.3% | +29.0 |
| **IP Δ**       |  +2.0 pp |  -8.3 pp  |       |

- **Fingerprint is the dominant lever in both IP environments.** +29–39 pp regardless of egress.
- **IP reputation is real but second-order.** With plain `requests`, the local→DC IP delta is +2 pp (essentially noise). With `curl_cffi`, that same IP delta is -8.3 pp. Once the easy WAF gates are passed, IP becomes the next gate.
- **The two levers are not independent.** Some hosts gate on (fingerprint AND IP); they only flip to ok when both are clean.

### Hetzner: baseline → curl_cffi (IP constant)

- 76 URLs recovered via fingerprint mimicry alone (no IP change).
- 3 regressions.
- Net: +73 URLs.

### local vs Hetzner under curl_cffi (fingerprint constant)

- Same-URL agreement: **87.3%**.
- 24 URLs work on local-curl_cffi but fail on Hetzner-curl_cffi — **the IP-reputation-bound set:**
  Bloomberg, Politico, Etsy, Stack Overflow, Reddit, Gap, Old Navy, Kohl's, Petco, Quora, Economist, Axios, The Hill, B&H Photo, Apartment Therapy, GameSpot, Science.org, Smitten Kitchen, Kotaku, DeviantArt, Fodor's, Rome2Rio, TMZ, plus Walmart/Best Buy/Kayak (200-but-empty under DC IP).
- 3 URLs work on Hetzner-curl_cffi but not on local-curl_cffi — too small to interpret; noise.

### Persistent issues on Hetzner under curl_cffi

- **28 hosts still 403** — superset of the 8 still-403 hosts on local. Adds the IP-bound set above.
- **6 hosts return 429** (Wayfair, Chewy, imgur, CBS Sports, stackshare, Star Tribune). Same set as `baseline_hetzner`. ASN-rate-limited regardless of fingerprint.
- **3 `blocked` markers fired** vs 2 locally — Cloudflare interstitials still served to DC IPs even with the right fingerprint.

### Hypothesis tested
- *Belief:* "curl_cffi handles fingerprint-based gates; IP reputation is a separate, smaller axis."
- *Result:* Confirmed. The 2×2 cleanly separates the two effects. **Fingerprint is worth ~+29–39 pp; IP is worth ~+8 pp on top of fingerprint** (and ~+2 pp without it). The interaction term is real: IP gating only becomes visible once fingerprint gating is removed.

### Implications for production deployment
- **Production runs from Hetzner today.** Expected lift after deploying #32 is closer to +29 pp than +39 pp. Still substantial.
- **The 24-URL IP-bound set is the next frontier.** Three options, ranked by cost:
  1. **Residential-proxy-on-demand** for the IP-bound host set — egress only those 24 hosts through a proxy, leave everything else on the DC IP. Lowest cost, focused leverage.
  2. **Run scraping from a residential VPN/co-located link.** Affects all egress, including non-target hosts.
  3. **Headless browser (Playwright)** for the residual 8 hosts that resist both fingerprint and IP measures.
- **Don't pursue (3) yet.** It's expensive (browser overhead, infra) and the residual is small. Revisit if those 8 hosts move into production-priority categories.

---

## curl_cffi_local — 2026-05-05

**Environment:** local M1 Max MacBook. Swapped `requests` → `curl_cffi.requests` impersonating `chrome131`. UA strings, browser headers, timeout, cookie-banner stripping, trafilatura/BS4 extraction all unchanged. One-line behavioral change.
**URL set:** `tools/fixtures/urls_top200.txt` (252 URLs, identical to baselines).
**Concurrency / per-domain:** 8 / 2.

### Metrics
- success rate (`ok` / total): **193 / 252 = 76.6%** (vs 37.3% baseline_local — **+39.3 pp**).
- median fetch_ms: 252 (vs 178 baseline), p95: 1046.
- median extract_ms: **567** (vs 46 baseline), p95: 1776.
- outcomes: `ok` 193, `empty_extraction` 42, `http_error` 15, `blocked` 2, `timeout` 0.
- HTTP status: **200×237** (vs 200 baseline), 202×2, 401×3, 403×8 (vs 39), 429×2.

### Side-by-side vs baseline_local

| | baseline | curl_cffi | Δ |
|---|---:|---:|---:|
| ok | 94 | **193** | **+99** |
| empty_extraction | 105 | 42 | -63 |
| http_error | 48 | 15 | -33 |
| timeout | 5 | 0 | -5 |
| 403 responses | 39 | 8 | **-31** |
| 200 responses | 200 | 237 | +37 |
| same-URL agreement | — | 54.0% | — |
| flipped TO ok | — | **104** | — |
| flipped FROM ok | — | 5 | — |

### Where the +99 ok URLs came from

- **69 from `empty_extraction` → `ok`.** These were Cloudflare-style interstitials all along — large HTML bodies that contained "Just a moment…" or challenge JavaScript with no extractable article text. We had been mis-classifying them as extractor failures. Real articles now.
- **30 from `http_error` → `ok`.** Direct 403 → 200 wins on Bloomberg, Politico, FT, Adidas, Macy's, Quora, Medium, every Dotdash Meredith property, etc.
- **5 from `timeout` → `ok`.** Fingerprint-based blocks were dropping the connection rather than rejecting cleanly; curl_cffi handshakes through.

### 22-host fingerprint-determined block list (from baseline_hetzner cross-reference)

The hypothesis from the Hetzner run was that 22 hosts 403 in *both* IP environments — those should be the curl_cffi target. The data confirms it:

- **31 of 39 baseline_local 403 hosts recovered.** Includes all 22 hosts on the cross-IP "fingerprint-determined" overlap list, plus 9 more that 403'd locally only.
- **8 hosts still 403 with curl_cffi:** `etsy.com`, `inc.com`, `petco.com`, `skynews.com`, `surfer.com`, `thekitchn.com`, `tripadvisor.com`, `www2.hm.com`. Likely behavioral signals beyond TLS — JS execution, prior session cookies, mouse-movement telemetry — and not addressable by impersonation alone. Filing this as the curl_cffi-resistant residue.

### Regressions (5 URLs were ok in baseline, not-ok with curl_cffi)

- `https://www.barrons.com/` → HTTP 401 (paywall hardening; possibly correlated with the cleaner fingerprint making us look more like a real reader).
- `https://www.pinterest.com/`, `https://www.a16z.com/`, `https://www.ulta.com/`, `https://www.statnews.com/` → all `empty_extraction` (returned 200 with bodies that don't extract). Plausibly: site-side feature detection serves a JS-only shell to chrome131 but a static fallback to `requests`. Small enough to ignore for now.

Net: +99 / -5 = +94 URL improvement.

### Cost

- Extract is ~10× slower (46 ms median → 567 ms median). That's because extracted bodies are now real article HTML (kilobytes of meaningful DOM) rather than thin interstitial pages. Not a regression — it's the cost of actually having content to extract.
- Fetch is ~40% slower (178 ms median → 252 ms median) — curl_cffi's full handshake mimicry has a per-request overhead. Negligible at our scale; the orchestrator already rate-limits well below this ceiling.
- Dependency footprint: `curl_cffi>=0.7.0` adds a wheel with native `curl-impersonate` binaries (~10 MB).

### Hypothesis tested
- *Belief:* "TLS fingerprint, not IP reputation, is the dominant lever for the 22-host fingerprint-determined block set."
- *Result:* **Confirmed and underestimated.** Fingerprint mimicry not only cleared the 22-host set but also recovered 9 more 403 hosts and 69 stealth-block pages we'd been mis-attributing to extractor failure. The intervention is more powerful than the cross-IP analysis predicted because Cloudflare interstitials disguised as 200-with-empty-extraction were a hidden category.

### Next step
- **Ship.** Standalone PR — drop-in `curl_cffi` replacement, no API change to `fetch_page`, all 619 tests green.
- The 8 still-403 hosts (etsy, inc, petco, skynews, surfer, thekitchn, tripadvisor, h&m) and 5 regressions go in the followup queue. Likely needs Playwright/headless-browser, not more impersonation.
- Empty-extraction count fell from 105 → 42, so the readability-lxml/jusText spike from the baseline notebook is much less urgent than it looked. Defer.

---

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


