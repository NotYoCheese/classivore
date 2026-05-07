#!/usr/bin/env python3
"""Live-scrape benchmark for the classivore scraper.

Purpose: produce comparable, reproducible signal about how well the current
scraper fetches and extracts text from real URLs. Used to evaluate proposed
scraper improvements across environments (network conditions, transport
options, header strategies, etc.) before spending engineering time on
integration.

The bench is a *research instrument*. It is read-only against the production
DomainTracker and writes one JSONL record per URL to `tools/results/`.
Records are designed for diff-friendly comparison across runs — same URL set
+ different `--label` should let you slice the data on label and see what
moved.

Subcommands:
  run   — fetch URLs and emit JSONL records
  stub  — print a notebook-entry skeleton to stdout (for runs done elsewhere)

JSONL record fields:
  ts                    ISO 8601 UTC timestamp
  label                 free-form run label (passed via --label)
  url                   the URL fetched
  host                  parsed hostname (lowercased)
  outcome               ok | empty_extraction | blocked | http_error
                        | timeout | connection_error | domain_blocked
                        | exception
  http_status           int or null
  bytes_received        len(response.content) or null
  extracted_chars       len(extract_text(html)) or null
  fetch_ms              wall time in fetch (ms) or null
  extract_ms            wall time in extract (ms) or null
  total_ms              wall time across the URL (ms)
  block_markers         list of marker names that fired (see BLOCK_MARKERS)
  fetch_failed_reason   short tag when fetch failed (e.g. timeout,
                        non_html_content_type, domain_quality_blocked,
                        upstream_429_aborted_run)
  would_have_used_exa   true if production would have fallen through to Exa
                        /contents (i.e. fetch+extract did not yield ≥100
                        chars and the URL is not a domain-blocked skip)
  error                 short str(exception) or null
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Make src/ importable when invoked from a checkout without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classivore.collection import scraper  # noqa: E402
from classivore.collection.domains import DomainTracker  # noqa: E402

USABLE_TEXT_MIN_CHARS = 100
SMALL_BODY_BYTES = 5_000
LARGE_BODY_BYTES = 50_000
TINY_BODY_BYTES = 2_048

BLOCK_MARKERS = {
    "just_a_moment": re.compile(r"just a moment", re.IGNORECASE),
    "checking_your_browser": re.compile(r"checking your browser", re.IGNORECASE),
    "access_denied": re.compile(r"access denied", re.IGNORECASE),
    "attention_required": re.compile(r"attention required", re.IGNORECASE),
    "captcha_present": re.compile(r"captcha", re.IGNORECASE),
    "cloudflare_challenge": re.compile(
        r"_cf_chl_opt|cf-challenge|<title[^>]*>\s*just a moment", re.IGNORECASE,
    ),
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _host_of(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _detect_block_markers(body, body_bytes):
    markers = []
    if body_bytes is not None and body_bytes < TINY_BODY_BYTES:
        markers.append("tiny_body")
    if body:
        for name, pattern in BLOCK_MARKERS.items():
            if pattern.search(body):
                markers.append(name)
    return markers


def _is_blocked(markers, body_bytes):
    """Decide whether marker hits should override extraction.

    - tiny_body alone is enough — there's nothing to extract.
    - For other markers: trust the markers when the body is small (< 5KB);
      ignore them when the body is large (> 50KB) since long articles
      legitimately mention "captcha" or "access denied".
    """
    if not markers:
        return False
    if "tiny_body" in markers:
        return True
    if body_bytes is None:
        return True
    if body_bytes < SMALL_BODY_BYTES:
        return True
    if body_bytes > LARGE_BODY_BYTES:
        return False
    # Mid-range: trust the markers (cheap insurance — extraction is still
    # attempted and recorded; this only affects the `outcome` label).
    return True


def _classify_exception(exc):
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return "timeout", "timeout"
    if "connection" in name or "ssl" in name or "dns" in name:
        return "connection_error", "connection_error"
    return "exception", name


def _build_record(label, url, host):
    return {
        "ts": _now_iso(),
        "label": label,
        "url": url,
        "host": host,
        "outcome": None,
        "http_status": None,
        "bytes_received": None,
        "extracted_chars": None,
        "fetch_ms": None,
        "extract_ms": None,
        "total_ms": None,
        "block_markers": [],
        "fetch_failed_reason": None,
        "would_have_used_exa": False,
        "error": None,
    }


def _process_url_sync(url, label, domains, host_state):
    """Synchronous fetch+extract+classify for a single URL.

    Returns a JSONL-ready dict. Pure function aside from reading host_state
    (a dict tracking 429s) and DomainTracker (read-only).
    """
    host = _host_of(url)
    rec = _build_record(label, url, host)
    t0 = time.perf_counter()

    if host and host_state.get(host) == "429_aborted":
        rec["outcome"] = "http_error"
        rec["http_status"] = 429
        rec["fetch_failed_reason"] = "upstream_429_aborted_run"
        rec["would_have_used_exa"] = True
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    if domains is not None and host and domains.is_blocked(host):
        rec["outcome"] = "domain_blocked"
        rec["fetch_failed_reason"] = "domain_quality_blocked"
        rec["would_have_used_exa"] = False
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    fetch_start = time.perf_counter()
    try:
        resp = scraper._fetch_response(url)
    except Exception as exc:
        outcome, reason = _classify_exception(exc)
        rec["outcome"] = outcome
        rec["fetch_failed_reason"] = reason
        rec["error"] = str(exc)[:200]
        rec["fetch_ms"] = (time.perf_counter() - fetch_start) * 1000.0
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        rec["would_have_used_exa"] = True
        return rec

    rec["fetch_ms"] = (time.perf_counter() - fetch_start) * 1000.0
    rec["http_status"] = resp.status_code
    body_bytes = len(resp.content) if resp.content is not None else 0
    rec["bytes_received"] = body_bytes

    if resp.status_code == 429 and host:
        host_state[host] = "429_aborted"

    if resp.status_code != 200:
        rec["outcome"] = "http_error"
        rec["fetch_failed_reason"] = f"http_{resp.status_code}"
        rec["would_have_used_exa"] = True
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    content_type = (resp.headers.get("content-type") or "").lower()
    if "html" not in content_type:
        rec["outcome"] = "http_error"
        rec["fetch_failed_reason"] = "non_html_content_type"
        rec["would_have_used_exa"] = True
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    body = resp.text or ""
    markers = _detect_block_markers(body, body_bytes)
    rec["block_markers"] = markers

    if _is_blocked(markers, body_bytes):
        rec["outcome"] = "blocked"
        rec["fetch_failed_reason"] = "block_markers_detected"
        rec["would_have_used_exa"] = True
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    extract_start = time.perf_counter()
    try:
        text = scraper.extract_text(body)
    except Exception as exc:
        rec["outcome"] = "exception"
        rec["error"] = f"extract: {str(exc)[:180]}"
        rec["extract_ms"] = (time.perf_counter() - extract_start) * 1000.0
        rec["would_have_used_exa"] = True
        rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    rec["extract_ms"] = (time.perf_counter() - extract_start) * 1000.0
    chars = len(text) if text else 0
    rec["extracted_chars"] = chars

    if chars < USABLE_TEXT_MIN_CHARS:
        rec["outcome"] = "empty_extraction"
        rec["would_have_used_exa"] = True
    else:
        rec["outcome"] = "ok"
        rec["would_have_used_exa"] = False

    rec["total_ms"] = (time.perf_counter() - t0) * 1000.0
    return rec


async def _bounded_process(url, label, domains, host_state, global_sem,
                           per_host_sems, per_domain_max, loop):
    host = _host_of(url)
    if host:
        if host not in per_host_sems:
            per_host_sems[host] = asyncio.Semaphore(per_domain_max)
        host_sem = per_host_sems[host]
    else:
        host_sem = None

    async with global_sem:
        if host_sem is not None:
            async with host_sem:
                return await loop.run_in_executor(
                    None, _process_url_sync, url, label, domains, host_state,
                )
        return await loop.run_in_executor(
            None, _process_url_sync, url, label, domains, host_state,
        )


def _read_urls(path, limit):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
            if limit and len(urls) >= limit:
                break
    return urls


async def _run_async(urls, label, domains, output_path, concurrency,
                     per_domain_max, progress_every):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    global_sem = asyncio.Semaphore(concurrency)
    per_host_sems = {}
    host_state = {}

    tasks = [
        asyncio.create_task(
            _bounded_process(
                url, label, domains, host_state, global_sem,
                per_host_sems, per_domain_max, loop,
            )
        )
        for url in urls
    ]

    counts = defaultdict(int)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counts[rec["outcome"]] += 1
            written += 1
            if progress_every and written % progress_every == 0:
                _print_progress(written, len(urls), counts)

    _print_progress(written, len(urls), counts, final=True)
    return counts


def _print_progress(written, total, counts, final=False):
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    prefix = "[done]" if final else "[..]"
    print(f"{prefix} {written}/{total}  " + "  ".join(parts), file=sys.stderr)


def cmd_run(args):
    urls = _read_urls(args.urls_file, args.limit)
    if not urls:
        print("No URLs found in fixture (after stripping comments).",
              file=sys.stderr)
        return 2

    domains = None
    if args.domain_state_dir:
        domains = DomainTracker(args.domain_state_dir)

    output_path = Path(args.output)

    print(
        f"[bench] label={args.label} urls={len(urls)} "
        f"concurrency={args.concurrency} per_domain={args.per_domain_max} "
        f"output={output_path}",
        file=sys.stderr,
    )

    counts = asyncio.run(
        _run_async(
            urls=urls,
            label=args.label,
            domains=domains,
            output_path=output_path,
            concurrency=args.concurrency,
            per_domain_max=args.per_domain_max,
            progress_every=args.progress_every,
        )
    )

    total = sum(counts.values())
    success = counts.get("ok", 0)
    print(
        f"[bench] success_rate={success}/{total}="
        f"{success / total:.1%}" if total else "[bench] no records",
        file=sys.stderr,
    )
    return 0


def cmd_stub(args):
    """Emit a notebook-entry skeleton for runs performed elsewhere.

    Use this when results came from a remote machine (e.g., a Hetzner box,
    a residential proxy run) that wasn't invoked through `run`. Paste the
    output into docs/scraper-notebook.md and fill in the metrics.
    """
    env_bits = []
    if args.residential:
        env_bits.append("residential proxy")
    if args.hetzner:
        env_bits.append("Hetzner")
    if args.curl_cffi:
        env_bits.append("curl_cffi")
    if args.notes:
        env_bits.append(args.notes)
    env = ", ".join(env_bits) if env_bits else "(describe environment)"

    print(f"""## {args.label} — {datetime.now(timezone.utc).date().isoformat()}

**Environment:** {env}
**URL set:** {args.urls or "tools/fixtures/urls_top200.txt"}
**Concurrency / per-domain:** TODO

### Metrics
- success rate (ok / total): TODO
- median fetch_ms: TODO
- p95 fetch_ms: TODO
- blocked: TODO
- empty_extraction: TODO
- http_error breakdown: TODO

### Observations
- TODO

### Hypothesis tested
- TODO

### Next step
- TODO
""")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="scraper_bench",
        description="Live-scrape bench for evaluating scraper improvements.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Run the bench against a URL fixture.")
    pr.add_argument("--urls-file", required=True,
                    help="Path to URL fixture (one URL per line, # for comments).")
    pr.add_argument("--output", required=True,
                    help="Path to write JSONL results.")
    pr.add_argument("--label", required=True,
                    help="Run label (e.g., baseline_local, residential_proxy).")
    pr.add_argument("--limit", type=int, default=0,
                    help="Cap to first N URLs in the fixture (0 = all).")
    pr.add_argument("--concurrency", type=int, default=8,
                    help="Global concurrent fetches.")
    pr.add_argument("--per-domain-max", type=int, default=2,
                    help="Max concurrent fetches per host.")
    pr.add_argument("--domain-state-dir", default=None,
                    help="Path to a DomainTracker state dir (read-only). "
                         "Skips fetches that production would skip.")
    pr.add_argument("--progress-every", type=int, default=20,
                    help="Print progress to stderr every N URLs.")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("stub",
                        help="Print a notebook-entry skeleton for an external run.")
    ps.add_argument("--label", required=True,
                    help="Run label for the notebook header.")
    ps.add_argument("--residential", action="store_true",
                    help="Mark environment: residential proxy.")
    ps.add_argument("--hetzner", action="store_true",
                    help="Mark environment: Hetzner cloud.")
    ps.add_argument("--curl-cffi", action="store_true",
                    help="Mark environment: curl_cffi browser impersonation.")
    ps.add_argument("--notes", default=None,
                    help="Free-form environment notes.")
    ps.add_argument("--urls", default=None,
                    help="URL set used (path or description).")
    ps.set_defaults(func=cmd_stub)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
