#!/usr/bin/env python3
"""Format a run record + all-time totals into a human-readable summary."""

from typing import Any


def print_summary(record: dict, all_time: dict) -> None:
    """Write the summary block to stdout."""
    print(format_summary(record, all_time))


def format_summary(record: dict, all_time: dict) -> str:
    """Render a side-by-side summary of this run vs. all-time totals."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("Run summary")
    lines.append("=" * 64)
    lines.append(f"  Command:   {record.get('command', '?')}  (taxonomy={record.get('taxonomy', '?')})")
    lines.append(f"  Started:   {record.get('started_at', '?')}")
    lines.append(f"  Runtime:   {_format_duration(record.get('runtime_seconds', 0))}")

    status = record.get("exit_status", "ok")
    if status == "ok":
        lines.append(f"  Status:    ok")
    elif status == "error":
        err = record.get("error", "unknown")
        lines.append(f"  Status:    error — {err}")
    elif status == "interrupted":
        lines.append(f"  Status:    interrupted (Ctrl-C)")
    else:
        lines.append(f"  Status:    {status}")

    metrics = record.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append(f"  {'':<26}{'THIS RUN':>12}  {'ALL TIME':>12}")

        if "search" in metrics or "search" in all_time:
            lines.extend(_format_search(metrics.get("search", {}), all_time.get("search", {})))

        if "scrape" in metrics or "scrape" in all_time:
            lines.extend(_format_scrape(metrics.get("scrape", {}), all_time.get("scrape", {})))

        if "labeling" in metrics or "labeling" in all_time:
            lines.extend(_format_labeling(metrics.get("labeling", {}), all_time.get("labeling", {})))

        if "coverage" in metrics:
            lines.extend(_format_coverage(metrics.get("coverage", {}), all_time.get("coverage", {})))

    lines.append("=" * 64)
    return "\n".join(lines)


def _format_search(this_run: dict, all_time: dict) -> list[str]:
    out = ["", "  Search queries"]
    providers = sorted(set(
        list(this_run.get("queries_by_provider", {}).keys()) +
        list(all_time.get("queries_by_provider", {}).keys())
    ))
    for p in providers:
        this_n = this_run.get("queries_by_provider", {}).get(p, 0)
        total_n = all_time.get("queries_by_provider", {}).get(p, 0)
        out.append(f"    {p:<24}{_fmt_int(this_n):>12}  {_fmt_int(total_n):>12}")

    if "results_by_provider" in this_run or "results_by_provider" in all_time:
        out.append("  Search results returned")
        providers = sorted(set(
            list(this_run.get("results_by_provider", {}).keys()) +
            list(all_time.get("results_by_provider", {}).keys())
        ))
        for p in providers:
            this_n = this_run.get("results_by_provider", {}).get(p, 0)
            total_n = all_time.get("results_by_provider", {}).get(p, 0)
            out.append(f"    {p:<24}{_fmt_int(this_n):>12}  {_fmt_int(total_n):>12}")
    return out


def _format_scrape(this_run: dict, all_time: dict) -> list[str]:
    out = ["", "  Pages"]
    for key in ("urls_surfaced", "fetched", "kept"):
        if key in this_run or key in all_time:
            out.append(_kv_row(key, this_run.get(key, 0), all_time.get(key, 0), indent=4))

    rejected = this_run.get("rejected", {})
    rejected_total = all_time.get("rejected", {})
    keys = sorted(set(list(rejected.keys()) + list(rejected_total.keys())))
    for key in keys:
        out.append(_kv_row(f"rejected_{key}", rejected.get(key, 0),
                           rejected_total.get(key, 0), indent=4))
    return out


def _format_labeling(this_run: dict, all_time: dict) -> list[str]:
    out = ["", "  Labeling"]
    for key in ("stage1_sent", "tier1_hits", "stage2_sent", "labels_emitted", "errors"):
        if key in this_run or key in all_time:
            out.append(_kv_row(key, this_run.get(key, 0), all_time.get(key, 0), indent=4))

    cache = this_run.get("cache", {})
    cache_total = all_time.get("cache", {})
    for stage in ("stage1", "stage2"):
        c = cache.get(stage, {})
        if not c:
            continue
        ct = cache_total.get(stage, {})
        rate_now = c.get("cache_hit_rate", 0.0) * 100
        # All-time hit rate = cache_read / (cache_read + cache_creation + input)
        cr = ct.get("total_cache_read_tokens", 0)
        cc = ct.get("total_cache_creation_tokens", 0)
        it = ct.get("total_input_tokens", 0)
        denom = cr + cc + it
        rate_all = (cr / denom * 100) if denom > 0 else 0.0
        out.append(f"    {f'cache_hit_rate ({stage})':<24}"
                   f"{rate_now:>11.1f}%  {rate_all:>11.1f}%")

        cost_now = c.get("estimated_cost_usd", 0.0)
        cost_all = ct.get("estimated_cost_usd", 0.0)
        out.append(f"    {f'est_cost ({stage})':<24}"
                   f"{('$' + f'{cost_now:.2f}'):>12}  {('$' + f'{cost_all:.2f}'):>12}")
    return out


def _format_coverage(this_run: dict, all_time: dict) -> list[str]:
    out = ["", "  Coverage"]
    if "at_target_after" in this_run:
        before = this_run.get("at_target_before", 0)
        after = this_run.get("at_target_after", 0)
        total = this_run.get("total_categories", 0)
        delta = after - before
        sign = "+" if delta >= 0 else ""
        out.append(f"    {'at target':<24}"
                   f"{f'{after}/{total}':>12}  {f'{sign}{delta} this run':>12}")
    if "thin_remaining" in this_run:
        out.append(_kv_row("thin remaining", this_run.get("thin_remaining", 0),
                           "", indent=4))
    return out


def _kv_row(label: str, this_val: Any, total_val: Any, indent: int = 4) -> str:
    spaces = " " * indent
    this_str = _fmt_int(this_val) if isinstance(this_val, (int, float)) else str(this_val)
    total_str = _fmt_int(total_val) if isinstance(total_val, (int, float)) else str(total_val)
    return f"{spaces}{label:<{26 - indent}}{this_str:>12}  {total_str:>12}"


def _fmt_int(n: Any) -> str:
    if isinstance(n, float) and not n.is_integer():
        return f"{n:.2f}"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds - minutes * 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {rem}s"
