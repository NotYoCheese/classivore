# Data Expansion Agent

## Modules

- `src/classivore/agent/runner.py` — Main orchestration loop: analyze → collect → label → evaluate → repeat
- `src/classivore/agent/coverage.py` — Pure coverage analysis: reads labels + taxonomy, returns gaps sorted by count ascending
- `src/classivore/agent/state.py` — AgentState: iteration history, stop condition evaluation, atomic persistence

## Shared Infrastructure

- `src/classivore/models.py` — Dataclasses: CategoryGap, CoverageReport, IterationPlan, IterationResult, AgentConfig
- `src/classivore/persistence.py` — atomic_json_save(), load_ndjson(), iter_ndjson(), append_ndjson()
- `src/classivore/errors.py` — Typed exception hierarchy (ClassivoreError, ConfigError, CorpusError, etc.)
- `src/classivore/logging_config.py` — Shared structlog configuration with JSON output and context propagation

## Workflow

```
analyze_coverage → should_stop? → plan_iteration → run_collection → run_labeling → evaluate → loop or end
```

## Key Behaviors

- Prioritizes categories with the fewest labeled pages (gaps sorted ascending by count)
- Focuses collection on gap categories by setting excluded_categories to everything NOT targeted
- Labels all unlabeled corpus pages after collection (labeling module handles skip-if-done)
- Stops on: max_iterations, all categories satisfied, consecutive zero yield, yield below minimum
- Supports crash recovery via AgentState persistence
- Dry run mode shows coverage analysis without API calls
- Status mode shows iteration history and current coverage

## Stop Conditions

Configured via AgentConfig:
- `max_iterations` (default 10): hard cap on collect/label cycles
- `max_consecutive_zero_yield` (default 2): stop if N iterations produce no new labels
- `min_yield_per_iteration` (default 5): stop if yield drops below threshold
- All gaps resolved: stop immediately

## Search Strategy

- Iteration 0: template queries (free, from enriched taxonomy descriptions)
- Iteration 1+: hybrid (template first, LLM queries on retry)
- Collection module handles provider fallback (Brave → Serper) and circuit breaker

## CLI

```bash
classivore agent --taxonomy iab-2.2 --dry-run           # Show coverage, no API calls
classivore agent --taxonomy iab-2.2 --status             # Show run history + coverage
classivore agent --taxonomy iab-2.2 --max-iterations 5   # Run 5 cycles
classivore agent --taxonomy iab-2.2 --target 50          # 50 labels per category target
classivore agent --taxonomy iab-2.2 -v                   # Verbose logging
```

## State

File: `data/agent/{taxonomy-slug}/agent_state.json`

```json
{
  "started_at": "...",
  "last_checkpoint_at": "...",
  "iterations": [
    {
      "iteration": 0,
      "started_at": "...",
      "completed_at": "...",
      "plan": {"target_categories": [...], "pages_to_collect": 100, "strategy": "template"},
      "result": {"pages_collected": 45, "pages_labeled": 42, "gaps_before": 500, "gaps_after": 458}
    }
  ]
}
```

## Dependencies

- Uses `run_collection()` and `run_labeling()` as black boxes
- No direct batch API access — delegates to existing orchestrators
- No LangGraph — simple sequential loop is sufficient

## Tests

- `tests/unit/test_agent_coverage.py` — gap detection, priority ordering, exclusions, multi-label counting
- `tests/unit/test_agent_state.py` — persistence, iteration tracking, all stop conditions
- `tests/unit/test_agent_runner.py` — dry run, single iteration, zero yield stop, category targeting
