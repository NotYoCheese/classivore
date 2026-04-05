# Agent Module Design Document

## 1. Purpose

The agent orchestrates the full data expansion loop:

```
analyze coverage → prioritize gaps → collect pages → label pages → update coverage → repeat
```

It automates what a human would do manually: check which taxonomy categories have too few
labeled pages, run targeted collection for those categories, label the new pages, and repeat
until coverage targets are met or a budget/iteration limit is reached.

---

## 2. Dependency Graph

```
cli/main.py
  └─ agent/
       ├─ runner.py          ← entry point: run_agent()
       │    ├─ agent/coverage.py     ← reads labels + taxonomy → coverage report
       │    ├─ collection/__init__.py ← run_collection() (existing)
       │    ├─ labeling/__init__.py   ← run_labeling() (existing)
       │    └─ agent/state.py        ← AgentState: tracks iterations, budgets, decisions
       ├─ coverage.py        ← coverage analysis: gap detection, prioritization
       │    ├─ labeling/state.py      ← reads existing label counts
       │    └─ taxonomy/loader.py     ← reads full category list
       └─ state.py           ← AgentState: iteration log, cost tracking, stop conditions
            └─ (no classivore imports — pure data + persistence)
```

**Data flow (per iteration):**

```
                     ┌─────────────────────────┐
                     │  AgentState (persisted)  │
                     └────────┬────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  1. coverage.py: analyze(categories, labels_dir)         │
  │     IN:  taxonomy categories, label state                │
  │     OUT: CoverageReport (gaps sorted by count ascending) │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  2. runner.py: plan_iteration(report, config)            │
  │     IN:  CoverageReport, agent config                    │
  │     OUT: CollectionPlan (categories, targets, strategy)   │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  3. collection: run_collection(...)                       │
  │     IN:  config, categories, data_dir, pages             │
  │     OUT: summary dict (pages collected per category)     │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  4. labeling: run_labeling(...)                           │
  │     IN:  config, categories, hierarchy, data_dir         │
  │     OUT: summary dict (pages labeled, errors)            │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  5. runner.py: evaluate(before, after)                    │
  │     IN:  pre/post CoverageReports                        │
  │     OUT: IterationResult (yield, cost, decision)         │
  └───────────────────────────────────────────────────────────┘
```

**No circular dependencies.** Agent imports from collection and labeling; neither
imports from agent. The agent module is a pure consumer of existing orchestrators.

**Existing batch.py stays untouched.** The agent calls `run_collection()` and
`run_labeling()` as black boxes — it does not directly use batch.py.

---

## 3. Shared Contracts

### 3.1 Data Models

These are the inter-module data structures. All new code uses dataclasses.
Existing modules continue to use dicts internally (converting at boundaries only),
avoiding a sprawling refactor.

```python
# src/classivore/models.py — shared data models

from dataclasses import dataclass, field


@dataclass
class CategoryGap:
    """A taxonomy category that needs more labeled pages."""
    name: str
    current_count: int
    target_count: int
    deficit: int             # target - current (always >= 0)
    tier1_name: str          # parent tier-1 for grouping


@dataclass
class CoverageReport:
    """Snapshot of label coverage across the taxonomy."""
    total_categories: int
    covered_categories: int        # categories with >= 1 label
    satisfied_categories: int      # categories meeting target
    total_labeled_pages: int
    gaps: list[CategoryGap]        # sorted by current_count ascending
    timestamp: str                 # ISO 8601

    @property
    def coverage_pct(self) -> float:
        if self.total_categories == 0:
            return 0.0
        return self.satisfied_categories / self.total_categories * 100

    @property
    def worst_gaps(self) -> list[CategoryGap]:
        """Top gaps — categories with fewest labels."""
        return self.gaps[:50]


@dataclass
class IterationPlan:
    """What the agent intends to do in one iteration."""
    iteration: int
    target_categories: list[str]   # category names to collect for
    pages_to_collect: int          # total pages target
    strategy: str                  # "template" | "llm" | "hybrid"


@dataclass
class IterationResult:
    """Outcome of one agent iteration."""
    iteration: int
    pages_collected: int
    pages_labeled: int
    categories_satisfied: int      # newly satisfied this iteration
    gaps_before: int               # unsatisfied count before
    gaps_after: int                # unsatisfied count after
    collection_summary: dict
    labeling_summary: dict
```

### 3.2 Module Interfaces

**coverage.py** — pure function, no side effects:
```python
def analyze_coverage(
    categories: list[dict],
    labels_dir: Path,
    target_per_category: int,
    excluded_categories: set[str],
    excluded_tier1: set[str],
) -> CoverageReport:
```

**runner.py** — orchestrator:
```python
def run_agent(
    config: TaxonomyConfig,
    categories: list[dict],
    hierarchy: dict,
    data_dir: str | Path,
    max_iterations: int = 10,
    target_per_category: int | None = None,
    dry_run: bool = False,
    poll_interval: int = 30,
    verbose: bool = False,
) -> AgentSummary:
```

**state.py** — persistence (same atomic pattern as LabelState/CollectionState):
```python
class AgentState:
    def __init__(self, state_dir: Path): ...
    def save(self): ...
    def start_iteration(self, plan: IterationPlan): ...
    def complete_iteration(self, result: IterationResult): ...
    def should_stop(self, config: AgentConfig) -> tuple[bool, str]: ...
    def summary(self) -> dict: ...
```

### 3.3 Communication Pattern

**Synchronous throughout.** Both `run_collection()` and `run_labeling()` are blocking
calls. The agent calls them sequentially within each iteration. No async, no threads,
no task queues. This matches the existing codebase and is appropriate because:

1. Batch API polling is inherently wait-heavy (30s intervals)
2. Collection is I/O-bound with rate limits (1 req/s Brave, circuit breaker)
3. The agent is designed to run unattended — latency is irrelevant

### 3.4 Logging Contract

**New requirement: structlog with JSON output.** This replaces the current mix of
`logging.getLogger(__name__)` and `print()` calls. See Section 6 (Refactoring) for
the migration plan.

```python
# src/classivore/logging_config.py

import structlog

def configure_logging(verbose: bool = False, json_output: bool = True):
    """Configure the shared structlog pipeline. Call once at CLI entry."""
    ...

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger bound with the module name."""
    ...
```

Every module replaces `logger = logging.getLogger(__name__)` with
`logger = get_logger(__name__)`.

The CLI entry point calls `configure_logging()` once, binds `run_id` context,
and all downstream log calls inherit it:

```python
logger = get_logger(__name__)
logger = logger.bind(run_id=run_id, job_type="agent")
```

Timestamps are handled by the shared processor pipeline — no module formats
its own timestamps.

### 3.5 Error Contract

**New: typed errors for cross-module boundaries.**

```python
# src/classivore/errors.py

class ClassivoreError(Exception):
    """Base for all classivore errors."""

class ConfigError(ClassivoreError):
    """Invalid or missing configuration."""

class CorpusError(ClassivoreError):
    """Problems with corpus data (missing, corrupt, empty)."""

class BatchAPIError(ClassivoreError):
    """Batch API submission or polling failure."""

class SearchExhaustedError(ClassivoreError):
    """All search providers exhausted with no results."""

class BudgetExhaustedError(ClassivoreError):
    """Agent budget (iterations, API calls, cost) exceeded."""
```

**Rules:**
- Existing silent failures in collection/labeling (return None on fetch error, etc.)
stay as-is — they are internal to those modules and represent graceful degradation.
- Cross-module boundaries raise typed errors: if `run_collection()` or
`run_labeling()` encounter a fatal problem (no API key, no corpus, config invalid),
they raise a ClassivoreError subclass.
- The agent catches these at the iteration level and decides whether to retry, skip,
or abort.

---

## 4. DRY Audit

### 4.1 Duplicated Patterns Found

**A) Atomic save (3 implementations, identical logic):**

| File | Lines |
|------|-------|
| `collection/state.py` | 44-67 |
| `labeling/state.py` | 38-62 |
| `collection/domains.py` | ~40-60 |

All three do:
```python
fd, tmp_path = tempfile.mkstemp(dir=..., prefix=..., suffix=".tmp")
try:
    with open(fd, "w") as f:
        json.dump(data, f, indent=2)
    Path(tmp_path).replace(self.state_file)
except BaseException:
    Path(tmp_path).unlink(missing_ok=True)
    raise
```

**Resolution:** Extract to a shared utility:

```python
# src/classivore/persistence.py

def atomic_json_save(data: dict, target: Path, dir: Path) -> None:
    """Write JSON atomically via temp+rename."""
    fd, tmp_path = tempfile.mkstemp(dir=dir, prefix=".", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
```

**B) NDJSON loading (2 implementations):**

| File | Function | Lines |
|------|----------|-------|
| `labeling/__init__.py` | `_load_corpus()` | 293-307 |
| `collection/__init__.py` | `_seed_from_labels()` | 226-244 (partial) |

Both iterate NDJSON with the same pattern: `for line in f: strip, skip empty, json.loads`.

**Resolution:** Extract to shared utility:

```python
# src/classivore/persistence.py

def load_ndjson(path: Path) -> list[dict]:
    """Load all records from an NDJSON file."""

def iter_ndjson(path: Path) -> Iterator[dict]:
    """Stream records from an NDJSON file without loading all into memory."""

def append_ndjson(path: Path, records: list[dict]) -> None:
    """Append records to an NDJSON file."""
```

**C) Verbose logging setup (2 implementations):**

| File | Lines |
|------|-------|
| `collection/__init__.py` | 66-71 |
| `labeling/__init__.py` | 53-58 |

Both do:
```python
if verbose:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s ...")
```

**Resolution:** Eliminated by the structlog migration — `configure_logging(verbose=verbose)`
called once in the CLI, not in each orchestrator.

**D) Content hash computation (1 implementation, but the concept is used in 2 places):**

`collection/filters.py:content_hash()` is the single implementation — no duplication.
Confirmed clean.

### 4.2 Near-Duplicates (Not Worth Consolidating)

- `CollectionState.save()` and `LabelState.save()` have the same atomic-write
  pattern but different data schemas. After extracting `atomic_json_save()`, each
  just builds its data dict and calls the shared function. No further consolidation
  needed.

- `_cmd_collect` and `_cmd_label` in `cli/main.py` both load taxonomy config and
  enriched file. This is a 5-line pattern that's clearer inline than abstracted.

---

## 5. Refactoring Inventory

These changes are prerequisites to the agent implementation. Each is a small,
testable unit.

### 5.1 Extract `persistence.py` (New Module)

**What:** `atomic_json_save()`, `load_ndjson()`, `iter_ndjson()`, `append_ndjson()`

**Why:** Three state classes duplicate atomic save. Two orchestrators duplicate NDJSON
loading. The agent will need both patterns (its own state + reading labels/corpus).
Without this, we'd add a fourth copy of atomic save and a third copy of NDJSON load.

**Who changes:**
- `collection/state.py` — replace inline atomic save with `atomic_json_save()`
- `labeling/state.py` — same
- `collection/domains.py` — same
- `labeling/__init__.py` — replace `_load_corpus()` with `load_ndjson()`
- `collection/__init__.py` — replace inline NDJSON read in `_seed_from_labels()` and
  `_load_existing_hashes()` with `iter_ndjson()`; replace `_save_checkpoint()` corpus
  append with `append_ndjson()`

### 5.2 Add `errors.py` (New Module)

**What:** Exception hierarchy (see Section 3.5)

**Why:** Currently no module raises typed errors. The agent needs to distinguish
"no API key" (abort) from "search failed" (retry) from "no pages collected"
(skip this iteration). Adding typed errors now lets existing modules raise them
at their boundaries.

**Who changes:**
- `batch.py` — `RuntimeError("No API key")` → `ConfigError`
- `config/settings.py` — `FileNotFoundError` stays (standard), but invalid config
  values become `ConfigError`
- Orchestrators — add `CorpusError` when corpus file is missing/empty
  (currently returns early silently)

### 5.3 Add `logging_config.py` (New Module)

**What:** Single structlog configuration, shared processor pipeline with JSON output,
timestamp injection, module name binding.

**Why:** Currently 6 modules independently call `logging.getLogger(__name__)` and 2
orchestrators independently configure `logging.basicConfig()`. The agent is the 7th
module and 3rd orchestrator — without centralization, the pattern will continue
to fragment.

**Who changes:**
- `collection/__init__.py` — remove `logging.basicConfig()`, replace `logging.getLogger`
  with `get_logger`
- `labeling/__init__.py` — same
- `collection/scraper.py`, `collection/search.py`, `collection/commoncrawl.py`,
  `labeling/parser.py` — replace `logging.getLogger` with `get_logger`
- `batch.py` — replace `print()` with structured log calls
- `cli/main.py` — call `configure_logging()` before dispatching to command handlers

### 5.4 Add `models.py` (New Module)

**What:** Shared dataclasses (see Section 3.1)

**Why:** The agent needs typed data structures for coverage reports and iteration
plans. Putting these in the agent package would make them inaccessible to future
modules. A shared `models.py` gives all modules a common vocabulary.

**Who changes:**
- No existing code changes — existing dicts continue to work. The agent uses
  dataclasses internally and converts at boundaries with existing APIs.

### 5.5 Fix File Handle Leak in `labeling/__init__.py`

**What:** Lines 176 and 270 open raw result files without `with` context manager.

**Why:** If an exception occurs during `iter_succeeded_results()`, the file handle
leaks. This is a bug, not a style issue.

**Fix:** Wrap in `with open(...) as raw_file:`.

---

## 6. Existing Anti-Patterns

### 6.1 Print-Based Progress Reporting

**Where:** `batch.py` lines 71-76, 106, 108; `cli/main.py` throughout

**Problem:** `print()` bypasses logging, has no timestamps, can't be suppressed,
can't be structured. The user has already asked for timestamps in output (and we
added them to logging.basicConfig format strings, but print calls don't go through
logging).

**Fix:** After structlog migration, all progress output goes through the logger.
CLI-facing output uses a structlog renderer that outputs human-readable lines
(with timestamps) when `json_output=False`, and JSON when piped/redirected.

### 6.2 Orchestrator Verbose Flag Configures Global Logging

**Where:** `collection/__init__.py:66-71`, `labeling/__init__.py:53-58`

**Problem:** Each orchestrator calls `logging.basicConfig()`, which is global and
can only be called effectively once. If collection runs before labeling in the
same process (which the agent does), only the first call takes effect.

**Fix:** `configure_logging()` in `logging_config.py` called once at CLI entry.
Orchestrators accept a logger instance or use `get_logger()`.

### 6.3 Raw Dict Interfaces Between Modules

**Where:** Category dicts, page dicts, label dicts — everywhere

**Problem:** No type checking at module boundaries. A typo in a dict key
(`"categroies"` vs `"categories"`) is a silent bug.

**Assessment:** Full migration to dataclasses across the entire codebase is out of
scope. Instead, new agent code uses dataclasses internally and converts at module
boundaries. The `models.py` dataclasses serve as documentation of the expected dict
shapes even where conversion isn't applied.

### 6.4 State Classes Expose Internal Dicts

**Where:** `CollectionState.categories`, `LabelState.pages` — accessed directly by
orchestrators

**Problem:** Orchestrators reach into state internals (e.g.,
`state.categories[name]["queries_tried"]` in `collection/__init__.py:125`).

**Assessment:** Refactoring all state access to go through methods would be a
significant change with marginal benefit for code that's stable and tested. The
agent will not access collection or labeling state directly — it uses the
orchestrator APIs (`run_collection()`, `run_labeling()`) as black boxes and reads
coverage from label output files.

---

## 7. Module Design

### 7.1 `src/classivore/agent/coverage.py`

Pure analysis — no side effects, no API calls.

```python
def analyze_coverage(
    categories: list[dict],
    labels_dir: Path,
    target_per_category: int,
    excluded_categories: set[str],
    excluded_tier1: set[str],
) -> CoverageReport:
    """Analyze label coverage across taxonomy categories.

    Reads labels.json (NDJSON), counts per-category, compares against
    taxonomy leaf categories, returns gaps sorted by count ascending
    (fewest labels first = highest priority).
    """
```

**Algorithm:**
1. Load all leaf category names from taxonomy (excluding excluded_categories
   and categories under excluded_tier1)
2. Load labels from `labels_dir/labels.json` via `iter_ndjson()`
3. Count occurrences of each category name across all labeled pages
4. For each leaf category: compute deficit = max(0, target - count)
5. Build sorted gap list (ascending by current_count — emptiest first)
6. Return CoverageReport

### 7.2 `src/classivore/agent/state.py`

Tracks iteration history, cumulative progress, and stop conditions.

```python
@dataclass
class AgentConfig:
    """Stop conditions and budget for the agent."""
    max_iterations: int = 10
    target_per_category: int = 50
    min_yield_per_iteration: int = 5      # stop if fewer pages labeled
    max_consecutive_zero_yield: int = 2   # stop after N fruitless iterations

class AgentState:
    """Persists agent run progress across crashes."""

    def __init__(self, state_dir: Path): ...

    def save(self): ...  # uses atomic_json_save()

    def start_iteration(self, plan: IterationPlan) -> None:
        """Record the start of an iteration."""

    def complete_iteration(self, result: IterationResult) -> None:
        """Record iteration outcome. Appends to history."""

    def should_stop(self, config: AgentConfig) -> tuple[bool, str]:
        """Evaluate stop conditions. Returns (should_stop, reason).

        Conditions checked:
        1. max_iterations reached
        2. All categories satisfied
        3. Last N iterations had zero yield
        4. Last iteration yield below min_yield threshold
        """

    def current_iteration(self) -> int:
        """Current iteration number (0-indexed)."""

    def summary(self) -> dict:
        """Summary for CLI output."""
```

**Persistence schema:**
```json
{
  "started_at": "...",
  "last_checkpoint_at": "...",
  "config": { "max_iterations": 10, "target_per_category": 50, ... },
  "iterations": [
    {
      "iteration": 0,
      "started_at": "...",
      "completed_at": "...",
      "plan": { "target_categories": [...], "pages_to_collect": 100 },
      "result": { "pages_collected": 45, "pages_labeled": 42, ... }
    }
  ]
}
```

### 7.3 `src/classivore/agent/runner.py`

The main loop. Each iteration is:
1. Analyze coverage
2. Check stop conditions
3. Plan: select categories, compute targets
4. Collect
5. Label
6. Evaluate
7. Log + checkpoint

```python
def run_agent(config, categories, hierarchy, data_dir,
              max_iterations=10, target_per_category=None,
              dry_run=False, poll_interval=30, verbose=False):
    """Run the data expansion agent.

    Iterates: analyze → collect → label → evaluate → repeat.
    Prioritizes categories with the fewest labeled pages.
    Stops when targets met, budget exhausted, or yield drops to zero.
    """
```

**Iteration planning logic:**

```python
def _plan_iteration(report: CoverageReport, config: AgentConfig,
                    iteration: int) -> IterationPlan:
    """Decide what to collect this iteration.

    Strategy:
    1. Take the worst gaps (fewest labels first)
    2. Cap at ~100 categories per iteration (keeps collection focused)
    3. Set per-category target proportional to deficit
    4. Use template queries first, LLM queries on retry iterations
    """
```

**Targeted collection:** The agent calls `run_collection()` with a modified config
where `excluded_categories` is set to everything EXCEPT the target categories.
This focuses collection on the gaps without modifying the collection module.

**Labeling scope:** After collection, the agent calls `run_labeling()` normally —
it labels all unlabeled pages in the corpus, not just newly collected ones. This
is correct because new pages are already in the corpus and the labeling module
skips already-labeled pages via LabelState.

### 7.4 `src/classivore/agent/__init__.py`

Re-exports `run_agent` for clean imports.

### 7.5 CLI Integration (`cli/main.py`)

```python
def _register_agent(subparsers):
    p = subparsers.add_parser("agent", help="Run data expansion agent")
    _add_common_args(p)
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--target", type=int, default=None,
                   help="Target labeled pages per category (default: from config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show coverage analysis without collecting or labeling")
    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--status", action="store_true",
                   help="Show agent run history and current coverage")
```

---

## 8. LangGraph Decision

The existing `docs/claude/agent.md` references LangGraph, and `pyproject.toml` has it
as an optional dependency. However, the agent workflow is a simple sequential loop
with no branching, no parallel nodes, no human-in-the-loop, and no conditional edges
that benefit from a graph framework.

**Recommendation: Do not use LangGraph.** The agent is a `for` loop with 5 steps.
LangGraph would add:
- A dependency to install and maintain
- Graph/node/edge ceremony for what is a linear sequence
- Complexity in debugging (graph execution traces vs. simple stack traces)
- No benefit — there are no conditional branches, retries happen at the
  iteration level, and state is already handled by AgentState

If the agent later needs conditional routing (e.g., "if collection yield is low,
switch to LLM queries and retry before labeling"), that can be added as simple
if/else in the runner without a framework.

---

## 9. structlog Migration Plan

### 9.1 New File: `src/classivore/logging_config.py`

```python
import logging
import sys
import structlog

def configure_logging(verbose: bool = False, json_output: bool = False):
    """Configure structlog with shared processor pipeline.

    Call once at CLI entry point before any logging calls.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
        json_output: If True, render as JSON. Otherwise human-readable.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            timestamp_key="timestamp",
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger for the given module name."""
    return structlog.get_logger(name)
```

### 9.2 Migration Scope

**Files to update (replace `logging.getLogger` with `get_logger`):**

| File | Current | After |
|------|---------|-------|
| `collection/__init__.py` | `logger = logging.getLogger(__name__)` + `logging.basicConfig()` | `logger = get_logger(__name__)` (remove basicConfig) |
| `collection/scraper.py` | `logger = logging.getLogger(__name__)` | `logger = get_logger(__name__)` |
| `collection/search.py` | `logger = logging.getLogger(__name__)` | `logger = get_logger(__name__)` |
| `collection/commoncrawl.py` | `logger = logging.getLogger(__name__)` | `logger = get_logger(__name__)` |
| `labeling/__init__.py` | `logger = logging.getLogger(__name__)` + `logging.basicConfig()` | `logger = get_logger(__name__)` (remove basicConfig) |
| `labeling/parser.py` | `logger = logging.getLogger(__name__)` | `logger = get_logger(__name__)` |
| `batch.py` | `print()` calls | `logger = get_logger(__name__)` + `logger.info()` |

**CLI entry point** (`cli/main.py`):
```python
def main():
    # ... parse args ...
    from classivore.logging_config import configure_logging
    configure_logging(verbose=getattr(args, 'verbose', False))
    # ... dispatch to command handler ...
```

### 9.3 Context Binding

The agent runner binds run-level context once:

```python
import structlog

def run_agent(...):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        run_id=f"agent-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        job_type="agent",
        taxonomy=config.slug,
    )
    # All downstream log calls automatically include run_id, job_type, taxonomy
```

Similarly for `_cmd_collect` and `_cmd_label`:

```python
structlog.contextvars.bind_contextvars(
    run_id=f"collect-{...}",
    job_type="collect",
    taxonomy=args.taxonomy,
)
```

---

## 10. New Dependencies

```toml
# pyproject.toml additions
dependencies = [
    ...
    "structlog>=24.1.0",
]
```

**LangGraph removed from optional dependencies** — not used.

---

## 11. Implementation Order

Each step is a commit-sized unit with tests. No step depends on a later step.

1. **`errors.py`** — Exception hierarchy. No existing code changes yet.
2. **`persistence.py`** + tests — `atomic_json_save()`, `load_ndjson()`, `iter_ndjson()`, `append_ndjson()`
3. **Refactor state classes** — Replace inline atomic saves in `collection/state.py`, `labeling/state.py`, `collection/domains.py` with `atomic_json_save()`
4. **Refactor NDJSON I/O** — Replace inline NDJSON in `labeling/__init__.py`, `collection/__init__.py` with shared functions
5. **Fix file handle leak** — `labeling/__init__.py` lines 176, 270
6. **`logging_config.py`** + structlog migration — Replace all `logging.getLogger` calls, remove `logging.basicConfig()`, convert `print()` in batch.py. Add `structlog` to dependencies.
7. **`models.py`** — Shared dataclasses (`CategoryGap`, `CoverageReport`, `IterationPlan`, `IterationResult`)
8. **`agent/coverage.py`** + tests — Coverage analysis
9. **`agent/state.py`** + tests — Agent state persistence, stop conditions
10. **`agent/runner.py`** + tests — Orchestrator loop
11. **CLI wiring** — `_cmd_agent`, `_register_agent` updates
12. **Integration test** — Dry run end-to-end with mock collection/labeling

---

## 12. Test Strategy

### Unit Tests

| Module | Key Tests |
|--------|-----------|
| `persistence.py` | atomic save survives crash, atomic save creates target, NDJSON round-trip, empty file, malformed lines skipped, append to existing |
| `errors.py` | hierarchy (`isinstance(ConfigError(), ClassivoreError)`) |
| `logging_config.py` | configure produces timestamps, JSON mode produces valid JSON, verbose sets DEBUG level, context vars propagate |
| `agent/coverage.py` | empty labels → all gaps, partial coverage, excluded categories omitted, sorted by count ascending, non-leaf categories handled |
| `agent/state.py` | init fresh, persistence round-trip, iteration history, stop conditions (max iterations, zero yield, all satisfied) |
| `agent/runner.py` | dry run shows coverage, plan prioritizes worst gaps, skips satisfied categories, stops on zero yield |

### Integration Tests

| Test | What It Verifies |
|------|-----------------|
| `test_agent_dry_run` | Load real taxonomy, analyze coverage against test labels, verify gap ordering |
| `test_agent_single_iteration` | Mock collection + labeling, verify state transitions and coverage improvement |

### Existing Test Updates

Tests that assert on logging output or error types may need updates after the
structlog migration. Run full suite after step 6.

---

## 13. Files Changed Summary

### New Files
| File | Purpose |
|------|---------|
| `src/classivore/errors.py` | Typed exception hierarchy |
| `src/classivore/persistence.py` | Atomic JSON save, NDJSON I/O utilities |
| `src/classivore/logging_config.py` | Shared structlog configuration |
| `src/classivore/models.py` | Shared dataclasses for inter-module data |
| `src/classivore/agent/__init__.py` | Re-exports `run_agent` |
| `src/classivore/agent/runner.py` | Agent orchestration loop |
| `src/classivore/agent/coverage.py` | Coverage analysis and gap detection |
| `src/classivore/agent/state.py` | Agent state persistence |
| `tests/unit/test_persistence.py` | Tests for shared utilities |
| `tests/unit/test_logging_config.py` | Tests for structlog setup |
| `tests/unit/test_agent_coverage.py` | Tests for coverage analysis |
| `tests/unit/test_agent_state.py` | Tests for agent state |
| `tests/unit/test_agent_runner.py` | Tests for agent orchestrator |

### Modified Files
| File | Change |
|------|--------|
| `collection/state.py` | Use `atomic_json_save()`, use `get_logger()` |
| `labeling/state.py` | Use `atomic_json_save()` |
| `collection/domains.py` | Use `atomic_json_save()` |
| `collection/__init__.py` | Use `get_logger()`, `iter_ndjson()`, `append_ndjson()`, remove `logging.basicConfig()` |
| `labeling/__init__.py` | Use `get_logger()`, `load_ndjson()`, fix file handle leak, remove `logging.basicConfig()` |
| `collection/scraper.py` | Use `get_logger()` |
| `collection/search.py` | Use `get_logger()` |
| `collection/commoncrawl.py` | Use `get_logger()` |
| `labeling/parser.py` | Use `get_logger()` |
| `batch.py` | Use `get_logger()`, replace `print()` |
| `cli/main.py` | Call `configure_logging()`, update `_cmd_agent`, `_register_agent` |
| `pyproject.toml` | Add `structlog`, remove `langgraph` from optional deps |
| `docs/claude/agent.md` | Update architecture doc to match implementation |
