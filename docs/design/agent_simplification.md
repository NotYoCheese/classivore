# Agent Simplification Plan

## The Target Design

One target number. One source of truth (`labels.json` counts). One rule: if a
category has fewer labeled pages than target, collect and label more.

```
check labels → find gaps → collect for gaps → label new pages → repeat
```

## 1. Inventory of What Gets Deleted

### Functions

| File | Function/Code | Why it goes |
|------|--------------|-------------|
| `agent/runner.py` | `_plan_iteration()` | Computes per-iteration deficit totals and strategy — unnecessary indirection. The collection module should just collect for unsatisfied categories. |
| `collection/__init__.py` | `_seed_from_labels()` | Seeds collection state from labels. This is the root cause of the target-vs-seeded-count conflict. The collection module shouldn't know about labels at all — the agent tells it what to collect. |
| `collection/__init__.py` | `pages` parameter to `run_collection()` | Distributes a page count across categories. Vestige of standalone collection mode. The target comes from config, not from the caller doing division. |

### Dataclasses / Fields

| File | Item | Why it goes |
|------|------|-------------|
| `models.py` | `IterationPlan.pages_to_collect` | Nobody needs a pre-computed total. Each category collects until satisfied. |
| `models.py` | `IterationPlan.strategy` | The "template vs hybrid" distinction can be derived from iteration number at the point of use. Doesn't need to be planned upfront. |
| `models.py` | `CategoryGap.deficit` | Deficit = target - current. Trivially computed from the other two fields. Storing it invites bugs when one changes without the other. |

### Behaviors

| Current behavior | Why it goes |
|-----------------|-------------|
| Agent computes `pages_to_collect = sum(deficits)` and passes to collection | Collection doesn't need a page budget. It collects until categories are satisfied or queries are exhausted. |
| Collection state tracks `collected` count per category and compares to `target` | This duplicates the label count. Labels.json is the source of truth. Collection state should only track operational concerns (queries tried, domains used). |
| `_seed_from_labels()` updates `state.categories[name]["collected"]` | Eliminated entirely. The concept of "seeded collected count" disappears. |

## 2. Inventory of What Gets Kept, Unchanged

| File | What | Why it stays |
|------|------|-------------|
| `collection/queries.py` | All template and LLM query generation | Implementation detail of how queries are constructed. Correct as-is. |
| `collection/commoncrawl.py` | CDX lookup, WARC download, probe | Content retrieval. Nothing to do with targets. |
| `collection/scraper.py` | Live scraping, text extraction | Same. |
| `collection/filters.py` | URL blocklist, content filtering, dedup | Same. |
| `collection/search.py` | SearchClient with provider fallback | Same. |
| `collection/domains.py` | DomainTracker quality scoring | Same. |
| `collection/dashboard.py` | Status formatting | Same (but will read from simplified state). |
| `labeling/*` | Entire labeling module | Black box. Agent calls `run_labeling()`, gets results. No changes. |
| `agent/coverage.py` | `analyze_coverage()` | Already does exactly the right thing: reads labels.json, counts per category, returns gaps sorted by fewest first. |
| `batch.py` | Batch API utilities | Unchanged. |
| `persistence.py` | Atomic save, NDJSON I/O | Unchanged. |
| `logging_config.py` | structlog configuration | Unchanged. |
| `errors.py` | Exception hierarchy | Unchanged. |
| `taxonomy/*` | Loader, enricher | Unchanged. |
| `validation/*` | Data quality checks | Unchanged. |

## 3. Inventory of What Gets Rewritten

### A. `agent/runner.py` — the core loop

**Current responsibility:** Manages iterations with planning, deficit math,
coverage analysis before/after, strategy selection, result evaluation with
complex labeled-count arithmetic.

**New responsibility:** Simple loop. Each iteration: check coverage, pick
unsatisfied categories, call collection, call labeling, check again.

**Why it changes:** The current code has 7 concerns tangled together (planning,
targeting, strategy, evaluation, result computation, state management, error
handling). The new version separates "what to do" (the loop) from "how to do
it" (collection and labeling modules).

### B. `collection/__init__.py` — `run_collection()`

**Current responsibility:** Initializes per-category targets in CollectionState,
seeds from labels, checks `state.is_satisfied()` to decide when to stop,
tracks collected counts.

**New responsibility:** Receives a set of category names to collect for and a
target count. For each category, generates queries, searches, retrieves, and
filters until the category reaches its target OR queries are exhausted. Does
NOT read labels or compute how many pages exist — the caller (agent) handles
that.

**Why it changes:** Collection currently owns two concerns: "how many pages does
this category need?" (should be the agent's job) and "how do I find and scrape
pages?" (its actual job). Removing the first concern eliminates `_seed_from_labels`
and the target-vs-seeded conflict.

**Key design choice:** `run_collection()` still uses CollectionState for
operational tracking (queries tried, URLs processed, domain diversity). But
`state.categories[name]["collected"]` becomes "pages collected THIS RUN", not
"total pages including labels." The `is_satisfied()` check compares against a
per-run target, not a global total.

### C. `collection/state.py` — `CollectionState`

**Current responsibility:** Tracks per-category targets, collected counts
(including seeded label counts), query history, URL history, domain diversity,
error counts.

**New responsibility:** Same operational tracking (queries, URLs, domains,
errors) but `collected` is purely "pages collected this run." No `target` field
in the category dict — the caller passes a target to `init_category()` and
`is_satisfied()` compares against it as before, but the count starts at 0, not
seeded from labels.

**Why it changes:** Removing label seeding makes the state self-consistent.
`collected` means what it says: pages we actually collected.

**Ambiguity flag:** The current `init_category()` is idempotent — if the
category already exists in state (from a previous run), it's skipped. With
per-run state, this means resumed runs would remember previous progress.
This is correct behavior — if a run is interrupted mid-category, resume
should continue from where it stopped. But it means clearing state between
agent iterations (which happens when the agent creates a new CollectionState
or resets targets).

### D. `models.py` — `IterationPlan` and `AgentConfig`

**Current:** `IterationPlan` has `pages_to_collect`, `strategy`, `iteration`,
`target_categories`.

**New:** Keep `IterationPlan` as a thin audit record for state persistence.
Remove `pages_to_collect` and `strategy`, add `use_llm_queries: bool`.
Fields: `iteration`, `target_categories`, `use_llm_queries`. No logic hangs
off this object — it's purely for the iteration history in `agent_state.json`.

### E. `agent/state.py` — stop conditions

**Current:** `should_stop()` checks max iterations, all satisfied (from last
result's `gaps_after`), consecutive zero yield, and min yield.

**New:** Same logic, but `pages_labeled` in results should be the real count
(post-labels minus pre-labels), not the complex arithmetic currently on lines
146-167 of runner.py.

**Why it changes:** Simplification of result computation, not of the stop
condition logic itself.

## 4. New Module/File Structure

No files renamed, merged, or split. The current structure is fine:

```
src/classivore/
├── agent/
│   ├── __init__.py       # re-exports run_agent
│   ├── runner.py          # simplified core loop
│   ├── coverage.py        # unchanged — reads labels, returns gaps
│   └── state.py           # minor: simpler result recording
├── collection/
│   ├── __init__.py        # simplified: no label seeding, target from caller
│   ├── state.py           # simpler: collected = this-run count only
│   ├── queries.py         # unchanged
│   ├── search.py          # unchanged
│   ├── scraper.py         # unchanged
│   ├── commoncrawl.py     # unchanged
│   ├── filters.py         # unchanged
│   ├── domains.py         # unchanged
│   └── dashboard.py       # minor: reads simplified state
├── models.py              # IterationPlan simplified
└── ... (everything else unchanged)
```

## 5. Ordered Change Sequence

Each step keeps the system runnable. Tests pass at each step.

### Step 1: Simplify `IterationPlan`

Remove `pages_to_collect` and `strategy`. Add `use_llm_queries: bool`.
Update `AgentState` serialization and tests. No behavior change yet —
runner still uses the old planning logic internally but stores the
simplified plan.

### Step 2: Remove label seeding, add per-category targets to `run_collection()`

Single atomic step — both changes land together so the system is never in
a state where collection starts at 0 without per-category targets to
compensate.

**Deletions:**
- Delete `_seed_from_labels()` function
- Remove the call from `run_collection()`
- Remove label seeding tests

**Additions:**
- Add `category_targets: dict[str, int] | None` parameter to
  `run_collection()`. When provided, each category gets its own target
  (how many NEW pages to collect). When `None` (standalone
  `classivore collect`), falls back to `config.target_per_category`
  for all categories.
- Remove `pages` parameter (vestige of old distributed-target mode)

The agent builds the dict from the gap analysis:
```python
category_targets = {g.name: g.target_count - g.current_count
                    for g in report.gaps[:100]}
```

`analyze_coverage()` only includes categories where `current_count < target`
in the gaps list (confirmed: the `if deficit > 0` guard on line 74 of
coverage.py ensures this). So `target_count - current_count` is always
positive.

Collection never reads labels. The agent reads labels once via
`analyze_coverage()`, computes per-category targets, and passes them
down. One source, one direction.

The init loop becomes:
```python
for cat in leaf_cats:
    cat_target = (category_targets or {}).get(
        cat["name"], config.target_per_category
    )
    state.init_category(cat["name"], target=cat_target)
```

`init_category()` receives the per-category target. `is_satisfied()`
compares collected (starting from 0) against that target. No seeding,
no conflict.

Standalone `classivore collect` passes `category_targets=None`, triggering
the fallback to `config.target_per_category` for all categories.

### Step 4: Rewrite `run_agent()` core loop

Replace `_plan_iteration()` with inline logic. The agent computes
per-category targets from the gap analysis and passes them directly to
collection. No plan objects for logic — just a thin audit record in state.

### Step 5: Simplify result computation in runner

Replace the complex arithmetic on lines 146-167 with:
```python
pages_labeled = post_report.total_labeled_pages - report.total_labeled_pages
```
Already computed this way on line 164-167 but the earlier lines try a different
formula first. Just use the simple version.

### Step 6: Update dashboard

`format_status_dashboard()` currently shows `collected` counts from
CollectionState. Switch it to read label counts from `labels.json` for the
coverage display — that's the source of truth. CollectionState counts
(actual pages scraped this run) can still be shown for operational metrics
(velocity, errors, domains) but should not be used for coverage.

### Step 7: Update tests

Tests that assert on seeding behavior get deleted. Tests that assert on
collection targets get updated to reflect the simplified flow. Agent runner
tests get simplified.

## 6. The New Core Loop (Pseudocode)

```python
def run_agent(config, categories, data_dir, target, max_iterations):
    labels_dir = data_dir / "labels" / config.slug

    for iteration in range(max_iterations):
        # 1. What do we have?
        report = analyze_coverage(categories, labels_dir, target)

        # 2. Are we done?
        if not report.gaps:
            break

        # 3. How many new pages does each category need?
        category_targets = {g.name: g.target_count - g.current_count
                            for g in report.gaps[:100]}

        # 4. Collect pages for those categories
        run_collection(config, categories, data_dir,
                       category_targets=category_targets,
                       use_llm_queries=(iteration > 0))

        # 5. Label everything new
        run_labeling(config, categories, data_dir)

        # 6. How did we do?
        new_report = analyze_coverage(categories, labels_dir, target)
        new_labels = new_report.total_labeled_pages - report.total_labeled_pages

        if new_labels == 0:
            break  # no progress, stop
```

12 lines. One source of truth (labels.json via `analyze_coverage`).
Per-category targets eliminate wasted collection. Collection never reads
labels — the agent tells it exactly what to collect. Coverage analysis is
the single point where labeled page counts are computed.

## Edge Cases That the Current Complexity Handles

### 1. Standalone `classivore collect` (without agent)

When `category_targets` is `None` (standalone mode), collection falls back
to `config.target_per_category` for all categories and counts from 0.
Without label seeding, it will try to collect `target` pages per category
regardless of existing labels. This is correct for standalone use — the user
explicitly asked to collect, and the extra pages benefit the shared corpus.

If a user runs `classivore collect` twice, the second run resumes from state
and skips already-tried queries. `collected` counts in state reflect actual
pages from that run. This is correct.

### 2. Multi-label pages

A page labeled `["Sedan", "Auto Buying and Selling"]` increments the count for
both categories in `analyze_coverage()`. This is correct — both categories
benefit from that page existing in the training set.

### 3. Non-leaf labels

A page labeled `["Automotive"]` (non-leaf) doesn't count toward any leaf
category in `analyze_coverage()` because coverage only checks leaf categories.
This is correct — the page doesn't help train leaf-level classifiers.

### 4. Categories with 0 labels but plenty of corpus pages

A category might have 0 labels but 50 unlabeled pages already in the corpus
(from collection for other categories). After simplification, the agent would
try to collect more pages. After labeling, many of those existing pages would
get labeled for this category. The next iteration would see the labels and
reduce the gap. This is correct behavior — potentially one wasted collection
pass, but labeling fixes it.

### 5. Resume after crash

Collection state persists queries tried and URLs processed. Agent state
persists completed iterations. Both resume correctly. The only change: agent
state's incomplete iteration cleanup (dropping unfinished iterations) still
works.

### 6. `--target` lower than current label counts

If a user sets `--target 10` but some categories already have 50 labels,
`analyze_coverage()` returns 0 gaps for those categories. Collection is
never called for them. Correct.
