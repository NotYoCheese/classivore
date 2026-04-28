# Classivore

Open-source taxonomy-agnostic text classification pipeline. Build multi-label classifiers for any hierarchical taxonomy.

## Quick Reference

- **Package:** `src/classivore/` — all source code lives here
- **Tests:** `tests/` — run with `pytest`
- **CLI:** `src/classivore/cli/` — entry points installed as `classivore <command>`
- **Taxonomies:** `taxonomies/<name>/config.yaml` — one dir per taxonomy

## Architecture Docs

Read the relevant file before modifying that subsystem:

- [Data Model](docs/claude/data_model.md) — shared corpus, label schema, batch API, dedup
- [Taxonomy](docs/claude/taxonomy.md) — loader, validator, enricher, config schema
- [Collection](docs/claude/collection.md) — Common Crawl, web scraper, content filters
- [Labeling](docs/claude/labeling.md) — LLM labeling, prompt builder, provider support
- [Training](docs/claude/training.md) — DeBERTa trainer, focal loss, thresholds
- [Validation](docs/claude/validation.md) — data quality checks via label-lens
- [Inference](docs/claude/inference.md) — classifier loading, prediction, batch
- [Agent](docs/claude/agent.md) — data expansion agent, collect/label/evaluate loop
- [Publishing](docs/claude/publishing.md) — HuggingFace model publishing, artifact contract
- [CLI](docs/claude/cli.md) — command structure, argument patterns

## Rules

- All code goes in `src/classivore/`. No standalone scripts in the root.
- Every module has tests. Write tests before implementation.
- No `sys.path` hacks. Use `pip install -e .` for development.
- Use relative paths. No hardcoded absolute paths.
- Taxonomy-specific logic goes in `taxonomies/<name>/config.yaml`, not in code.
- Keep functions small. If a function needs a comment explaining what it does, it should be two functions with clear names.
- Use `#!/usr/bin/env python3` as the first line of Python files.
- Prefer `uv` over `pip` when possible.

## Git workflow

- `feature/<name>` → `develop`: **squash-merge** (feature branch dies after merge).
- `develop` → `main`: **`--no-ff` merge commit** at release time, then tag `v{X.Y.Z}` on the merge commit. Do NOT squash develop into main — squashing long-lived branches breaks the merge-base and creates phantom conflicts on the next release.
- Never commit directly to `main` or `develop`. Both branches are protected.
- Use gitmoji at the start of commit messages (`✨`, `🐛`, `📝`, `🔧`, `🚸`, `🔖`).

## Environment

- Development machine: M1 Max MacBook Pro
- Python virtual environment: `./venv`
- Activate before installing: `source venv/bin/activate`
- Shell: zsh

## Companion Repo

The private API server lives in a separate repo: `classivore-api`. It imports this package as a dependency.
