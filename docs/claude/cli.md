# CLI Subsystem

## Commands

All CLI entry points live in `cli/` and are registered in `pyproject.toml` as console scripts.

```
classivore init      — Initialize new taxonomy (validate CSV, enrich, configure)
classivore enrich    — Enrich taxonomy with LLM-generated descriptions
classivore collect   — Collect training data (Common Crawl + live scrape)
classivore validate  — Validate data quality (scraped or labeled)
classivore label     — Label scraped data with LLM
classivore train     — Train DeBERTa model
classivore classify  — Run inference (text, file, or interactive)
classivore agent     — Run data expansion agent
classivore hints     — Generate domain hints for tier1 categories
classivore publish   — Publish trained model to HuggingFace Hub
classivore hf init   — Create HuggingFace repo
classivore taxonomy  — Show taxonomy info, stats, and coverage gaps
classivore serve     — Start local API server (stub)
```

## Common Arguments

Every command accepts:
- `--taxonomy <slug>` — which taxonomy to use (e.g., `iab-2.2`)
- `--data-dir <path>` — override data directory
- `--verbose` / `-v` — increase output detail

## Argument Patterns

- Use `argparse` (not click/typer) for zero extra dependencies
- Subcommand pattern: `classivore <command> [options]`
- Environment variables as fallbacks for API keys (never as required)
- All file paths relative to data-dir or taxonomy dir

## Entry Point Registration

In `pyproject.toml`:
```toml
[project.scripts]
classivore = "cli.main:main"
```

Using subcommand dispatch in `cli/main.py`.

## Tests

- `tests/unit/test_cli.py` — test argument parsing, subcommand routing (no actual execution)
