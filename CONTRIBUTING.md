# Contributing to Classivore

Thanks for your interest in contributing. This guide covers the basics of how the project is organized and what we expect in a pull request.

## Branching

- `main` — release-tagged versions only. Updated from `develop` via a `--no-ff` merge commit at release time, then tagged `v{X.Y.Z}`.
- `develop` — integration branch. All feature branches target `develop`.
- `feature/<short-name>` — your work goes here. Branch from `develop`, PR back to `develop` (squash-merge).

Never commit directly to `main` or `develop`.

**Why `--no-ff` for `develop` → `main` (and not squash):** squash-merging a long-lived branch into another long-lived branch breaks the merge-base, which causes phantom conflicts (most visibly in `CHANGELOG.md`) on the next release. Squashing is fine `feature` → `develop` because the feature branch dies after the merge.

## Pull request checklist

- [ ] Branch from `develop`, not `main`
- [ ] Tests pass locally: `pytest`
- [ ] No `.env`, secrets, or large data files staged (check `git status` before committing)
- [ ] Commit messages use [gitmoji](https://gitmoji.dev) prefixes (e.g. `✨ Add ...`, `🐛 Fix ...`, `📝 Update ...`)
- [ ] PR description includes a Summary and a Test plan section
- [ ] CHANGELOG entry added under an `## [Unreleased]` heading if your change is user-visible

## Setting up a development environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in the keys you need
pytest
```

You only need keys for the features you're touching. The Anthropic key is required for any LLM-driven step (enrichment, labeling, agent loop). Search-provider keys are only needed for collection. The HuggingFace token is only needed for `classivore publish`.

## Code style

- `ruff check .` — must be clean (config in `pyproject.toml`, line length 100)
- Keep functions small. If a function needs a comment to explain what it does, it's probably two functions.
- All Python files start with `#!/usr/bin/env python3`.
- All source code lives in `src/classivore/`. No standalone scripts in the repo root.

## Tests

- Every module has tests under `tests/unit/`.
- Use real I/O against `tmp_path` rather than mocking the filesystem when possible.
- New collection/labeling/training features should ship with at least one test that exercises the happy path and one for an edge case.

## Reporting issues

Open a GitHub issue with:

- What you expected to happen
- What actually happened (full stack trace if applicable)
- Minimum command or code that reproduces it
- Your Python version and OS

Issues with reproduction steps get triaged faster than vague reports.

## Licensing

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
