<!--
Title tip (recommended): use Conventional Commits, e.g.
feat(train): add cosine LR schedule
fix(io): handle empty files in reader
-->

## Summary
<!-- What does this PR do in 1–3 sentences? Why now? -->

## Motivation & Context
<!-- Link issues, discussions, incidents, design docs. -->
E.g. Fixes issue nr xyz

## Approach
<!-- High-level description of the solution. Point out key design choices and trade-offs. -->

## Screenshots / Logs (optional)
<!-- Paste images or brief logs if helpful for reviewers. -->

## API / CLI Changes
<!-- List any new/changed public functions, classes, dataclasses, CLI flags, env vars. -->
- `foo.bar(x: int) -> str` (new)
- `baz(qux: PathLike)` (removed `strict: bool`)

## Breaking Changes
<!-- Describe exactly what breaks and how users migrate. If none, write “None”. -->
- None

## Performance (optional)
<!-- Provide before/after numbers if perf-sensitive. Include methodology. -->
| Case | Before | After | Notes |
|------|--------|-------|-------|
| foo() | 123 ms | 88 ms | Median of 50 runs |

## Security & Privacy
<!-- Secrets removed? Inputs validated? Untrusted data paths safe? -->
- [ ] No secrets committed
- [ ] Input validation added where needed

## Dependencies
<!-- New or removed packages, wheels, system deps. -->
- Added: `orjson>=3.10`
- Removed: `ujson`

## Testing Plan
<!-- How did you verify this change? Add reproduction steps for reviewers. -->
- [ ] Unit tests
- [ ] Integration tests
- [ ] e2e / smoke test
- [ ] Manual steps: `python -m app --help`

## Documentation
<!-- Docs and examples updated? Docstrings? Changelog entry? -->
- [ ] Docstrings updated
- [ ] User docs / README updated
- [ ] CHANGELOG entry

## Checklist
- [ ] Code formatted (Ruff) → `ruff format --check`
- [ ] Lint passes (Ruff) → `ruff check .`
- [ ] Types pass (mypy/pyright) → `mypy src`
- [ ] Tests pass (pytest) → `pytest -q`
<!-- #- [ ] Pre-commit passes → `pre-commit run -a` -->
- [ ] Backward compatibility considered
- [ ] Adequate comments for tricky parts
- [ ] CI green

## Risk & Rollback Plan
Probably not needed in the beginning
<!-- Risk level (low/med/high). How to rollback if needed (revert PR, feature flag, config switch)? -->

## Notes for Reviewers
<!-- Anything you want the reviewer to focus on; files to review first; areas you're unsure about. -->
