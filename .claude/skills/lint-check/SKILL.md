---
name: lint-check
description: |
  Run lint, format, and type checks on the BaseSim codebase. Use when the user
  wants to verify code quality before committing, or after making changes.
  Runs ruff linter, ruff formatter, mypy type checker, and pytest.
argument-hint: "[fix]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
---

Run code quality checks on the BaseSim Framework.

## Arguments
- `$0`: (Optional) Pass "fix" to auto-fix lint and format issues where possible.

## Checks to Run

### 1. Ruff Linter
```bash
poetry run ruff check .
```
If `$0` is "fix":
```bash
poetry run ruff check --fix .
```

### 2. Ruff Formatter
```bash
poetry run ruff format --check .
```
If `$0` is "fix":
```bash
poetry run ruff format .
```

### 3. Mypy Type Checker
```bash
poetry run mypy .
```

### 4. Pytest
```bash
poetry run pytest -q
```

## Procedure

1. Run all four checks sequentially, capturing output from each.
2. Report a summary:
   - Number of lint issues found (and auto-fixed if "fix" mode)
   - Number of format issues
   - Number of type errors
   - Number of test failures
3. If "fix" was specified, re-run the check variants after fixing to confirm resolution.
4. Report any remaining issues that require manual attention.
