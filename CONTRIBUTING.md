# Contributing to Apeiron

Thank you for your interest in contributing to Apeiron! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps** — the TOML config you ran, and the `python -m src.main --config ...` invocation
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible** — W&B or MLflow charts are often the clearest way to show a drift/adaptation problem
- **Include your environment details** (OS, Python version, accelerator and CUDA/ROCm version, torch version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and the proposed behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

- Follow the Python code style guidelines (see below)
- Document new code based on the Documentation Styleguide
- End all files with a newline
- Avoid platform-dependent code
- Add tests for any new functionality
- Include appropriate commit messages
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change

## Development Setup

Apeiron requires Python `>=3.13,<3.14` and uses [Poetry](https://python-poetry.org/) for
dependency management.

1. Fork the repository
2. Clone your fork: `git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git`
3. Create a new branch: `git checkout -b feature/my-feature`
4. Set up development environment:
   ```bash
   poetry install
   ```
5. Make your changes
6. Run the checks:
   ```bash
   make lint        # ruff check + ruff format --check
   make type-check  # mypy
   make test        # pytest
   ```
   Run `make help` to see all available targets. The equivalent direct commands are
   `poetry run ruff check .`, `poetry run ruff format --check .`, `poetry run mypy .`, and
   `poetry run pytest`.
7. Commit your changes: `git commit -am "Add my feature"`
8. Push to the branch: `git push origin feature/my-feature`
9. Submit a pull request

## Style Guidelines

### Python Code Style

- Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) as the coding standard
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Use type hints everywhere — `mypy` runs in CI
- Prefer frozen dataclasses for configuration objects
- Extension points follow the ABC pattern (`BaseModelHarness`, `BaseDriftDetector`,
  `BaseUpdater`); dynamic loading goes through the existing factory functions
  (`get_example`, `create_updater`, `load_drift_detector`)
- Enforce with: `make lint` (lint and format check) and `make format` (apply formatting)

### Docstrings

- Document all public functions, classes, and modules with docstrings
- Use [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  docstrings — the Sphinx build runs `sphinx.ext.napoleon`, and the `docs/api/` pages are
  generated from these docstrings by autodoc
- Include parameter descriptions, return types, and examples where helpful

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Documentation

- Use MyST-Markdown for documentation — `docs/` is a Sphinx project published on Read the Docs
- Keep documentation up-to-date with code changes
- Write clear, concise documentation
- Include code examples where appropriate
- New pages must be added to a toctree in `docs/index.md`, and the Read the Docs build runs with
  `fail_on_warning: true`, so build locally before opening a PR. See
  [`docs/README.md`](./docs/README.md) for the local build and page conventions.
- API reference pages are auto-generated from docstrings; adding a new public module usually means
  adding a page under `docs/api/`

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage
- Use descriptive test names
- Shared fixtures live in `tests/conftest.py` — reuse `tiny_model`, `tiny_cnn`, `dummy_harness`,
  and `make_harness` rather than building new throwaway models
- Mark long-running tests with `@pytest.mark.slow` so they can be deselected with
  `-m "not slow"`

## Guidelines for AI/LLM-Assisted Contributions

- **Remain accountable for all your outputs and decisions.**
   Individuals remain fully responsible and accountable for the accuracy, quality, appropriateness, and consequences of their work. Use of AI does not transfer this responsibility to the AI model, agent, or other tool.
- **Understand your work.**
   Regardless of how code or PR was produced, this project requires that authors illustrate a thorough understanding of any proposed changes. You must review such code line-by-line; it is your responsibility to ensure that it is correct, and that it does not breach copyright. Always critically engage with AI outputs, do not trust them implicitly. AI-assisted code, analysis, and artifacts must be tested and validated at a level appropriate to their impact. Authors are responsible for ensuring that generated code is correct, secure, maintainable, non-obfuscated, appropriately scoped, documented, and reproducible where relevant.
- **Disclose AI-generated or AI-assisted work.**
   If AI/LLM tools were primarily used to generate code or artifacts, this should be clearly indicated in the PR.
- **Use of AI to review PRs.**
   All PRs must be reviewed by a human reviewer. An LLM review may be used in addition to a human reviewer since this can help spot issues that a human may have missed, but this should not be the sole reviewer. The human reviewer should be fully accountable and responsible for the review feedback or comments (see 1).
- **Proprietary or personal information.**
   For this project, proprietary or personal information should never be sent to code generators or AI tools.
- **Be transparent, assume goodwill, and share what you learn.**
   Contributors should be open about relevant AI use, disclose details of AI use as appropriate to the project, engage constructively with colleagues, and share experiences and lessons learned with the project.

This repository ships agent skills for Claude Code and Codex under `.claude/skills/` and
`.codex/skills/` (see [`docs/agent_skills.md`](./docs/agent_skills.md)). Those skills are tools for
working with Apeiron; using them does not change any of the responsibilities above.

## Questions?

Feel free to open an issue with the label `question` if you have any questions.

Thank you for contributing!
