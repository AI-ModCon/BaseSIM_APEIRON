# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community health files: `CONTRIBUTING.md` (including guidelines for AI/LLM-assisted
  contributions), `CODE_OF_CONDUCT.md`, `SECURITY.md`, and this changelog
- GitHub issue templates for bug reports and feature requests
- `Makefile` with `install`, `test`, `test-cov`, `lint`, `format`, `type-check`, `docs`, and
  `clean` targets
- Package metadata in `pyproject.toml`: `license`, `keywords`, `classifiers`, and
  `[project.urls]`
- Acknowledgment of U.S. Department of Energy Genesis Mission support in the README and docs
- `BaseModelHarness.get_optimizer()`, the correctly spelled replacement for
  `get_optmizer()`

### Changed
- Migrated dependency management from Poetry to [uv](https://docs.astral.sh/uv/). `poetry.lock`
  is replaced by `uv.lock`, the build backend is now `uv_build`, and `uv sync` replaces
  `poetry install`. CI, the Dockerfile, the Makefile, the HPC install scripts, the docs, and the
  agent skills are updated to match. CI and Docker now install with `uv sync --locked`, which
  fails when `uv.lock` is out of date.
- `[project.license]` uses a PEP 639 SPDX expression (`Apache-2.0`) with `license-files`; the
  deprecated `License ::` classifier is removed.
- `make docs` builds in a throwaway uv environment instead of a hand-made `.venv-docs`.
- Copyright holder filled in on the Apache 2.0 `LICENSE` appendix
- CI now also runs on `pull_request`, so status checks are reported on pull requests
- Renamed `tests/test_valiadation_tests.py` to `tests/test_validation_tests.py`

### Deprecated
- `BaseModelHarness.get_optmizer()` is deprecated in favour of `get_optimizer()` and will be
  removed in a future release. Existing harnesses keep working: a subclass that implements
  `get_optmizer()` is bridged onto `get_optimizer()` automatically and raises a
  `DeprecationWarning` at class definition. Calling `get_optmizer()` directly also still works
  and warns.

### Removed
- `black` from the dev dependency group. Nothing used it — formatting is enforced with
  `ruff format` — and it carried three advisories.

### Fixed
- The Poetry lock was unsatisfiable on Linux: `torch` 2.13 reaches `nvidia-cublas` and
  `nvidia-cuda-nvrtc` through two paths with differently shaped markers, and Poetry locked two
  versions of each whose markers overlap. uv resolves a single version, which is what broke the
  CI install and `poetry install` on Linux.
- The Frontier and Perlmutter install scripts ran `poetry lock` on the cluster, re-resolving
  dependencies at install time instead of using the lock. They now run `uv sync --locked`.
- The Frontier install script pinned `torch==2.9.1`, left over from before the `torch` upgrade.
  It now reads the `torch` and `torchvision` versions from `uv.lock`.
- Install docs suggested `pip install apeiron` / `apeiron = "^0.1.0"`; the `apeiron` name on PyPI
  belongs to an unrelated project. They now install from GitHub.
- README build and coverage badges pointed at the former `BaseSim_Framework` repository instead
  of `BaseSIM_APEIRON`
- Stale `BaseSim_Framework` clone URLs in the Frontier and Perlmutter deployment guides
- Typos: "Froniter", "Requirested", "beecause"

### Security
- Updated locked dependencies to clear 121 of 122 open Dependabot alerts (6 critical, 69 high,
  38 medium, 9 low). Notable bumps: `torch` 2.9.1 -> 2.13.0, `torchvision` 0.24.1 -> 0.28.0,
  `transformers` 5.8.1 -> 5.17.0, `mlflow` 3.12.0 -> 3.16.1, `nltk` 3.9.4 -> 3.10.3,
  `gitpython` 3.1.50 -> 3.1.62, `aiohttp` 3.13.5 -> 3.14.3, `pillow` 12.2.0 -> 12.3.0,
  `cryptography` 46.0.7 -> 50.0.1, `starlette` 0.52.1 -> 1.7.0.
- Widened the `torch` and `torchvision` constraints to `>=2.13.0` and `>=0.28.0,<0.29.0`.
  `torchvision` pins `torch` exactly, so the previous `torchvision <0.25.0` bound held `torch`
  at 2.9.1 and blocked four upstream fixes.
- One alert remains open and has no published fix: `nltk` GHSA-8mgp-746c-j5xp (high). `nltk`
  3.10.3 is the latest release and is still listed as affected. It reaches the project
  transitively through `evidently`; Apeiron does not call `nltk` directly.

## [0.1.0]

### Added
- Initial release of Apeiron
- Drift detection over a live data stream with statistical and model-performance detectors
- Continual-learning updaters: `base`, `ewc_online`, `kfac_online`, `jvp_reg`, and `none`
- Model harness extension point for integrating custom models and data streams
- TOML-driven experiment configuration
- Metrics logging to Weights & Biases and MLflow
- FLOPS profiler and `cperf_*` metrics
- HPC deployment scripts for Frontier and Perlmutter
- Sphinx documentation published on Read the Docs
- Agent skills for Claude Code and Codex
