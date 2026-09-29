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

### Fixed
- README build and coverage badges pointed at the former `BaseSim_Framework` repository instead
  of `BaseSIM_APEIRON`
- Stale `BaseSim_Framework` clone URLs in the Frontier and Perlmutter deployment guides
- Typos: "Froniter", "Requirested", "beecause"

### Security

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
