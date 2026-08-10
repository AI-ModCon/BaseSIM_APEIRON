# Apeiron Documentation

This directory is a [Sphinx](https://www.sphinx-doc.org/) project written in
[MyST-Markdown](https://myst-parser.readthedocs.io/) and published on
Read the Docs. Every `.md` file here is a page in that site; `conf.py` and
`../.readthedocs.yaml` configure the build.

## Building locally

Heavy runtime dependencies (torch, river, evidently, wandb, ...) are mocked in
`conf.py`, so a docs build does **not** need the full project environment:

```bash
python -m venv .venv-docs && source .venv-docs/bin/activate
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

Add `-W` to turn warnings into errors, and `-a -E` to force a full rebuild after
changing `conf.py`.

## Page map

| Page | Contents |
| --- | --- |
| `index.md` | Landing page and the toctrees that define site navigation. |
| `installation.md` | Python/Poetry setup, using Apeiron as a dependency, dev commands. |
| `quickstart.md` | First run, reading the metrics CSV, config overrides. |
| `architecture.md` | Runtime flow, module map, the four extension points. |
| `configurations.md` | Every TOML section and key the config parser accepts. |
| `model_harness.md` | Model + data-stream integration contract. |
| `drift_detectors.md` | Detector classes, options, and wiring. |
| `choosing_a_detector.md` | Decision guide for picking and tuning a detector. |
| `continuous_learning.md` | CL trainer, updater modes, training config. |
| `tracking.md` | W&B / MLflow backends, logged metric namespace, reading charts. |
| `profiler.md` | FLOPS profiler and the `cperf_*` metrics. |
| `deployment.md` | Frontier / Perlmutter HPC setup (included from the deployment READMEs). |
| `agent_skills.md` | The Claude Code and Codex skills shipped with the repo. |
| `api/` | Autodoc API reference generated from `src/apeiron/` docstrings. |

## Conventions

- `profiler.md` and `deployment.md` use `{include}` to pull in READMEs that live
  next to the code, so those pages stay in sync with the scripts they document.
- Prefer `{doc}` / `{ref}` cross-references over raw relative links so Sphinx
  can validate them at build time.
- New pages must be added to a toctree in `index.md`, otherwise Sphinx warns
  that the document is not included in any toctree.
