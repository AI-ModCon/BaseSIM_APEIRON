# Apeiron
[![Build Status](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml/badge.svg)](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml)
[![Coverage Status](https://codecov.io/gh/AI-ModCon/BaseSim_Framework/badge.svg?branch=main)](https://codecov.io/gh/AI-ModCon/BaseSim_Framework?branch=main)

A PyTorch framework for continual learning that automatically detects concept drift in data streams and adapts models through JVP regularized retraining.

## What This Repository Does

The pipeline runs on a changing data stream and loops through these stages:

1. Evaluate the current model on stream batches.
2. Aggregate monitored metrics at a configured interval.
3. Run a drift detector on the aggregated metric.
4. If drift is detected, pause monitoring and run a continual-learning update loop.
5. Resume monitoring on the updated model and continue until stream limits are reached.

Core modules:

- `src/main.py`: entry point
- `src/apeiron/config/configuration.py`: TOML/env/CLI config assembly
- `src/apeiron/driver/continuous_monitor.py`: monitoring + drift loop
- `src/apeiron/training/continuous_trainer.py`: CL training loop
- `src/apeiron/training/updater/`: CL update strategies
- `src/apeiron/drift_detection/`: detectors and detector factory
- `examples/`: standalone example projects (each declares `apeiron` as a dependency)

## Installation

### As a dependency in your project

```toml
# pyproject.toml
[tool.poetry.dependencies]
apeiron = "^0.1.0"  # once published to PyPI

# Or as a path dependency during development
apeiron = { path = "../apeiron/", develop = true }
```

```python
from apeiron import BaseModelHarness, ContinuousMonitor, build_config
from apeiron.drift_detection import ADWINDetector
from apeiron.training.updater import BaseUpdater
```

### For development in this repo

Requires Python `>=3.13,<3.14` and Poetry.

```bash
poetry install
```

## Running Experiments

From the project root:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml
poetry run python -m src.main --config examples/cifar/cifar10_vit.toml
poetry run python -m src.main --config examples/imagenet/imagenet_vit.toml  # requires ImageNet data at data.path
```

## Metrics Logging
Currently, we support two metrics logging backends: Weights & Biases (WandB) and MLflow. You can configure the desired `backend` in the `config` file's `logging` section. To disable logging, you can set the `logging` section to `none` to disable logging. Alternatively, you can set the logging choice via command line arguments, for example:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml --set logging.backend=mlflow --set logging.experiment_name="My Experiment"
# To view results for MLflow, run `mlflow ui` in another terminal and navigate to http://localhost:5000
```

Currently the mnist example sets the logging to wandb in the toml `config` file. The other examples do not set any metric for the logging backend, which defaults to wandb.

## Configuration Overview

Primary sections in config TOML:

- `[model]`
- `[data]`
- `[train]`
- `[drift_detection]`
- `[continual_learning]` (optional but recommended)
- `[visualization]` (optional)

Top-level fields commonly used:

- `seed`
- `device`
- `multi_gpu`
- `verbosity`

Override precedence:

1. Base TOML (`--config`)
2. Environment overrides prefixed with `APP_`
3. CLI overrides via repeated `--set key=value`

Example override:

```bash
poetry run python -m src.main \
  --config examples/mnist/mnist.toml \
  --set drift_detection.detector_name=\"KSWINDetector\" \
  --set train.max_iter=200
```

## Agent Skills (Claude Code & Codex)

This repo ships task-oriented **agent skills** that walk an AI coding agent
through the common Apeiron workflows. The same four skills are maintained for
both tools:

- **Claude Code** — `.claude/skills/<name>/SKILL.md`
- **Codex** — `.codex/skills/<name>/SKILL.md`

| Skill | What it does |
|---|---|
| `install-apeiron` | Add Apeiron as a dependency to **another** project (path/git), verify `import apeiron`, pick CPU vs CUDA PyTorch. |
| `explore-examples` | Run a bundled example (MNIST/CIFAR) to see drift detection + CL in action; picks a config and reports the metrics CSV. |
| `custom-experiment` | Scaffold a harness, data utils, and TOML for **your own** dataset/model, register it in the example factory, smoke-test, and run. |
| `integrate-apeiron` | Add Apeiron's drift detection / CL to an **existing** training loop; inspects your repo and writes the lightest adapter that fits. |

### Using them

**Claude Code** — the skills are exposed as slash commands. Type `/` and the
skill name, e.g.:

```
/explore-examples
/install-apeiron ../my-project
```

You can also just describe the task in plain language ("add apeiron to my
training loop") and the matching skill triggers from its description.

**Codex** — the equivalent skills live under `.codex/skills/`. Invoke a skill by
name or describe the task; Codex selects the skill whose description matches your
request. The skills are tool-agnostic in intent — only the file format differs
between the two trees.

> Keep the two trees in sync: a change to a workflow should be reflected in both
> `.claude/skills/<name>/SKILL.md` and `.codex/skills/<name>/SKILL.md`.

## Documentation

Detailed docs are in `docs/`:

- `docs/README.md`
- `docs/model_harness.md`
- `docs/drift_detectors.md`
- `docs/continuous_learning.md`

## Development Commands

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy .
```


### What `main.py` Does
- Builds the `DummyCNN_MNIST` model defined in `src/model/DummyCNN_MNIST.py`, a cross-entropy loss, and an Adam optimizer.
- Loads the MNIST training split, stacks the tensors, and iterates over 10 tasks (digits 0–9). Each task applies random rotation and translation to encourage continual adaptation.
- Maintains replay buffers (`memory_image`, `memory_label`, etc.) so past samples remain available for rehearsal while training new tasks.
- Calls `CL(...)` to assemble task-specific dataloaders and drive the `One_task_CL` loop. The loop trains for five epochs, records loss/accuracy metrics, and prints periodic progress reports.
- Computes sensitivity scores with `src/validation/validation_utils/return_score` after each task; you can repurpose these values for analysis or adaptive triggers.

## Tuning Tips
- Change the number of epochs by editing `n_epoch` inside `CL`.
- Adjust replay/adversarial update counts through the `params` dictionaries in `One_task_CL` and `util.update_CL_`.
- Experiment with different transforms or task definitions by modifying `data.py`.
- Update batch sizes by changing the `batch_size` parameter used when constructing the dataloaders.

## Output
Training logs report the task id, training/test accuracy, and replay-memory accuracy every five epochs. Accuracy is computed via `test(...)` on both the current task and the accumulated memory set.

## Deployment

Platform-specific deployment guides:

- [OLCF Frontier](./src/apeiron/deployment/frontier/README.md)
- [NERSC Perlmutter](./src/apeiron/deployment/perlmutter/README.md)