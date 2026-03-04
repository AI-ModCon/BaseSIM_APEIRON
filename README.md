# SIM: Self Improving Model framework
[![Build Status](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml/badge.svg)](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml)
[![Coverage Status](https://codecov.io/gh/AI-ModCon/BaseSim_Framework/badge.svg?branch=main)](https://codecov.io/gh/AI-ModCon/BaseSim_Framework?branch=main)

A PyTorch framework for continuous learning that automatically detects concept drift in data streams and adapts models through JVP regularized retraining.

## What This Repository Does

The pipeline runs on a changing data stream and loops through these stages:

1. Evaluate the current model on stream batches.
2. Aggregate monitored metrics at a configured interval.
3. Run a drift detector on the aggregated metric.
4. If drift is detected, pause monitoring and run a continual-learning update loop.
5. Resume monitoring on the updated model and continue until stream limits are reached.

Core modules:

- `src/main.py`: entry point
- `src/config/configuration.py`: TOML/env/CLI config assembly
- `src/driver/continuous_monitor.py`: monitoring + drift loop
- `src/training/continuous_trainer.py`: CL training loop
- `src/training/updater/`: CL update strategies
- `src/drift_detection/`: detectors and detector factory
- `examples/`: concrete model harness implementations

## Installation

Requires Python `>=3.13,<3.15` and Poetry.

```bash
poetry install
```

## Running Experiments

From the project root:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml
poetry run python -m src.main --config examples/cifar/cifar10_vit.toml
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

## Current Integration Notes

- `EnsembleDetector` exists but is not wired in `load_drift_detector(...)`.
- `ModelPerformanceDetector` and `EvalDetector` need additional runtime wiring for full monitor integration.
