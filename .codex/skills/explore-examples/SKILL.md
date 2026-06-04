---
name: explore-examples
description: Run a bundled apeiron example experiment to explore the framework. Use when the user wants to try apeiron, run a default or demo experiment, see drift detection and continual learning behavior, or choose from shipped MNIST/CIFAR configs. Presents available example configs, runs the chosen one, and reports the metrics output. For the user's own data and model, use custom-experiment instead.
metadata:
  short-description: Run bundled apeiron examples
---

# Explore Examples

Run one of apeiron's bundled examples end to end.

## Inputs

- Optional config path: if the user provides one, run that bundled config.
- If no config path is provided, discover the available configs and let the user choose.

## Procedure

### 1. Discover Configs

Build the menu dynamically from the repo:

```bash
find examples -name "*.toml" -type f | sort
```

For each config, read enough TOML fields to summarize it:

- `data.name`
- `model.name`
- `drift_detection.detector_name`
- `continual_learning.update_mode`

Present a numbered menu and ask the user which config to run. Recommend MNIST for a first run when no preference is given.

### 2. Check Pretrained Weights

- MNIST is the expected low-friction path when `examples/mnist/mnist.pth` exists.
- For non-MNIST configs, read any configured `pretrained_path`.
- If the referenced file is missing, tell the user plainly that the run may train from scratch or fail to load weights, then ask whether to continue or switch configs.

### 3. Choose Logging Backend

Before running, ask which metrics backend to use and pass it as an override instead of editing the config:

- `none`: no account or network, best for local smoke runs.
- `wandb`: requires an authenticated Weights & Biases session.
- `mlflow`: uses MLflow tracking.

Default to `none` when the user asks for a quick local run.

### 4. Summarize And Run

Summarize the selected config: dataset, model, detector, updater, device, and batch size.

Run from the project root:

```bash
poetry run python -m src.main --config <config_path> --set logging.backend=<choice>
```

This is a real training and monitoring run. Stream output and do not silently background it.

### 5. Report Results

Summarize from the run output:

- whether drift was detected
- number of drift events when available
- final accuracy or final reported metric
- output CSV path from `visualization.input`

The package emits a CSV for inspection; it does not ship a built-in dashboard renderer.

## Useful Commands

Quick local first run:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml --set logging.backend=none
```

Useful overrides:

```bash
--set drift_detection.detector_name=PageHinkleyDetector
--set continual_learning.update_mode=ewc_online
--set device=cpu
```

If Poetry is not set up, complete the repo's development install before running examples.
