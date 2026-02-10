---
name: run-experiment
description: |
  Run a BaseSim continual learning experiment. Use when the user wants to
  execute a training/monitoring run with a TOML config file. Supports
  overriding config values. Examples: run MNIST experiment, run CIFAR with
  EWC updater, run experiment with PageHinkley detector.
argument-hint: "<config_path> [--set key=val ...]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

Run a BaseSim continual learning experiment.

## Arguments
- `$0`: Path to a TOML config file (e.g., `examples/mnist/mnist.toml`). If not provided, list available configs and ask.
- Remaining arguments: Optional `--set key=val` overrides passed through to the runner.

## Available Configs
- `examples/mnist/mnist.toml` -- MNIST with ADWIN detector
- `examples/mnist/mnist-generic.toml` -- MNIST generic
- `examples/cifar/cifar10_vit.toml` -- CIFAR-10 with Vision Transformer
- `examples/cifar/cifar10_vgg11.toml` -- CIFAR-10 with VGG11

## Common Overrides
- Change detector: `--set drift_detection.detector_name=KSWINDetector`
- Change CL updater: `--set continual_learning.update_mode=ewc_online`
- Change batch size: `--set train.batch_size=128`
- Change learning rate: `--set train.init_lr=0.0001`
- Set max CL iterations: `--set train.max_iter=300`
- Force CPU: `--set device=cpu`
- Change max stream updates: `--set drift_detection.max_stream_updates=50`

## Procedure

1. **Validate config exists.** If `$0` is empty or not a valid file path, list available configs:
   ```bash
   find examples/ -name "*.toml" -type f
   ```
   Then ask the user which config to use.

2. **Show the config** so the user can confirm settings before running:
   Read the TOML file and display a summary of key settings (dataset, detector, updater, batch size, device).

3. **Check Poetry environment** is ready:
   ```bash
   poetry env info --path 2>/dev/null || echo "Poetry env not found. Run: poetry install"
   ```

4. **Run the experiment**:
   ```bash
   poetry run python -m src.main --config $ARGUMENTS
   ```
   The `$ARGUMENTS` variable includes the config path and any `--set` overrides.

5. **Report results** after completion:
   - Whether drift was detected and how many times
   - Final accuracy metrics (from terminal output)
   - Path to the output CSV file (from the config's `visualization.input` field)
   - Suggest running `/visualize` to generate a dashboard from the results
