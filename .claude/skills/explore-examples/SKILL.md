---
name: explore-examples
description: |
  Run a bundled apeiron example experiment to explore the framework's
  capabilities. Use when the user wants to try the software, run a default/demo
  experiment, see drift detection and continual learning in action, or pick from
  the shipped MNIST/CIFAR configs. Presents a menu of available example configs,
  runs the chosen one, and reports where the metrics CSV landed. For running the
  user's OWN data/model/config, use the custom-experiment skill instead.
argument-hint: "[config_path]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

Run a bundled apeiron example so the user can see the framework working end-to-end.

## Arguments
- `$1`: Optional path to a specific bundled config. If given, skip the menu and run it directly (still apply steps 3–5). If omitted, present the menu (step 1).

## Procedure

### 1. Build the menu dynamically (do not hardcode the list — it rots)
Discover the shipped configs and summarize each from its own contents:
```bash
find examples -name "*.toml" -type f | sort
```
For each config, read the key fields to describe it (`data.name`, `model.name`, `drift_detection.detector_name`, `continual_learning.update_mode`). Present a numbered menu like:
`1) examples/mnist/mnist.toml — MNIST, ADWIN detector, base updater`
Then ask the user which to run.

### 2. Default to MNIST; flag missing pretrained weights for others
- **MNIST is the guaranteed hands-off path** — `examples/mnist/mnist.pth` ships with the repo. Recommend it for a first run.
- For any non-MNIST choice (e.g. CIFAR), check the config's `pretrained_path` before running:
  ```bash
  ls -la <pretrained_path> 2>/dev/null || echo "MISSING"
  ```
  If the weight file is missing, tell the user plainly: this example needs weights that don't ship with the repo, so the run will train from scratch (slow) or fail to load. Let them decide whether to continue or switch to MNIST.

### 3. Ask which metrics-logging backend to use (per run)
The config default is `wandb`. Before running, ask the user to choose, and pass it as an override so no edits are needed:
- **none** — `--set logging.backend=none` (no account/network; best for a quick local look)
- **wandb** — `--set logging.backend=wandb` (run `wandb login` first if not authenticated)
- **mlflow** — `--set logging.backend=mlflow` (local tracking by default)

### 4. Show the config and run it
- Briefly summarize the chosen config (dataset, model, detector, updater, device, batch size) so the user can confirm.
- Run from the project root:
  ```bash
  poetry run python -m src.main --config <config_path> --set logging.backend=<choice>
  ```
- This is a real training/monitoring run and may take a while. Stream output; do not silently background it.

### 5. Report results
- Summarize from the run output: whether drift was detected and how many times, final accuracy, and the output CSV path (the config's `visualization.input`).
- The package emits this CSV for inspection; it does not ship a built-in dashboard renderer, so point the user at the CSV for further plotting.

## Notes
- Quick first run, copy-paste safe: `poetry run python -m src.main --config examples/mnist/mnist.toml --set logging.backend=none`
- Useful overrides to demonstrate capabilities: `--set drift_detection.detector_name=PageHinkleyDetector`, `--set continual_learning.update_mode=ewc_online`, `--set device=cpu`.
- If `poetry` isn't set up yet, point the user at the install/dev-setup step first.
