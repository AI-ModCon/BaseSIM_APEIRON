---
name: integrate-apeiron
description: |
  Add apeiron's continual-learning / drift-detection capabilities to a user's
  EXISTING training framework. Use when the user already has their own training
  loop (vanilla PyTorch, Lightning, HF Trainer, etc.) and wants to bolt on drift
  detection and/or CL adaptation rather than adopt apeiron's runner. Inspects the
  user's repo, recommends the lightest viable integration path, writes the
  adapter glue into their repo, and smoke-tests it. Assumes apeiron is importable
  (`import apeiron`) — if not, run install-apeiron first. For a self-contained
  apeiron run on custom data, use custom-experiment instead.
argument-hint: "[target_project_dir]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

Integrate apeiron into the user's existing training framework, lightest path first.

## Arguments
- `$1`: The user's project directory. Defaults to the current working directory.

## Background: what apeiron exposes (verify against source, don't assume)
- **Drift detectors are standalone** — `detector.update(metric_value: float) -> DriftSignal`; the signal carries `drift_detected`, `regime` (`LearningRegime`), and `drift_score`. This is the lowest-coupling entry point. Build one via `from apeiron.drift_detection import ADWINDetector` (or the others).
- **ContinuousMonitor drives the full loop** but requires a `BaseModelHarness` wrapping the model + data stream, plus a `Config` and a detector. Mirror `src/main.py` for exact construction.
- **CL updaters (EWC/JVP/KFAC) are harness-coupled** — they take a `modelHarness`, so using them implies the harness/monitor route.

Read these before writing any glue so it matches the current API:
```bash
cat src/main.py                                       # full wiring reference
cat src/apeiron/drift_detection/detectors/base.py     # DriftSignal / LearningRegime fields
cat src/apeiron/drift_detection/load_drift_detector.py # building a detector from config
grep -nA12 "class ContinuousMonitor" src/apeiron/driver/continuous_monitor.py
```

## Procedure

### 1. Confirm apeiron is importable
```bash
python -c "import apeiron; print('apeiron', apeiron.__file__)" 2>&1
```
If this fails, stop and direct the user to the `install-apeiron` skill, then resume.

### 2. Discover the user's framework (detect at runtime)
In `$1`, find the training loop and the evaluation signal:
- Detect the stack: `grep -rlE "pytorch_lightning|lightning|transformers|Trainer|accelerate" $1` and look for a manual loop (`loss.backward()`, `optimizer.step()`).
- Locate where a scalar quality metric per step/epoch is available (val accuracy, loss) — this is what a detector consumes.
- Locate the model object and the data iterator (needed only if the full path is chosen).
Summarize what you found before proposing anything.

### 3. Recommend the lightest path that meets the need, and confirm
Based on what the user wants out of apeiron:
- **Just detect drift / trigger their own retrain** → *detectors-only* (no harness). Lowest coupling — recommend this unless they need apeiron's CL math.
- **Want apeiron's CL regularizers (EWC/JVP/KFAC) or the full monitor→adapt loop** → *harness + ContinuousMonitor*.
Present the recommendation with its trade-offs and get the user's pick before writing code.

### 4a. Detectors-only adapter (lightest)
Write a small module into the user's repo (e.g. `<their_pkg>/apeiron_drift.py`) that:
- Constructs a detector once (choice of ADWIN/KSWIN/PageHinkley).
- Exposes a hook called from their existing eval step: `signal = detector.update(metric); if signal.drift_detected: <their callback>`.
- Leaves the decision of what to do on drift (log / retrain / reload) to a user-provided callback.
Wire the hook into their loop with a minimal, clearly-marked edit.

### 4b. Harness + monitor adapter (full)
When CL adaptation is wanted:
- Write a `BaseModelHarness` subclass in their repo that wraps their existing model and data loaders, implementing the abstract methods (read `src/apeiron/model/torch_model_harness.py` and `examples/mnist/model.py` for the current set — includes `get_optmizer`, `update_data_stream`, `get_stream_dataloader`, `get_hist_dataloaders`, `get_train_dataloaders`, `get_criterion`).
- Build a `Config` (via `build_config` from a small TOML, or constructed directly) selecting the detector and `continual_learning.update_mode`.
- Construct and run `ContinuousMonitor` exactly as `src/main.py` does.

### 5. Smoke-test the integration
Prove the wiring with a tiny run before handing back:
- Detectors-only: a short script feeding a handful of synthetic metric values through the hook, asserting a `DriftSignal` comes back and the drift callback fires on an obvious shift.
- Full path: run their loop (or the monitor) for a couple of iterations with minimal settings (small `max_iter`, few stream updates, `device=cpu`, `logging.backend=none`).
Fix and re-run until it completes cleanly.

### 6. Report
Summarize: detected stack, chosen path, files added/edited (with the exact insertion points), how to invoke it, and what happens on drift. Note any assumptions the user should revisit (e.g. which metric drives detection, detector sensitivity params).

## Notes
- Keep edits to the user's training loop minimal and clearly commented so they remain easy to revert.
- Do not assume detector/monitor/harness signatures — they are read from source in this skill precisely so the glue doesn't rot.
