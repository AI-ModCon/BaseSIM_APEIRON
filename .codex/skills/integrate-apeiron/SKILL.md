---
name: integrate-apeiron
description: Add apeiron drift-detection or continual-learning behavior to an existing training framework. Use when the user already has a training loop, such as vanilla PyTorch, Lightning, Hugging Face Trainer, or Accelerate, and wants to integrate apeiron rather than adopt apeiron's runner. Inspects the target repo, recommends the lightest viable integration path, writes adapter glue, and smoke-tests it. If `import apeiron` fails, use install-apeiron first.
metadata:
  short-description: Integrate apeiron into a training loop
---

# Integrate Apeiron

Integrate apeiron into an existing training framework with the least coupling that meets the user's goal.

## Input

- Target project directory: use the path the user gives, otherwise the current working directory.

## Background

Verify APIs against source before writing glue:

```bash
cat src/main.py
cat src/apeiron/drift_detection/detectors/base.py
cat src/apeiron/drift_detection/load_drift_detector.py
grep -nA12 "class ContinuousMonitor" src/apeiron/driver/continuous_monitor.py
```

Current conceptual paths:

- Drift detectors are standalone: `detector.update(metric_value: float) -> DriftSignal`.
- `DriftSignal` carries fields such as `drift_detected`, `regime`, and `drift_score`; confirm exact fields in source.
- `ContinuousMonitor` drives the full apeiron loop and requires a `BaseModelHarness`, config, and detector.
- CL updaters such as EWC, JVP, and KFAC are harness-coupled, so using them implies the harness and monitor path.

## Procedure

### 1. Confirm Apeiron Is Importable

From the target project environment, run:

```bash
python -c "import apeiron; print('apeiron', apeiron.__file__)"
```

If this fails, stop and use or recommend `install-apeiron` before continuing.

### 2. Discover The Framework

In the target project, locate:

- framework indicators such as `pytorch_lightning`, `lightning`, `transformers`, `Trainer`, or `accelerate`
- manual PyTorch loops with `loss.backward()` and `optimizer.step()`
- the scalar quality metric available per step or epoch, such as validation accuracy or loss
- the model object and data iterator when the full monitor path may be needed

Summarize what you found before proposing changes.

### 3. Choose The Integration Path

Recommend the lightest path that satisfies the user's goal:

- Detectors-only adapter: best when the user wants drift detection or wants to trigger their own retraining.
- Harness plus `ContinuousMonitor`: required when the user wants apeiron CL regularizers or the full monitor-to-adapt loop.

Confirm the chosen path with the user before editing their training loop.

### 4. Detectors-Only Adapter

For the light path, add a small module such as `<their_pkg>/apeiron_drift.py` that:

- constructs one detector, such as ADWIN, KSWIN, or PageHinkley
- exposes a hook called from the existing eval step
- calls `signal = detector.update(metric)`
- calls a user-provided callback when `signal.drift_detected`
- leaves the drift response, such as log, retrain, or reload, to the user's callback

Wire the hook into the loop with a minimal, clearly marked edit.

### 5. Harness And Monitor Adapter

For the full path:

- Write a `BaseModelHarness` subclass wrapping the existing model and data loaders.
- Read `src/apeiron/model/torch_model_harness.py` and `examples/mnist/model.py` for the current abstract methods.
- Preserve current method names exactly, including `get_optmizer` if that is what the ABC declares.
- Build a config with `build_config` from a small TOML or construct the current config object directly.
- Construct and run `ContinuousMonitor` using `src/main.py` as the wiring reference.

### 6. Smoke-Test

Prove the wiring before handing back:

- Detectors-only: run a short script feeding synthetic metric values through the hook and assert that a `DriftSignal` is returned. Use an obvious shift when trying to exercise the callback.
- Full path: run only a tiny CPU-bound loop or monitor pass with minimal iterations, no network logging, and small stream settings.

Read failures, fix the glue, and re-run until the smoke test passes.

### 7. Report

Report:

- detected framework
- chosen integration path
- files added or edited
- exact insertion point in the loop
- how to run the smoke test
- what happens when drift is detected
- assumptions the user should revisit, especially the detection metric and detector sensitivity

## Notes

- Keep edits to the user's loop minimal and easy to revert.
- Do not hardcode detector, monitor, or harness signatures; read them from source first.
