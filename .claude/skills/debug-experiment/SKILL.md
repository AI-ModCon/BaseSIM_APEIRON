---
name: debug-experiment
description: |
  Debug a failed or misbehaving BaseSim experiment. Use when the user reports
  errors, unexpected behavior, no drift being detected, poor accuracy, OOM
  errors, or other issues with an experiment run.
argument-hint: "<config_path_or_error_description>"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

Debug a failed or misbehaving BaseSim continual learning experiment.

## Arguments
- `$ARGUMENTS`: Either a config file path, an error message, or a description of the problem.

## Diagnostic Checklist

### 1. Environment Issues
- Check Poetry environment: `poetry env info`
- Check Python version (requires >=3.13): `python --version`
- Verify dependencies installed: `poetry check`
- Install if needed: `poetry install`

### 2. Config Parsing Errors
- Validate TOML syntax: `python -c "import tomllib; tomllib.load(open('<config>', 'rb'))"`
- Required sections: `[model]`, `[data]`, `[train]`, `[drift_detection]`
- Valid `update_mode` values: `base`, `jvp_reg`, `ewc_online`, `kfac_online`, `none`
- Valid `detector_name` values: `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector`, `ModelPerformanceDetector`, `ModelEvalDetector`, `EnsembleDetector`

### 3. No Drift Detected
- Check `detection_interval` is > 0 (0 disables detection)
- Check `max_stream_updates` is sufficient (default 20)
- Detector sensitivity tuning:
  - **ADWIN**: Lower `adwin_delta` for more sensitivity (default 0.002)
  - **KSWIN**: Lower `kswin_alpha` (default 0.005), increase `kswin_window_size` (default 100)
  - **PageHinkley**: Lower `ph_threshold` (default 50), lower `ph_delta` (default 0.005)
- Check `metric_index` matches the intended metric (0=first eval metric, typically accuracy)
- Check `aggregation` method: "mean", "median", or "last"

### 4. Too Many False Drift Detections
- Increase detector thresholds (higher delta/alpha/threshold values)
- Increase `detection_interval` to aggregate more batches before checking
- Switch `aggregation` to "mean" for a smoother signal

### 5. CUDA / Device Errors
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
- Try `device = "cpu"` in config as fallback
- For OOM: reduce `batch_size`, reduce `grad_accumulation_steps`
- Check nvidia-smi: `nvidia-smi`

### 6. Model Loading Errors
- Check `pretrained_path` exists: `ls -la <path>`
- Known pretrained paths:
  - MNIST: `examples/mnist/mnist.pth`
  - CIFAR ViT: `examples/cifar/cifar10_vit.pth`
  - CIFAR VGG: `examples/cifar/cifar10_vgg11.pth`
- For state dict mismatch, check if `_orig_mod.` prefix stripping is needed (see cifar/imagenet model.py)

### 7. Poor CL Performance / Catastrophic Forgetting
- `update_mode = "base"` has NO forgetting prevention -- switch to `jvp_reg`, `ewc_online`, or `kfac_online`
- JVP: increase `jvp_lambda` for stronger regularization (default 0.001, MNIST example uses 10)
- EWC: increase `ewc_lambda` for stronger weight consolidation (default 1000.0)
- KFAC: increase `kfac_lambda` (default 0.01)
- Check `max_iter` is sufficient for convergence (default 600)

### 8. WandB Issues
- Login: `wandb login`
- Disable entirely: set env var `WANDB_MODE=disabled`
- Check connectivity: `wandb status`

## Procedure

1. Read the config file if a path is provided.
2. If an error message is given, identify the category from the checklist above.
3. Read relevant source files to trace the error origin.
4. Run diagnostic commands to verify the environment state.
5. Suggest specific, actionable fixes with code or config references.
6. If multiple issues exist, prioritize by severity (environment > config > tuning).
