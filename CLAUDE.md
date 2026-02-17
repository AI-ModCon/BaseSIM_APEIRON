# BaseSim Framework (SIM: Self Improving Model)

A PyTorch continuous learning framework for real-time concept drift detection and model adaptation.

## Quick Reference

### Running experiments
```bash
poetry run python -m src.main --config <path_to_toml>
```

### Visualizing results
```bash
poetry run python -m src.visualize --config <path_to_toml>
```

### Running tests
```bash
poetry run pytest
```

### Linting and type checks
```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy .
```

## Architecture

### Entry Points
- `src/main.py` -- Main experiment runner. Builds config, loads model harness, runs ContinuousMonitor.
- `src/visualize.py` -- Visualization entry point. Reads CSV metrics and generates dashboard PNGs.

### Core Pipeline
1. **Config** (`src/config/configuration.py`): TOML-based config parsed into frozen dataclasses (`Config`, `ModelCfg`, `DataCfg`, `TrainCfg`, `ContinualLearningCfg`, `DriftDetectionCfg`, `VisualizationCfg`). Supports `--set key=val` CLI overrides and `APP_` env var overrides.
2. **Model Harness** (`src/model/torch_model_harness.py`): Abstract `BaseModelHarness` providing `get_cur_data_loaders()`, `get_hist_data_loaders()`, `update_data_stream()`, `get_criterion()`, `get_optmizer()`, and `eval_metrics` dict.
3. **Driver** (`src/driver/continuous_monitor.py`): `ContinuousMonitor` orchestrates the monitoring loop -- evaluates batches, checks drift at intervals, dispatches CL training on drift.
4. **Drift Detection** (`src/drift_detection/`): `BaseDriftDetector` ABC with `update(value) -> DriftSignal`. Implementations: ADWINDetector, KSWINDetector, PageHinkleyDetector, ModelPerformanceDetector, ModelEvalDetector, EnsembleDetector.
5. **Training** (`src/training/continuous_trainer.py`): `ContinuousTrainer` runs outer/inner CL loops with gradient accumulation.
6. **Updaters** (`src/training/updater/`): `BaseUpdater` with hooks `cl_preprocessing()`, `fwd_bwd()`, `update_pre_fwd_bwd()`, `update_post_fwd_bwd()`, `update_post_optimizer_call()`, `cl_postprocessing()`. Implementations: base (vanilla), jvp_reg (JVP regularization), ewc_online (EWC), kfac_online (KFAC), none (no-op).
7. **Evaluation** (`src/evaluation/metrics.py`): `accuracy()` and `accuracy_topk()`.
8. **Logger** (`src/logger/`): Singleton `Logger` combining WandB metrics and console output. Stages: eval, drift, cl.
9. **Profilers** (`src/profilers/`): `FLOPSProfiler` using PyTorch FlopCounterMode.
10. **Visualization** (`src/visualization/metrics.py`): Dashboard generation from CSV metrics.

### Example Harnesses
- `examples/mnist/model.py`: `MNIST_CNN` -- CNN on MNIST with affine drift simulation.
- `examples/cifar/model.py`: `CIFAR_VISION` -- ViT/VGG on CIFAR-10 with affine drift.
- `examples/imagenet/model.py`: `IMAGENET_VISION` -- ViT on ImageNet with affine drift.
- `examples/utils.py`: `get_example(cfg)` factory dispatching on `cfg.data.name`.

### Configuration Format (TOML)
Required sections: `[model]` (name, pretrained_path), `[data]` (name, path), `[train]` (batch_size, num_workers, init_lr), `[drift_detection]` (detector_name, detection_interval, etc).
Optional sections: `[continual_learning]` (update_mode, lambda params), `[visualization]` (baseline, input, output).
Top-level keys: `seed`, `device` ("auto"|"cpu"|"cuda"|"mps"), `multi_gpu`.

### Available Drift Detectors
| Detector | Algorithm | Key Params |
|---|---|---|
| `ADWINDetector` | Adaptive windowing (river) | adwin_delta, adwin_minor_threshold, adwin_moderate_threshold |
| `KSWINDetector` | KS-test windowing (river) | kswin_alpha, kswin_window_size, kswin_stat_size |
| `PageHinkleyDetector` | Page-Hinkley test (river) | ph_min_instances, ph_delta, ph_threshold, ph_alpha |
| `ModelPerformanceDetector` | evidently batch analysis | (uses evidently defaults) |
| `ModelEvalDetector` | Direct eval comparison | metric_index |
| `EnsembleDetector` | Multi-detector voting | voting strategy |

### Available CL Update Modes
| Mode | Strategy | Key Params |
|---|---|---|
| `base` | Vanilla gradient descent | (none) |
| `jvp_reg` | JVP regularization | jvp_lambda, jvp_deltax_norm |
| `ewc_online` | Elastic Weight Consolidation | ewc_lambda, ewc_ema_decay |
| `kfac_online` | KFAC approximation | kfac_lambda, kfac_ema_decay |
| `none` | No-op (skip CL) | (none) |

### Coding Conventions
- Python 3.13+, type hints everywhere
- Formatting: ruff format, ruff check, mypy
- Frozen dataclasses for config
- ABC pattern for extension points (BaseModelHarness, BaseDriftDetector, BaseUpdater)
- Factory functions for dynamic loading (get_example, create_updater, load_drift_detector)
- Poetry for dependency management
