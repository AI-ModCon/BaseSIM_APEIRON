# BaseSim Framework (SIM: Self Improving Model)

A PyTorch continuous learning framework for real-time concept drift detection and model adaptation.

## Quick Reference

### Running experiments
```bash
poetry run python -m src.main --config <path_to_toml>
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

The installable package lives under `src/apeiron/` (imported as `apeiron`; see `pyproject.toml` `packages = [{ include = "apeiron", from = "src" }]`).

### Core Pipeline
1. **Config** (`src/apeiron/config/configuration.py`): TOML-based config parsed into frozen dataclasses (`Config`, `ModelCfg`, `DataCfg`, `TrainCfg`, `ContinualLearningCfg`, `DriftDetectionCfg`, `LoggingCfg`, `VisualizationCfg`). Supports `--set key=val` CLI overrides and `APP_` env var overrides.
2. **Model Harness** (`src/apeiron/model/torch_model_harness.py`): Abstract `BaseModelHarness` providing `get_stream_dataloader()`, `get_train_dataloaders()`, `get_hist_dataloaders()`, `update_data_stream()`, `get_criterion()`, `get_optmizer()`, and `eval_metrics` dict. Also keeps a per-task registry for transfer metrics -- `register_task()`, `eval_past_tasks()`, `task_diagonals` -- which subclasses inherit unchanged (see `docs/tracking.md` "Transfer Metrics"). Task records are now *re-evaluable references* (`apeiron.model.task_record`): a pointer into a committed window when one is available, else an in-memory copy that spills to disk when a checkpoint path is configured. When `[model] ckpts_path` is set, task records persist under `<ckpts_path>/task_records/` and `load_task_records()` restores them on resume.
   - **Data management** (`src/apeiron/data/`, see `docs/data_management.md`): `WindowStore` writes committed, immutable data *windows* to disk as memory-mapped `.npy` files with atomic-rename commits and self-describing manifests; `WindowHandle` hands out memmap-backed datasets/loaders with split + shard support. `WindowCatalog` is a sqlite index over manifests (query by time range / detected / seq; rebuildable from the store). `WindowedHarness` implements every `BaseModelHarness` data method in terms of a `WindowStore` -- the multi-node-ready on-ramp for real (non-synthetic) datasets. The shipped MNIST/CIFAR/ImageNet harnesses still generate drift on the fly and do *not* use the store.
   - **Checkpointing** (`src/apeiron/model/checkpoint.py`): `CheckpointStore` writes a metric sidecar per checkpoint and applies a *retention rule* (which files survive) plus a *promotion rule* (`deployed` pointer). Rule grammar: clauses joined by `+`, e.g. `fifo` (newest N, the default), `latest:2`, `max:test_hist_acc`, `min:loss`, or a union like `latest:1+max:test_hist_acc+max:test_curr_acc`.
3. **Driver** (`src/apeiron/driver/`): one `StreamEngine` (`stream_engine.py`) runs the monitoring loop -- per-batch evaluation, metric aggregation, FLOPs profiling, stream extension, the decision point, and the unified `drift/`-stage log schema -- parameterized by a `TriggerPolicy` (`trigger_policy.py`: when/whether to fire) and a `TriggerAction` (`trigger_action.py`: what to do on a fire). The three run types are just wirings of that engine: `ContinuousMonitor` = `DetectorPolicy` (interval cadence, fire on detected drift) + `AdaptAction` (run CL); `src/drift_only.py` = `DetectorPolicy` + `RecordOnlyAction` (frozen model); `src/cl_only.py` = `SchedulePolicy` (window cadence, `schedules.py`) + `AdaptAction` (budget-matched control arm). The logger is injected (constructed once in the entry point and threaded through the engine and trainer), not fetched from a global. All three modes now emit one CSV schema (`detected`, `score`, `regime`, `confidence`, `metric_<idx>`, `decision_idx`, `trigger_count`, `stream_idx`, timestamps, `cperf_*`); the former `drift_only` `check_idx` is now `decision_idx`.
4. **Drift Detection** (`src/apeiron/drift_detection/`): `BaseDriftDetector` ABC with `update(value) -> DriftSignal`. Implementations: ADWINDetector, KSWINDetector, PageHinkleyDetector, ModelPerformanceDetector, ModelEvalDetector, EnsembleDetector.
5. **Training** (`src/apeiron/training/continuous_trainer.py`): `ContinuousTrainer` runs outer/inner CL loops with gradient accumulation.
6. **Updaters** (`src/apeiron/training/updater/`): `BaseUpdater` with hooks `cl_preprocessing()`, `fwd_bwd()`, `update_pre_fwd_bwd()`, `update_post_fwd_bwd()`, `update_post_optimizer_call()`, `cl_postprocessing()`. Implementations: base (vanilla), jvp_reg (JVP updater -- now a first-order SAM robust update), ewc_online (EWC), kfac_online (KFAC), none (no-op).
7. **Evaluation** (`src/apeiron/evaluation/metrics.py`): `accuracy()` and `accuracy_topk()`.
8. **Logger** (`src/apeiron/logger/`): `Logger` with pluggable metrics backends -- `WandBLogger` and `MLFlowLogger` (configured via `[logging] backend = "wandb"|"mlflow"|"none"`), plus console output. Stages: eval, drift, cl.
9. **Profilers** (`src/apeiron/profilers/`): `FLOPSProfiler` (`count_flops.py`) using PyTorch FlopCounterMode.
10. **Distributed** (`src/apeiron/distributed/`, see `docs/distributed.md`): `comm` is a thin facade over `torch.distributed` that is a **no-op single-process** (every collective is identity when `world_size == 1`), so the whole framework calls it unconditionally while single-process behavior is byte-identical. Launch detection reads torchrun (`RANK`/`LOCAL_RANK`/`WORLD_SIZE`) or SLURM srun (`SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS`); a plain `sbatch` (no `srun`) stays single-process. Multi-node is **data-parallel** and applies to `WindowedHarness`: each rank runs inference/training on a contiguous window shard; the engine all-gathers per-batch metrics per window, rank 0 owns the detector decision (one decision per window — invariant to node count) and broadcasts the verdict; adaptation broadcasts rank-0 weights then averages gradients across ranks before `optimizer.step()`. Only rank 0 writes logs/CSV/checkpoints/task-records. Device binding is `cuda:LOCAL_RANK` under a distributed launch. Tested on CPU/`gloo` via `tests/dist_smoke.py` (+ `slow` wrapper `tests/test_distributed_smoke.py`).

Note: `[visualization]` config (`VisualizationCfg`) is parsed but there is no bundled dashboard/renderer in the current package; runs emit a CSV at `visualization.input` for external plotting.

### Example Harnesses
- `examples/mnist/model.py`: `MNIST_CNN` -- CNN on MNIST with affine drift simulation.
- `examples/cifar/model.py`: `CIFAR_VISION` -- ViT/VGG on CIFAR-10 with affine drift.
- `examples/imagenet/model.py`: `IMAGENET_VISION` -- ViT on ImageNet with affine drift.
- `examples/well/`: **The Well** streaming neural PDE surrogate under regime drift (`data.name = "well"`, see `examples/well/README.md`) -- the regression example and the scaling benchmark. `convert.py` turns [The Well](https://polymathic-ai.github.io/the_well) HDF5 (or a schema-identical `fixture.py`, no download) into a committed, `tcool`-ordered `WindowStore` of next-step-prediction windows; `model.py` is a residual CNN surrogate + `WellHarness` (a `WindowedHarness`, so Phase-3 sharding applies); metrics are VRMSE/MAE (lower-is-better). `benchmark.py` measures the branch: `memory` (pointer vs copy task eval sets), `throughput` (strong/weak scaling + `comm.timings()` Amdahl breakdown, launch under torchrun/srun), `frontier` (accuracy-cost, preserved across world size), `resume` (kill/restart continuity). Needs `h5py` (a dependency).
- `examples/utils.py`: `get_example(cfg)` factory dispatching on `cfg.data.name`.
- For a harness backed by committed windows on disk (rather than on-the-fly drift), subclass `apeiron.data.WindowedHarness` (or pass `criterion=`/`optimizer_factory=`/`eval_metrics=` to it directly) and point `[data] window_store_path` at a `WindowStore` root.

### Configuration Format (TOML)
Required sections: `[model]` (name, pretrained_path), `[data]` (name, path), `[train]` (batch_size, num_workers, init_lr), `[drift_detection]` (detector_name, detection_interval, etc).
`[model]` also accepts checkpoint controls: `max_ckpts` (0 disables), `ckpts_path`, `ckpts_retention` (default `"fifo"`), `deploy_rule` (default empty). `[data]` also accepts `window_store_path` (root of a committed-window store, used by `WindowedHarness`).
Optional sections: `[continual_learning]` (update_mode, lambda params), `[logging]` (backend = "wandb"|"mlflow"|"none", experiment_name, mlflow_tracking_uri), `[visualization]` (baseline, input, output -- parsed but not rendered by the package).
Top-level keys: `seed`, `device` ("auto"|"cpu"|"cuda"|"mps"), `multi_gpu`.

### Available Drift Detectors
The `detector_name` config value must be one of the strings the loader accepts
(`src/apeiron/drift_detection/load_drift_detector.py`):

| `detector_name` | Algorithm | Key Params |
|---|---|---|
| `ADWINDetector` | Adaptive windowing (river) | adwin_delta, adwin_minor_threshold, adwin_moderate_threshold |
| `KSWINDetector` | KS-test windowing (river) | kswin_alpha, kswin_window_size, kswin_stat_size |
| `PageHinkleyDetector` | Page-Hinkley test (river) | ph_min_instances, ph_delta, ph_threshold, ph_alpha |
| `ModelPerformanceDetector` | evidently batch analysis | (uses evidently defaults) |
| `EvalDetector` | Direct eval comparison (`ModelEvalDetector`) | metric_index |
| `EnsembleDetector` | Voting over sub-detectors | ensemble_detectors, ensemble_voting |

`EnsembleDetector` builds each name in `ensemble_detectors` from the same `[drift_detection]` block (so a detector type can appear at most once) and combines their verdicts per `ensemble_voting`: `majority`, `any` (alias `or`), or `unanimous` (aliases `all`, `and`). An unknown voting name or an empty detector list raises `ValueError`.

### Available CL Update Modes
| Mode | Strategy | Key Params |
|---|---|---|
| `base` | Vanilla gradient descent | (none) |
| `jvp_reg` | JVP updater -- implements a first-order SAM / Bertsimas robust update | jvp_rho_theta, jvp_rho_x, jvp_data_sign |
| `ewc_online` | Elastic Weight Consolidation | ewc_lambda, ewc_ema_decay |
| `kfac_online` | KFAC approximation | kfac_lambda, kfac_ema_decay |
| `none` | No-op (skip CL) | (none) |

Replay of historical data is handled in `BaseUpdater.fwd_bwd()` and gated by
`[continual_learning] mix_historic_data` (default `false`). When enabled and the
harness supplies historical dataloaders, half of each stream is combined into a
single forward/backward pass, so the sample count per step stays at
`train.batch_size` whether or not mixing is on (an undersized historical batch is
topped up from the current one). `base`, `ewc_online`, and `kfac_online` inherit this. `jvp_reg` ignores the flag: it
overrides `fwd_bwd()` and mixes the two streams itself (it needs a current-only
gradient as the SAM parameter-perturbation direction before evaluating the combined
loss). `none` skips training entirely.

### Coding Conventions
- Python 3.13+, type hints everywhere
- Formatting: ruff format, ruff check, mypy
- Frozen dataclasses for config
- ABC pattern for extension points (BaseModelHarness, BaseDriftDetector, BaseUpdater)
- Factory functions for dynamic loading (get_example, create_updater, load_drift_detector)
- Poetry for dependency management
