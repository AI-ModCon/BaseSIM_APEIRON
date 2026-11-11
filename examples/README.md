# Apeiron Examples

Each subdirectory is a self-contained **model harness** — a concrete
`BaseModelHarness` implementation plus one or more TOML configs — that shows the
framework running end to end: stream evaluation → drift detection → continual
learning → resumed monitoring.

All examples run through the same entry point:

```bash
uv run python -m src.main --config <path_to_toml>
```

## Available Examples

| Directory | `data.name` | Model | Drift source | Data | Runs on a laptop? |
|---|---|---|---|---|---|
| [`mnist/`](mnist/README.md) | `mnist` | 3-layer CNN (`Cnn`, ~1M params) | Simulated: cumulative random affine per stream window | Auto-downloaded to `./data` | Yes — CPU is fine |
| [`cifar/`](cifar/README.md) | `cifar10` | ViT-B/16 or VGG-11 (`VisionModelCifar`) | Simulated: random affine per stream window | Auto-downloaded to `./data` | GPU strongly recommended |
| [`imagenet/`](imagenet/README.md) | `imagenet` | ViT-B/16 (`VisionModelImageNet`) | Simulated: cumulative random affine per stream window | **You provide** ILSVRC-2012 in `ImageFolder` layout | No — multi-GPU scale |
| [`slac_fel/`](slac_fel/README.md) | `slac-fel` | Pretrained fully connected surrogate | Measured: chronological accelerator operating regimes | **You provide** the focused 14-file FEL subset | Yes — CPU is fine |

**Start with `mnist/`** for the simplest self-contained workflow. It
downloads its own data and finishes in minutes on CPU. Use `slac_fel/`
for a measured scientific stream after obtaining the focused dataset.

## How an Example Is Wired Up

An example becomes runnable through three pieces:

1. **`examples/<name>/model.py`** — subclasses `BaseModelHarness` and implements
   `get_stream_dataloader()`, `get_train_dataloaders()`, `get_hist_dataloaders()`,
   `update_data_stream()`, `get_criterion()`, `get_optimizer()`, plus the
   `eval_metrics` / `higher_is_better` dicts.
2. **`examples/<name>/*.toml`** — the config. The `[data] name` field is the key
   the factory dispatches on.
3. **`examples/utils.py`** — `get_example(cfg)` maps `cfg.data.name` to the
   harness class. A `data.name` that is not listed there raises
   `NotImplementedError`.

To add your own, mirror the MNIST layout and add a branch to `get_example()`.
See [`docs/model_harness.md`](../docs/model_harness.md) for the full contract.

## Where Drift Comes From

The computer-vision harnesses manufacture drift with seeded random affine
transformations such as rotation, scale, shear, and translation. MNIST and
ImageNet accumulate those transformations over time, while CIFAR applies a
fresh perturbation to each stream window.

The SLAC-FEL example instead consumes chronological accelerator observations.
Its drift signals arise from changes in the measured surrogate error, not from
synthetic input transformations.

## What Every Run Produces

- **Console** — detector setup banner, a `Processing batches` progress bar,
  `==== DRIFT DETECTED (Event #N)! ====` blocks with regime / drift score /
  confidence, then a `CL Updates (drift_event_id=N)` bar and before/after test
  accuracy on both the current and historical distributions.
- **CSV** — long-format `step,metric,value` at the path in
  `[logging] metrics_output_path` (e.g. `output/mnist.csv`). Metric names are stage
  prefixed: `eval/*` (per-batch accuracy and loss, plus `test_curr_acc` and
  `test_hist_acc` after each CL round), `drift/*` (score, regime, confidence,
  the monitored metric), `cl/*` (per-iteration losses and FLOP/time counters).
- **Experiment tracker** — Weights & Biases or MLflow when
  `[logging] backend` is set to `"wandb"` or `"mlflow"`; `"none"` keeps it to
  console + CSV. See [`docs/tracking.md`](../docs/tracking.md) for backend
  setup, the full metric namespace, and annotated screenshots of a run.

The package does not bundle a common plotting step. Long-format CSV metrics are
written to `[logging] metrics_output_path` for external analysis and plotting.

## Common Overrides

Any config key can be overridden on the command line, so a single TOML covers a
lot of ground:

```bash
# Swap the CL strategy
uv run python -m src.main --config examples/mnist/mnist.toml \
  --set continual_learning.update_mode=\"ewc_online\"

# Swap the detector
uv run python -m src.main --config examples/mnist/mnist.toml \
  --set drift_detection.detector_name=\"KSWINDetector\"

# Shorten a run while iterating
uv run python -m src.main --config examples/mnist/mnist.toml \
  --set train.max_iter=20 --set drift_detection.max_stream_updates=3

# Turn off experiment tracking
uv run python -m src.main --config examples/mnist/mnist.toml \
  --set logging.backend=\"none\"
```

Detector names, update modes, and every config field are documented in
[`docs/`](../docs/README.md).

## Other Directories

Directories not listed in the table above are experiment scratch space and are
not registered in `get_example()`, so they cannot be run with `-m src.main` as
is.
