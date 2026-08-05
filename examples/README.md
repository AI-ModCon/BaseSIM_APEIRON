# Apeiron Examples

Each subdirectory is a self-contained **model harness** — a concrete
`BaseModelHarness` implementation plus one or more TOML configs — that shows the
framework running end to end: stream evaluation → drift detection → continual
learning → resumed monitoring.

All examples run through the same entry point:

```bash
poetry run python -m src.main --config <path_to_toml>
```

## Available Examples

| Directory | `data.name` | Model | Drift source | Data | Runs on a laptop? |
|---|---|---|---|---|---|
| [`mnist/`](mnist/README.md) | `mnist` | 3-layer CNN (`Cnn`, ~1M params) | Simulated: cumulative random affine per stream window | Auto-downloaded to `./data` | Yes — CPU is fine |
| [`cifar/`](cifar/README.md) | `cifar10` | ViT-B/16 or VGG-11 (`VisionModelCifar`) | Simulated: random affine per stream window | Auto-downloaded to `./data` | GPU strongly recommended |
| [`imagenet/`](imagenet/README.md) | `imagenet` | ViT-B/16 (`VisionModelImageNet`) | Simulated: cumulative random affine per stream window | **You provide** ILSVRC-2012 in `ImageFolder` layout | No — multi-GPU scale |
| [`matey/`](matey/README.md) | `matey`, `matey_stream` | MATEY ViT surrogate (`MATEYHarness`) | **Real**: SOLPS simulations arriving from new scenarios and machines | **You provide** a SOLPS root and a MATEY checkpoint | No — multi-GPU scale, and needs the MATEY package |

**Start with `mnist/`.** It is the only example that ships a pretrained
checkpoint, downloads its own data, and finishes in minutes on CPU.

## How an Example Is Wired Up

An example becomes runnable through three pieces:

1. **`examples/<name>/model.py`** — subclasses `BaseModelHarness` and implements
   `get_stream_dataloader()`, `get_train_dataloaders()`, `get_hist_dataloaders()`,
   `update_data_stream()`, `get_criterion()`, `get_optmizer()`, plus the
   `eval_metrics` / `higher_is_better` dicts.
2. **`examples/<name>/*.toml`** — the config. The `[data] name` field is the key
   the factory dispatches on.
3. **`examples/utils.py`** — `get_example(cfg)` maps `cfg.data.name` to the
   harness class. A `data.name` that is not listed there raises
   `NotImplementedError`.

To add your own, mirror the MNIST layout and add a branch to `get_example()`.
See [`docs/model_harness.md`](../docs/model_harness.md) for the full contract.

## How Drift Is Simulated

`matey/` is the exception to everything in this section: its stream is a real
sequence of simulations, so it needs no synthetic drift at all. See
[`matey/README.md`](matey/README.md).

None of the other datasets drift on their own, so each of those harnesses
manufactures drift the same way: `update_data_stream()` draws a seeded random affine transform
(rotation / scale / shear / translation) and rebuilds the train, validation, and
stream loaders through it. Every time the stream is exhausted, another transform
is drawn, so the input distribution keeps moving away from what the model was
trained on and monitored accuracy degrades.

MNIST and ImageNet **compose** the whole history of transforms (drift accumulates
and gets progressively worse); CIFAR applies only the most recent one (each
window is a fresh, independent perturbation). Historical replay loaders follow
the same convention — MNIST replays *every* prior regime, CIFAR replays only the
immediately preceding one.

The number of windows is capped by `[drift_detection] max_stream_updates`.

## What Every Run Produces

- **Console** — detector setup banner, a `Processing batches` progress bar,
  `==== DRIFT DETECTED (Event #N)! ====` blocks with regime / drift score /
  confidence, then a `CL Updates (drift_event_id=N)` bar and before/after test
  accuracy on both the current and historical distributions.
- **CSV** — long-format `step,metric,value` at the path in
  `[visualization] input` (e.g. `output/mnist.csv`). Metric names are stage
  prefixed: `eval/*` (per-batch accuracy and loss, plus `test_curr_acc` and
  `test_hist_acc` after each CL round), `drift/*` (score, regime, confidence,
  the monitored metric), `cl/*` (per-iteration losses and FLOP/time counters).
- **Experiment tracker** — Weights & Biases or MLflow when
  `[logging] backend` is set to `"wandb"` or `"mlflow"`; `"none"` keeps it to
  console + CSV. See [`docs/tracking.md`](../docs/tracking.md) for backend
  setup, the full metric namespace, and annotated screenshots of a run.

The package does not bundle a plotting step: `[visualization] input` is written
for you to chart externally.

## Common Overrides

Any config key can be overridden on the command line, so a single TOML covers a
lot of ground:

```bash
# Swap the CL strategy
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set continual_learning.update_mode=\"ewc_online\"

# Swap the detector
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set drift_detection.detector_name=\"KSWINDetector\"

# Shorten a run while iterating
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set train.max_iter=20 --set drift_detection.max_stream_updates=3

# Turn off experiment tracking
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set logging.backend=\"none\"
```

Detector names, update modes, and every config field are documented in
[`docs/`](../docs/README.md).

## Other Directories

Directories not listed in the table above are experiment scratch space and are
not registered in `get_example()`, so they cannot be run with `-m src.main` as
is.
