# Quickstart

This page takes you from a fresh checkout to a running drift-detection
experiment. If you have not installed Apeiron yet, start with {doc}`installation`.

## 1. Run a bundled example

Every experiment is driven by a TOML config file passed to `src/main.py`:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml
poetry run python -m src.main --config examples/cifar/cifar10_vit.toml
poetry run python -m src.main --config examples/imagenet/imagenet_vit.toml  # needs ImageNet at data.path
```

MNIST is the fastest way to see the whole loop: the harness applies a random
affine transform to each new task, which drives the monitored accuracy down and
triggers the detector.

## 2. Read the output

Per-batch metrics are written to the CSV at `visualization.input`
(default `output/output.csv`) in long form:

```text
step,metric,value
10,eval/accuracy,62.5
10,eval/loss,2.0406203269958496
10,drift/score,0.0
10,drift/regime,stable
10,cl/jvp_reg_total_loss,3.4211268424987793
```

Metric names are namespaced by stage — `eval/`, `drift/`, and `cl/`. The full
list of emitted metrics is in {ref}`the visualization section <visualization>`
of the configuration reference.

```{note}
`[visualization]` is parsed and the CSV is written, but the package does not
bundle a dashboard or renderer — plot the CSV with your tool of choice.
```

## 3. Turn on metrics logging

Apeiron ships two metrics backends, Weights & Biases and MLflow, selected with
`[logging] backend`. Setting it to `none` disables remote logging (console
output is unaffected).

```bash
poetry run python -m src.main \
  --config examples/mnist/mnist.toml \
  --set logging.backend=mlflow \
  --set logging.experiment_name="My Experiment"
```

For MLflow, run `mlflow ui` in another terminal and open
<http://localhost:5000>. The MNIST example sets `backend = "wandb"` in its TOML;
the other examples leave it unset, which also defaults to W&B.

## 4. Override config without editing files

Values resolve in this order, later winning over earlier:

1. Base TOML from `--config`
2. Environment variables prefixed with `APP_`
3. Repeated `--set key=value` CLI flags

```bash
poetry run python -m src.main \
  --config examples/mnist/mnist.toml \
  --set drift_detection.detector_name=\"KSWINDetector\" \
  --set train.max_iter=200
```

```{tip}
String values passed to `--set` need TOML quoting, hence the escaped quotes
above. Numbers and booleans do not.
```

## 5. A minimal config of your own

```toml
seed = 1337
device = "auto"

[model]
name = "mnist"
pretrained_path = "examples/mnist/mnist.pth"

[data]
name = "mnist"
path = ""
batch_size = 32

[train]
batch_size = 64
num_workers = 4
init_lr = 0.001

[continual_learning]
update_mode = "base"

[drift_detection]
detector_name = "ADWINDetector"

[logging]
backend = "none"

[visualization]
input = "output/results.csv"
```

See {doc}`configurations` for every key, and {doc}`choosing_a_detector` for
picking and tuning the detector.

## Where to go next

- Bringing your own model and dataset → {doc}`model_harness`
- Choosing a drift detector and its thresholds → {doc}`choosing_a_detector`
- Changing what happens on drift → {doc}`continuous_learning`
- Measuring FLOPs and wall time → {doc}`profiler`
