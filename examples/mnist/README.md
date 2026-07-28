# MNIST Example

The reference example, and the best place to start. A small CNN classifies MNIST
digits while the stream is progressively distorted by affine transforms; ADWIN
watches accuracy, and a continual-learning update fires when it detects the
degradation.

Everything needed is in the repo — the dataset downloads itself and a pretrained
checkpoint (`mnist.pth`) is committed — so a full run finishes in minutes on CPU.

## Contents

| File | Purpose |
|---|---|
| `model.py` | `Cnn` (3 conv + 2 FC, log-softmax output) and the `MNIST_CNN` harness |
| `utils.py` | Dataset loaders, `FixedAffine`, `TransformedView`, `sample_aug`, `make_loader` |
| `mnist.toml` | Main config — ADWIN + `base` updater, W&B logging |
| `mnist-mlflow.toml` | Same experiment, logging to MLflow instead |
| `mnist-generic.toml` | Monitoring only (`detection_interval = 0`) — no drift checks, no CL |
| `mnist.pth` | Pretrained weights — see the note below |
| `sweep/` | Six detector × updater configs and their recorded metrics CSVs |

> **About `mnist.pth`.** It is deliberately *under*-trained: measured with the
> harness's own preprocessing it scores **~50%** on the clean MNIST test split,
> not the ~99% a converged CNN reaches. That leaves clear headroom for the
> continual-learning loop to demonstrate recovery — post-CL accuracy climbing
> into the mid-90s is the point of the example. If you want a run that starts
> from a strong model and shows drift *degrading* it, train your own checkpoint
> and point `[model] pretrained_path` at it.

## Running It

From the **repository root** (paths in the TOML are root-relative):

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml
```

MNIST downloads to `./data` on first run. The default config logs to Weights &
Biases; to skip that:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set logging.backend=\"none\"
```

For a quick smoke test (~3 stream windows, 20 CL steps per drift event):

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set logging.backend=\"none\" \
  --set train.max_iter=20 \
  --set drift_detection.max_stream_updates=3
```

### Variants

```bash
# MLflow instead of W&B  (then `mlflow ui` → http://localhost:5000)
# See ../../docs/tracking.md for the metric namespace and how to read the charts
poetry run python -m src.main --config examples/mnist/mnist-mlflow.toml

# Monitoring only, no drift detection and no CL
poetry run python -m src.main --config examples/mnist/mnist-generic.toml

# EWC instead of vanilla SGD updates
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set continual_learning.update_mode=\"ewc_online\"
```

## How the Drift Is Built

`update_data_stream()` calls `sample_aug(seed = cfg.seed + task_counter)` to draw
a rotation (0–10°), scale (0.75–1.25), matching shear, and small translation,
appends it to `aug_history`, and rebuilds all loaders through
`FixedAffine(aug_history)` — which applies **every** transform in the history in
sequence. Distortion therefore *accumulates*: window 5 has five stacked affines
on it, and each new window is strictly harder than the last. The transform for a
given window is fully determined by `seed`, so runs are reproducible.

`get_hist_dataloaders()` returns a `ConcatDataset` over one view per prior
cumulative regime (`aug_history[:1]`, `aug_history[:2]`, …). Replay spans the
whole history rather than just the previous window, which keeps the model
anchored to the early, near-clean regimes. That is what `test_hist_acc` in the
logs measures.

Note the two batch sizes: `[data] batch_size = 32` is the *stream* batch (one
monitored evaluation point), while `[train] batch_size = 64` is used for CL
training and replay.

## Expected Outcome

Console output follows this shape:

```
==== ContinuousMonitor initialized ====
	Detector: ADWINDetector
	Monitoring metric index: 0
	Detection interval: 10 batches
	...
==== Starting Continuous Monitoring ====
Mutating the picture further using an angle of 3.79... and a scale of 0.86...
Processing batches: ...
==== DRIFT DETECTED (Event #1)! ====
	Regime: ...
	Drift Score: 0.4213
	Confidence: ...
-> Dispatching continual learning module...
==== Continual Learning ====
	Initial test acc: ...
	Initial historical test acc: ...
CL Updates (drift_event_id=1): 100%|...| 600/600
	Test Accuracy: 94.8%
	Hist Test Accuracy: 95.8%
<- Continual learning complete.
==== RESUMING MONITORING! ====
```

What to look for:

- **Monitored accuracy starts low.** Per-batch `eval/accuracy` opens in the
  40–60% range in the recorded sweep runs — partly because the shipped
  checkpoint is only ~50% accurate to begin with, partly because the very first
  window is already affine-transformed. It degrades further as transforms
  accumulate.
- **ADWIN fires once the drop is statistically established**, not on the first
  bad batch — it aggregates the mean accuracy over `detection_interval = 10`
  batches before each check.
- **CL recovers most of the loss.** In the recorded `sweep/metrics` runs (a
  shortened configuration: 5 windows, 20 CL iterations), post-CL
  `test_curr_acc` lands at **94–96%** with `test_hist_acc` at **95–96%** — the
  model adapts to the new regime without forgetting the old one. The default
  `mnist.toml` trains 600 iterations per event, so it should do at least this
  well.
- **Run length**: the loop stops after `max_stream_updates = 20` windows.

`drift/detected = 0` rows are sampled at 10% in the CSV to hold volume down;
`drift/detected = 1` rows are never dropped, so counting drift events from the
CSV is exact.

### Artifacts

- `output/mnist.csv` — long-format `step,metric,value`. Filter on `eval/accuracy`
  for the monitoring curve, `drift/score` for the detector signal, and
  `eval/test_curr_acc` / `eval/test_hist_acc` for post-CL recovery.
- W&B or MLflow run, per `[logging] backend`.
- Checkpoints are **off** by default (`max_ckpts = 0`); raise it to write
  per-drift-event checkpoints into `output/mnist/`.

## The Sweep Directory

`sweep/configs/` holds six pre-built combinations — `ADWINDetector`,
`KSWINDetector`, `PageHinkleyDetector` × `base`, `ewc_online` — all with
`logging.backend = "none"` and a short budget (5 windows, 20 CL iterations) so
they run fast. `sweep/metrics/` holds the CSVs already recorded from them, so
you can compare detector behaviour without re-running anything.

```bash
poetry run python -m src.main --config examples/mnist/sweep/configs/KSWINDetector__ewc_online.toml
```

Each config writes to its own CSV under `sweep/metrics/`, so running one
overwrites that recorded baseline. A useful comparison from the committed data:
ADWIN and KSWIN each fire once over the 5 windows and end near 95% on both
current and historical data, while Page-Hinkley is more trigger-happy — it fires
twice, and the first, earlier adaptation costs it historical accuracy
(`test_hist_acc` 84.5% after event 1, recovering to 92.2% after event 2).
