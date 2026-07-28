# CIFAR-10 Example

The same drift-and-adapt loop as the MNIST example, scaled up to real vision
backbones: a ViT-B/16 or a VGG-11 classifying CIFAR-10 while each stream window
is perturbed by a random affine transform.

This is the example to use when you want the pipeline exercised at a realistic
model size — attention blocks, 224×224 inputs, gradient accumulation. A GPU is
strongly recommended; the ViT config in particular is not practical on CPU.

## Contents

| File | Purpose |
|---|---|
| `model.py` | `VisionModelCifar` (backbone wrapper) and the `CIFAR_VISION` harness |
| `src/utils.py` | Dataset loaders, `load_model` backbone factory, `FixedAffine`, `TransformedView`, `sample_aug`, `make_loader` |
| `src/vision_transformers.py` | ViT constructors (`vit_b16`, `vit_l16`, `vit_l32`, `vit_h14`, `vit_g14`) |
| `src/cnns.py` | CNN constructors (VGG, ResNet, DenseNet, RegNet, AlexNet, Inception) |
| `cifar10_vit.toml` | ViT-B/16 — batch 256, 4 accumulation steps, 50 CL iterations |
| `cifar10_vgg11.toml` | VGG-11 — batch 32, no accumulation, 300 CL iterations |

## Prerequisite: A Checkpoint

**No pretrained checkpoint is committed for this example.** Both configs point at
files that are not in the repo:

```toml
pretrained_path = "examples/cifar/cifar10_vit.pth"    # cifar10_vit.toml
pretrained_path = "examples/cifar/cifar10_vgg11.pth"  # cifar10_vgg11.toml
```

When the file is missing, the harness prints

```
Warning: Pretrained model not found at examples/cifar/cifar10_vit.pth, using randomly initialized weights
```

and continues with the ImageNet-pretrained backbone plus a **freshly initialised
10-class head**. The run still completes, but starting accuracy is near chance
and the "drift" you observe is dominated by the untrained head rather than by the
affine perturbation. Fine-tune a CIFAR-10 checkpoint and save its `state_dict` to
the path above before drawing conclusions from a run. Keys prefixed with
`_orig_mod.` (from `torch.compile`) are stripped automatically on load.

## Running It

From the **repository root**:

```bash
# ViT-B/16 — 224x224 inputs, needs a GPU
poetry run python -m src.main --config examples/cifar/cifar10_vit.toml

# VGG-11 — 32x32 inputs, much lighter
poetry run python -m src.main --config examples/cifar/cifar10_vgg11.toml
```

CIFAR-10 downloads to `./data` on first run. Neither config sets
`[logging] backend`, so logging **defaults to Weights & Biases**; add
`--set logging.backend=\"none\"` to stay offline.

Input resolution is chosen by the model name: `load_model` resizes to 224×224
when `model.name` starts with `vit`, and keeps native 32×32 otherwise. That is
why the two configs use such different batch sizes.

### Swapping the Backbone

`model.name` accepts any key in `load_model` — `vit16b`, `vit16l`, `vit32l`,
`vit14h`, `vit14g`, `vgg11/16/19`, `resnet18/34/50/101`, `resnext50_32x4d`,
`resnext101_32x8d`, `densenet121/169/201`, `regnet_x_400mf/8gf/16gf`, `alexnet`,
`inception`. Anything else raises `NotImplementedError`.

```bash
poetry run python -m src.main --config examples/cifar/cifar10_vgg11.toml \
  --set model.name=\"resnet18\" \
  --set model.pretrained_path=\"examples/cifar/cifar10_resnet18.pth\"
```

CIFAR-100 also works — the harness reads `data.name` to size the head
(`cifar10` → 10 classes, `cifar100` → 100) — but you must register `cifar100` in
`examples/utils.py`, which currently dispatches only on `cifar10`.

## How the Drift Is Built

`update_data_stream()` draws a seeded transform per window via `sample_aug`:
rotation 0–18°, scale 1.0–1.1, matching shear, small translation.

Unlike MNIST, CIFAR's `FixedAffine` uses **only the most recent** entry in
`aug_history`, so each window is an independent perturbation rather than a
compounding one — drift jumps around instead of steadily worsening.
`get_hist_dataloaders()` matches that convention, replaying only the
immediately-preceding regime (`aug_history[:-1]`) rather than the full history.

Two other differences from MNIST worth knowing: the stream loader here is the
validation loader itself (there is no separate `[data] batch_size` stream batch),
and the loss is `CrossEntropyLoss` because the backbones emit raw logits.

## Expected Outcome

Structurally identical to MNIST — detector banner, per-batch evaluation, a
`==== DRIFT DETECTED (Event #N)! ====` block, a `CL Updates` progress bar,
before/after accuracy on the current and historical distributions, then resumed
monitoring — and it stops after `max_stream_updates = 20` windows.

Both configs use ADWIN on `metric_index = 0` (accuracy) with
`detection_interval = 10` and `reset_after_learning = false`.

What to expect concretely:

- **With a proper CIFAR-10 checkpoint**: clean accuracy in the 90s, a clear drop
  when a window's affine transform lands hard, ADWIN firing within a few
  detection intervals, and CL pulling `test_curr_acc` back up. Because each
  window is independent, `test_hist_acc` is a one-window-back check rather than
  a long-horizon forgetting measure.
- **Without a checkpoint**: accuracy starts near 10% and climbs as CL trains the
  new head. Detections still occur, but they reflect head initialisation, not
  the injected drift.

Absolute numbers depend entirely on the checkpoint, so no reference values are
quoted here.

Cost note: the ViT config is heavy — batch 256 × 4 accumulation steps at 224×224,
50 CL iterations per drift event, up to 20 windows. Trim
`drift_detection.max_stream_updates` and `train.max_iter` before a first run, and
drop `train.batch_size` if you hit OOM (raise `grad_accumulation_steps` to keep
the effective batch constant).

### Artifacts

- `output/cifar.csv` — long-format `step,metric,value`; both configs write to
  this same path, so the second run overwrites the first. Change
  `[visualization] input` if you want to keep both.
- W&B run (default backend) unless overridden.
- Checkpointing is not configured in either TOML; add `max_ckpts` and
  `ckpts_path` under `[model]` to enable it.
