# ImageNet Example

The largest bundled harness: a ViT-B/16 classifying ILSVRC-2012 (1000 classes)
while the stream is progressively distorted by accumulating affine transforms.

This example exists to show the pipeline at cluster scale. Unlike the other
examples it cannot download its own data: **you supply ILSVRC-2012 and point the
config at it**. Read this whole file before running — the compute bill is real.

## Contents

| File | Purpose |
|---|---|
| `model.py` | `VisionModelImageNet` (backbone wrapper) and the `IMAGENET_VISION` harness |
| `src/utils.py` | `ImageFolder` loaders, `load_model` backbone factory, `FixedAffine`, `TransformedView`, `sample_aug`, `make_loader` |
| `imagenet_vit.toml` | ViT-B/16 config — ADWIN + `base` updater, 5 stream windows, logging off |

Backbone constructors are reused from the CIFAR example
(`examples/cifar/src/{cnns,vision_transformers}.py`), so `model.name` accepts the
same keys: `vit16b`, `vit16l`, `vit32l`, `vit14h`, `vit14g`, `vgg11/16/19`,
`resnet18/34/50/101`, `resnext50_32x4d`, `resnext101_32x8d`,
`densenet121/169/201`, `regnet_x_400mf/8gf/16gf`, `alexnet`, `inception`.

## Prerequisite 1: The Dataset

ImageNet is not downloadable from the code — get it from
[image-net.org](https://www.image-net.org/) and lay it out for
`torchvision.datasets.ImageFolder`:

```
<data.path>/
  train/
    n01440764/*.JPEG
    n01443537/*.JPEG
    ...
  val/
    n01440764/*.JPEG
    ...
```

`data.path` points at the parent; `train/` and `val/` are appended by
`get_imagenet_train()` and `get_imagenet_val()`. Note that the stock ILSVRC
validation archive is a flat directory of 50k files — it must be reorganised
into per-synset subfolders first.

Preprocessing is standard ImageNet eval: resize shorter side to 256 (bicubic),
center-crop 224, `ToTensor`, normalise with mean `(0.485, 0.456, 0.406)` and std
`(0.229, 0.224, 0.225)`.

## Prerequisite 2: Point the Config at Your Data

`imagenet_vit.toml` ships with everything filled in except the one thing only you
know — where the dataset lives. Edit it:

```toml
[data]
name = "imagenet"           # the factory key; leave this alone
path = "/path/to/imagenet"  # <- change to the parent of train/ and val/
```

or override it per run without touching the file:

```bash
--set data.path=\"/scratch/datasets/imagenet\"
```

The shipped defaults are deliberately conservative: `max_stream_updates = 5`,
`max_iter = 50`, `logging.backend = "none"`, and checkpointing off. Raise them
once you have measured what one window costs on your hardware.

On checkpoints: unlike CIFAR, a missing `pretrained_path` here is benign —
`load_model("vit16b", num_classes=1000)` already returns ImageNet-pretrained
weights with a correctly sized 1000-class head, so the run starts from a genuinely
accurate model. Point `pretrained_path` at a `state_dict` only if you have your
own fine-tune; `_orig_mod.` prefixes from `torch.compile` are stripped on load.

## Running It

From the **repository root**:

```bash
poetry run python -m src.main --config examples/imagenet/imagenet_vit.toml
```

Sanity-check the wiring on a small subset before committing real compute — point
`data.path` at a directory holding a handful of synsets in the same
`train/`+`val/` layout:

```bash
poetry run python -m src.main --config examples/imagenet/imagenet_vit.toml \
  --set data.path=\"/path/to/imagenet-subset\" \
  --set drift_detection.max_stream_updates=1 \
  --set train.max_iter=5
```

Note that a subset still loads a 1000-class head, so accuracy will look poor —
this checks that data, model, detector, and trainer are wired together, nothing
more.

## How the Drift Is Built

`update_data_stream()` draws a seeded transform per window via `sample_aug`:
rotation 0–10°, scale 0.75–1.25, matching shear, small translation.

Like MNIST (and unlike CIFAR), `FixedAffine` composes **every** transform in
`aug_history`, so distortion accumulates and each window is strictly harder than
the last. Historical replay, however, follows the CIFAR convention: only the
immediately-preceding regime (`aug_history[:-1]`) is replayed, not the full
history — so `test_hist_acc` is a one-window-back forgetting check.

The stream loader is the validation loader itself, and the loss is
`CrossEntropyLoss` (the ViT emits raw logits).

## Expected Outcome

The loop is identical in shape to the other examples — detector banner,
per-batch evaluation, `==== DRIFT DETECTED (Event #N)! ====` with regime, drift
score and confidence, a `CL Updates (drift_event_id=N)` progress bar, before/after
accuracy on current and historical data, resumed monitoring — terminating after
`max_stream_updates` windows.

Starting from the stock pretrained ViT-B/16, expect clean top-1 in the high 70s
to low 80s, degrading window by window as affines compose, with ADWIN firing once
the aggregated accuracy drop is established and CL recovering part of the loss.
Absolute recovery depends on `train.max_iter`, learning rate, and update mode, so
no reference numbers are quoted.

### Artifacts

- The CSV at `[visualization] input` — long-format `step,metric,value` with
  `eval/*`, `drift/*`, and `cl/*` rows.
- A W&B or MLflow run if `[logging] backend` is set to one of those.

### Cost

This is the expensive example. Every stream window rebuilds loaders over the
**full** 1.28M-image train split and 50k val split, and each drift event trains
on it. One pass over the validation split alone is 50k forward passes at 224×224.
Budget accordingly: set `multi_gpu = true`, keep `max_stream_updates` and
`train.max_iter` low, raise `num_workers`, and expect data loading — not the
model — to be the bottleneck on spinning disks.
