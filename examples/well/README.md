# The Well — continual learning under real physical regime change

The MNIST, CIFAR and ImageNet examples simulate drift. Each draws a random
affine transform per stream window and pushes the inputs away from what the
model was trained on. That exercises the machinery, but the drift is synthetic
and its size is a knob.

This example does not simulate anything.
[The Well](https://polymathic-ai.org/the_well/) is a collection of physics
simulation datasets from PolymathicAI. Each is published as a parameter sweep:
one file per value of some physical parameter. Putting one of those files in
each stream window moves the model through real physical change, in physical
order, with the inputs untouched.

| | |
|---|---|
| `data.name` | `well:turbulent_radiative_layer_2D` |
| Model | The Well's `UNetClassic` baseline at `init_features=16` (1.94M params) |
| Drift source | Real. Nine regimes, cooling time `tcool` from 0.03 to 3.16 |
| Detector | Page-Hinkley on per-batch VRMSE |
| CL | `ewc_online` with replay of every earlier regime |
| Data | Downloaded a regime at a time into `./data/well`, ~0.68 GB each |
| Runs on a laptop? | Yes. 4 minutes once the data is local, ~6 GB of disk for all nine |

## Running it

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml
```

Nothing to download by hand. Each window fetches its regime, converts it once
into a memory-mappable array, and drops the HDF5. A shorter run costs
proportionally less:

```bash
# three regimes instead of nine
poetry run python -m src.main --config examples/well/well_trl2d.toml \
    --set drift_detection.max_stream_updates=2
```

If you already have The Well on disk, point at it. Nothing is downloaded or
deleted:

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml \
    --set data.path=/scratch/the_well/datasets
```

## What a run looks like

```
Well dataset turbulent_radiative_layer_2D: 9 regimes,
  turbulent_radiative_layer_tcool_0.03 .. turbulent_radiative_layer_tcool_3.16
Window 0: regime turbulent_radiative_layer_tcool_0.03
Processing batches: 194it ...
Window 1: regime turbulent_radiative_layer_tcool_0.06
...
Window 4: regime turbulent_radiative_layer_tcool_0.32
==== DRIFT DETECTED (Event #1)! ====
-> Dispatching continual learning module...
CL Updates (drift_event_id=1): 100%|██████████| 200/200
<- Continual learning complete.
==== RESUMING MONITORING! ====
```

Nine windows of 194 batches. The detector fires three times, in windows 4, 5
and 7. Each firing runs 200 gradient steps on the current regime mixed with
replay of the earlier ones, then writes a checkpoint. The whole run takes
3 min 55 s with the regimes already on disk, on an M-series laptop where
`device = "auto"` selects MPS.

### Splits

The stream and the continual-learning rounds both read `train`. Evaluation
during the run reads `valid`. The run never touches `test`, which leaves it for
a separate pass afterwards:

```bash
poetry run python -m examples.well.evaluate \
    --config examples/well/well_trl2d.toml --baselines --random
```

Streaming and training from the same split is prequential, or test-then-train:
the detector scores a batch before any round of learning touches it, so it
never reads data the model has already fitted.

## What the adaptation is worth

`evaluate.py` scores every model a run produced against all nine regimes of the
test split.

Rows are the models scored: three reference points, then the run's three
checkpoints in the order it wrote them. Columns are the nine test regimes,
labelled by cooling time `tcool`. Cells are mean VRMSE on that regime, the last
column averages the nine, and lower is better.

| model scored | 0.03 | 0.06 | 0.10 | 0.18 | 0.32 | 0.56 | 1.00 | 1.78 | 3.16 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| persistence (repeat the last input frame) | 0.620 | 0.644 | 0.593 | 0.608 | 0.576 | 0.555 | 0.530 | 0.523 | 0.536 | **0.576** |
| zero (predict a zero field) | 1.200 | 1.180 | 1.301 | 1.205 | 1.588 | 1.826 | 2.640 | 3.506 | 5.861 | **2.256** |
| untrained (what the run started from) | 1.186 | 1.163 | 1.354 | 1.293 | 1.865 | 2.296 | 3.288 | 4.383 | 7.025 | **2.650** |
| after drift event 1 (fired in window 4) | 0.800 | 0.783 | 0.884 | 0.834 | 1.246 | 1.569 | 2.305 | 3.159 | 5.096 | **1.853** |
| after drift event 2 (window 5) | 0.713 | 0.699 | 0.696 | 0.702 | 0.849 | 0.905 | 1.136 | 1.487 | 2.313 | **1.055** |
| after drift event 3 (window 7) | 0.692 | 0.679 | 0.655 | 0.661 | 0.737 | 0.740 | 0.844 | 1.076 | 1.617 | **0.856** |

Every column decreases at every drift event, including for regimes the stream
had not reached and regimes it had left. No forgetting shows up at all. That
follows from starting untrained: the model is far enough from good that
anything it learns anywhere helps everywhere. Forgetting becomes measurable
when a run starts from a trained checkpoint, and this matrix is the right shape
to read it off.

The model does not beat persistence anywhere: 0.856 against 0.576, after
improving 3.1x from where it started. Persistence is the floor for next-step
prediction on a smooth field and is hard to beat at short horizons, which is
why The Well reports it beside their own baselines. A 1.94M-parameter U-Net
with 600 gradient steps in total demonstrates the monitoring loop; it is not an
attempt at the benchmark.

The high-`tcool` regimes are not harder in themselves. Persistence gets
slightly better as `tcool` rises, so those fields are smoother in time. What
climbs is the error of a model that has not seen them: 1.19 to 7.02 across the
untrained row. That climb is the signal the detector reads.

### Does the detector earn its place?

`cl_only.py` runs the same continual learning on a schedule you pick, with no
detector. Same model, same seed, same three rounds, placed differently:

| when learning ran | windows | mean test VRMSE |
|---|---|---|
| detector | 4, 5, 7 | 0.856 |
| replay of the detector's own windows | 4, 5, 7 | 0.856 |
| every third window | 2, 5, 8 | 0.871 |
| random placement, seed 0 | — | 0.856 |
| random placement, seed 1 | — | 0.940 |

The detector's placement scores best, by about 2% over a fixed periodic
schedule. One random placement landed on the same windows and scored
identically. What dominates the result is that adaptation happens at all: every
arm improves 2.650 to below 1.0.

So the detector does not buy much on the final number here. What it buys is not
having to choose the schedule in advance, which is the position a deployed
monitor is in.

## How the data works

The Well publishes each dataset as HDF5, one file per regime, under
`data/train/`, `data/valid/` and `data/test/`. Two things happen to a regime
before it becomes a window:

1. **Fetched.** One file per split, landed through a `.part` rename, so a file
   that exists is a file that finished.
2. **Converted.** Rewritten as `.npy`, shaped
   `[trajectory, time, channel, *spatial]`, float32. HDF5 is chunked and often
   compressed, so the operating system cannot map it; `.npy` it can. A sample
   then faults in only the pages it touches, which is what lets a regime larger
   than memory be a window.

A downloaded HDF5 is deleted after conversion. Nothing reads it again, it can
be fetched back, and it is half of what a regime costs on disk.

The conversion carries a recipe string. Change the conversion, bump
`CONVERSION_RECIPE`, and existing arrays are rebuilt rather than trusted.

Normalisation comes from The Well's `stats.yaml` and is fixed across regimes.
Normalising per regime would divide out the shift the run is watching for.

## Adding another Well dataset

Every dataset-specific detail lives in one row in [`datasets.py`](datasets.py):

```python
RAYLEIGH_BENARD = WellDataset(
    name="rayleigh_benard",
    fields=(
        ("t0_fields", "buoyancy", None),
        ("t0_fields", "pressure", None),
        ("t1_fields", "velocity", 0),
        ("t1_fields", "velocity", 1),
    ),
    spatial_resolution=(512, 128),
    n_spatial_dims=2,
    order=lambda name: float(name.split("_Rayleigh_")[1].split("_")[0]),
)
```

Three fields to get right:

- `fields` — which HDF5 groups and fields become which channels. The Well
  stores scalars under `t0_fields` and vectors under `t1_fields`, and a vector
  needs a component index. Channel count follows from this, and so does the
  model's input width.
- `spatial_resolution` and `n_spatial_dims` — the grid. Conversion checks it
  against the file, so a wrong row fails with the shape it actually found. The
  U-Net pools four times, so each axis must divide by 16.
- `order` — how to read the swept parameter out of a filename, for sorting.
  This is the one that will bite you. The default takes the trailing number,
  which is only right for a single-parameter sweep:

  ```
  turbulent_radiative_layer_tcool_0.06      -> 0.06   correct
  rayleigh_benard_Rayleigh_1e10_Prandtl_1   -> 1.0    sorts by Prandtl
  shear_flow_Reynolds_1e4_Schmidt_1e-1      -> 0.1    sorts by Schmidt
  viscoelastic_instability_AH               -> inf    no number at all
  ```

  A two-parameter sweep needs its own key function. A dataset whose files are
  named flow states rather than numbers needs an explicit sequence;
  `viscoelastic_instability` is `AH`/`CAR`/`EIT`. That case is arguably more
  interesting, because you choose the order instead of deriving it, so you can
  put several normal regimes ahead of an anomalous one.

`order` also sets how hard the detection problem is. Monotonic cooling time is
what ships and is the physically honest ordering; it gives the detector four of
eight boundaries. Interleaving cool and hot regimes makes each boundary a
bigger step and gets six of eight, which is a better number for a less
meaningful sweep.

Nothing else in the example is dataset-specific. The harness reads the row.

### Picking a dataset

| Dataset | Regimes | GB per regime | Swept parameter | Why |
|---|---|---|---|---|
| `turbulent_radiative_layer_2D` | 9 | 0.68 | cooling time | shipped; the smallest, which is why it is here |
| `active_matter` | 45 | 0.9 | activity, alignment, box size | ordered to disordered, and the only other one in laptop range |
| `rayleigh_benard` | 35 | 7.8 | Rayleigh, Prandtl | the canonical conduction / convection / turbulence ladder |
| `shear_flow` | 28 | 12.5 | Reynolds, Schmidt | laminar to turbulent |
| `viscoelastic_instability` | 7 | 7.5 | named flow states | an anomaly taxonomy rather than a sweep |

One caveat before reaching for a big one. This example keeps every regime it
has touched, because earlier regimes are replayed as historical data. Nine
regimes at 0.68 GB is about 6 GB and fine. For `rayleigh_benard` it would be 35
at 7.8 GB, so roughly 270 GB. A dataset that size needs regimes released once
nothing replays them, which this example does not do.

## Why Page-Hinkley and not ADWIN

The other examples use ADWIN. ADWIN never fires here. Feeding the real
per-batch VRMSE trace from a run, 1,744 batches across all nine regimes, to
river's ADWIN at deltas from 0.002 through 0.9 gives zero detections. Adaptive
windowing looks for a split where two sub-windows differ by more than their own
spread allows, and here the per-batch scatter within a regime is larger than
the step between neighbouring regimes.

Page-Hinkley accumulates a one-sided departure from a running mean, which is
the shape a regime change leaves in a model's error. Its settings come from
replaying that same trace with learning switched off and scoring candidates two
ways: boundaries found, and how often the same settings fire on shuffled copies
of the same values, where no boundaries are left to find. The second number is
the one that matters.

The shipped settings fire four times on the real stream against 1.75 on
shuffled, a ratio of 2.3. Settings that fire nine times, and so land near all
eight boundaries, fire ten times on shuffled data. That is detection by
arithmetic.

Four of eight is what the signal supports. The fifth boundary, `tcool` 1.78 to
3.16, is a +0.5 sigma step arriving straight after a +1.75 sigma one, and no
single absolute threshold covers both. A log transform, the `ph_alpha`
forgetting factor, and an ensemble with `any` voting over Page-Hinkley, ADWIN
and KSWIN all cap at four.

A run that adapts fires three times rather than four. Each adaptation flattens
the error climb, so later regimes arrive at a model that already handles them.
Tuning against a learning-off trace over-predicts how often a real run fires.

See [`docs/choosing_a_detector.md`](../../docs/choosing_a_detector.md).

## Why the U-Net is vendored

[`unet.py`](unet.py) is The Well's `UNetClassic` (BSD-3, itself adapted from
PDEBench), copied in rather than imported. Importing it from `the_well` reaches
their model package's `__init__`, which imports their FNO and so pins
`neuraloperator==0.3.0`. That is a heavy and exact dependency for one example.
The copy needs torch and nothing else.

Layer names and shapes are untouched, because a published checkpoint only loads
into the architecture it was trained with. PolymathicAI's weights for this
dataset are `init_features=48`: set `[model] name = "unet"` and
`pretrained_path` to a downloaded `model.safetensors` to start from one. That
width is 17.5M parameters against 1.94M, so it wants a GPU.

### One thing the harness changes

BatchNorm holds its running statistics still during training. Every window is a
different physical regime, so live batch statistics follow whichever regime is
current, which is a second adaptation underneath the one being measured. Twenty
forward passes in `train()` mode with no optimizer take The Well's published
checkpoint from 0.204 to 1.569 VRMSE on the regime it was trained for.

Freezing them adds no parameters and renames nothing, so published checkpoints
still load with `strict=True`. Without it this run adapts and then gets worse:
1.146 after the first event, 1.481 after the second.

## Files

| File | What it is |
|---|---|
| `datasets.py` | The registry, the fetch, and the HDF5 to `.npy` conversion |
| `model.py` | The harness: one regime per window, VRMSE, replay of earlier regimes |
| `unet.py` | The Well's U-Net baseline, vendored (BSD-3) |
| `evaluate.py` | Scores a run's checkpoints against the test split |
| `well_trl2d.toml` | The config |

## Citation

```bibtex
@inproceedings{ohana2024thewell,
  title={The Well: a Large-Scale Collection of Diverse Physics Simulations
         for Machine Learning},
  author={Ohana, Ruben and McCabe, Michael and Meyer, Lucas and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
