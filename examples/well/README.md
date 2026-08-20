# The Well: streaming neural PDE surrogate under regime drift

A real scientific-ML continual-learning problem, and the benchmark for the
memory-scaling branch.

[The Well](https://polymathic-ai.github.io/the_well) is a 15 TB collection of
physics-simulation datasets (Polymathic AI / Flatiron). We use
`turbulent_radiative_layer_2D`, whose files are each a different **cooling time**
`tcool` — a clean physical drift axis. The task is **next-step field prediction**
(a neural surrogate maps the fields at time *t* to *t+1*), which is regression
(VRMSE / MAE, lower is better). Streaming the windows in `tcool` order makes the
input distribution drift, so a surrogate trained on early regimes degrades — the
drift detector catches it and continual learning adapts, ideally without
forgetting earlier regimes.

This is a faithful stand-in for the sensor/instrument regression apeiron targets
(e.g. SLAC-FEL), and it stresses exactly what the branch scaled: large fields
(memory), an expensive surrogate (multi-node), and a long drifting stream.

## Why it exercises the whole framework

* **WindowStore** — each committed window is a time chunk of a trajectory as
  memmapped next-step pairs; storage stays on disk (the memory win).
* **WindowedHarness / Phase 3** — the harness shards the stream/train loaders, so
  the same run goes data-parallel across ranks unchanged.
* **Regression path** — VRMSE/MAE are lower-is-better, exercising the BWT/FWT
  sign handling the classification examples never hit.
* **Real drift** — crossing `tcool` regimes is physical distribution shift, not
  synthetic affine noise.

## Quick start (no download — fixture)

```bash
# 1. Write Well-schema HDF5 with parameter-dependent dynamics (real drift):
poetry run python -m examples.well.fixture /tmp/wellsrc --n-time 40 --height 64 --width 96

# 2. Materialize a committed, drift-ordered WindowStore:
poetry run python -m examples.well.convert --files "/tmp/wellsrc/*.hdf5" \
    --out /tmp/wellstore --window-steps 20

# 3. Point the config at it and run the full monitor + CL loop:
poetry run python -m src.main --config examples/well/well.toml \
    --set data.path=/tmp/wellstore --set data.window_store_path=/tmp/wellstore
```

## Real Well data

```bash
# Downloads the N smallest HDF5 files of a split (109 MB each for this dataset):
poetry run python -m examples.well.convert --dataset turbulent_radiative_layer_2D \
    --split test --max-files 6 --out /data/wellstore --window-steps 20
```

Needs `h5py` (a project dependency) and network access to the HuggingFace Hub.
Other Well datasets work too — the reader assembles `t0/t1/t2` fields
generically; pass `--dataset <name>`.

## Multi-node on Perlmutter

The harness is a `WindowedHarness`, so data parallelism is automatic (see
`docs/distributed.md`). Build the store once (rank 0 / a login node), then:

```bash
srun --ntasks-per-node=4 python -m src.main --config examples/well/well.toml
```

Scale the surrogate with `[model] width`/`depth` (or `--set model.width=...`) so
each rank does enough compute to amortize the gradient all-reduce.

## Measuring the improvement

`examples/well/benchmark.py` has four modes (each writes a CSV):

```bash
# Memory: pointer (new) vs copy (pre-Phase-1) task eval sets, vs task count.
poetry run python -m examples.well.benchmark memory --store /tmp/wellstore \
    --out out/mem.csv --task-counts 1,10,50,200

# Throughput / scaling: launch under torchrun/srun, vary the world size.
# comm.timings() gives the Amdahl breakdown (all-gather / broadcast / all-reduce).
torchrun --nproc_per_node=4 -m examples.well.benchmark throughput \
    --store /data/wellstore --out out/scale_w4.csv --width 128

# Accuracy-cost frontier: final error vs adaptation budget (run at each world size).
poetry run python -m examples.well.benchmark frontier --store /tmp/wellstore \
    --out out/frontier.csv --budgets 0,1,3,7,14

# Resumability: kill/resume preserves the forgetting history + model.
poetry run python -m examples.well.benchmark resume --store /tmp/wellstore \
    --out out/resume.csv --ckpt-dir out/ck
```

* **memory** — `rss_added_mb` is flat for `pointer`, grows ~`val_split × tasks`
  for `copy`. The capability headline: at full field resolution and full task
  history, `copy` is GBs and OOMs; `pointer` is flat.
* **throughput** — strong scaling (fixed data, vary ranks) and weak scaling
  (grow data with ranks) from `wall_s` / `samples_per_s`; `comm_*_s` is the
  serial/communication fraction.
* **frontier** — `final_vrmse` vs `triggers`; compare across world sizes to
  confirm scaling did not move the science.
* **resume** — `tasks_reloaded == tasks_before` and `diagonals_match`.

## Bring your own data

Swap the converter's reader. `convert_files` takes any files that
`examples.well.wellio.read_well_file` can turn into a `[time, channel, H, W]`
trajectory; for a non-Well source, replace `read_well_file` (or commit windows
to a `WindowStore` directly) and everything downstream — harness, model,
detector, CL, sharding, benchmark — is unchanged.

## Files

| file | purpose |
|---|---|
| `wellio.py` | read a Well HDF5 file into `[time, channel, H, W]` |
| `fixture.py` | write Well-schema HDF5 with parameter-dependent drift (no download) |
| `convert.py` | trajectories → committed, drift-ordered `WindowStore` (+ HF download) |
| `model.py` | `PDESurrogate` (residual next-step CNN) + `WellHarness` |
| `well.toml` | example config |
| `benchmark.py` | memory / throughput / frontier / resume measurements |
