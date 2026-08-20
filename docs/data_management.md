# Data & Memory Management (committed windows)

This page documents the `apeiron.data` package and the checkpoint/task-record
changes that go with it. It is the storage foundation for scaling apeiron to
long-running, multi-node online streams.

## The window model

An **online dataset grows as committed windows**: data collected over some
`delta_t` is closed ("committed") at a specific point and never changes again.
Because a window is immutable once committed, every consumer can read it
concurrently with no locking, and it can be stored once and mapped many times
instead of being held in RAM.

`apeiron.data.window_store.WindowStore` writes each window to its own directory
under a store root:

```
<root>/<window_id>/
    x.npy          # inputs, one row per sample (np.save, C-contiguous)
    y.npy          # targets, one row per sample
    manifest.json  # shapes, dtypes, splits, timestamps -- written last
<root>/catalog.db  # sqlite index over all manifests (optional)
```

* **Atomic commit.** The writer stages the window in `<root>/.tmp-*` and then
  `os.replace`s the directory into place (one `rename(2)`). A reader either sees
  a fully-formed window or none at all; a crash mid-write leaves only a `.tmp-*`
  directory, which listing ignores. "Committed" = the directory exists and holds
  `manifest.json`.
* **Memory-mapped reads.** `WindowHandle.dataset()/loader()` return
  `torch.utils.data.Dataset`/`DataLoader` objects that `np.load(..., mmap_mode="r")`
  the data and materialize one sample at a time, re-opening the memmap lazily
  inside each DataLoader worker. A whole window is never loaded into memory.
* **Raw dtype on disk.** Store uint8/float16 at sensor precision and cast in the
  read path; storing float32 quadruples I/O for nothing.

### Splits and sharding

A window declares named splits in its manifest:

* **range split** -- contiguous `[lo, hi)` (the common case; `commit(..., val_fraction=0.1)`
  makes the trailing 10% the `val` split).
* **index split** -- an explicit index array on disk, for non-contiguous (e.g.
  random) splits; use `commit_indexed(x, y, train_idx=..., val_idx=...)`.

Because sample order within a window does not matter for either monitoring
inference or data-parallel updates, `WindowHandle.dataset(split, shard=(rank, world))`
returns a **contiguous** block of the split for one rank. This is the hook
multi-node inference/training (a later phase) builds on; on one process it is a
no-op. Contiguous (not strided) shards keep reads aligned with a parallel
filesystem's stripes.

### The catalog

`apeiron.data.catalog.WindowCatalog` is a sqlite index over manifests, updated on
every commit. It answers "which windows do I need?" without globbing the
filesystem:

```python
store.catalog.query(t_start_gte="2026-03-01", t_end_lte="2026-04-01")  # historic load
store.catalog.query(detected=True)                                     # windows a detector fired on
store.catalog.all()                                                    # full-retrain enumeration
```

The catalog is a cache; the manifests on disk are the source of truth. Drop it
and rebuild any time with `store.rebuild_catalog()`.

## WindowedHarness

`apeiron.data.windowed_harness.WindowedHarness` implements every
`BaseModelHarness` data method in terms of a `WindowStore`, mapping the
framework's three streams onto committed windows:

| stream | source |
|---|---|
| monitoring (`get_stream_dataloader`) | a loader over the current window (default: the whole window) |
| adaptation (`get_train_dataloaders`) | the current window's `train`/`val` splits |
| historical (`get_hist_dataloaders`) | the concatenation of all prior windows' `train`/`val` splits, or `None` before any history |

Each committed window is one stream window, so drive a run with
`drift_detection.max_stream_updates` no larger than `len(store)`. The model,
loss, and optimizer are not the store's concern -- pass `criterion=`,
`optimizer_factory=`, `eval_metrics=` to the constructor, or subclass and
override `get_criterion`/`get_optmizer`. Point `[data] window_store_path` at the
store root, or pass a `WindowStore` directly.

The shipped MNIST/CIFAR/ImageNet example harnesses generate affine drift on the
fly and deliberately do **not** use the store -- they are fast synthetic demos.
`WindowedHarness` is the on-ramp for real datasets.

## Task records: pointer, not copy

Backward-transfer (forgetting) measurement needs to re-score the current model on
each past task's validation split. Previously the harness copied that whole split
into RAM per drift event and retained up to 50 of them -- the copy existed only
because an on-the-fly harness drops its window tensors as the stream advances.

With immutable windows, a task's frozen eval set is a **pointer**
(`WindowEvalSetRef`: `window_id` + split) re-derived from the committed window on
demand -- no copy. Harnesses without a store fall back to `InMemoryEvalSet`,
which now spills to a `.pt` file when a checkpoint path is configured, so even
that path is durable and need not stay resident.

Either way a task record is small metadata, so the full task history persists as
JSONL under `<ckpts_path>/task_records/` and `harness.load_task_records()`
restores it -- a crashed multi-day run recovers its forgetting history instead of
losing it with the process.

## Checkpoints: retention and promotion by rule

`apeiron.model.checkpoint.CheckpointStore` writes a metric **sidecar**
(`drift_adaptation_<event>.json`) next to each checkpoint, recording the post-CL
metrics in hand at save time (`test_curr_acc`, `test_hist_acc`, `fwt`, `bwt`).
"Which checkpoints do I keep" and "which do I serve" then become rules over those
sidecars:

* **Retention** (`[model] ckpts_retention`) -- which files survive. A spec is one
  or more clauses joined by `+`; the keep set is their union:
  * `fifo` / `latest` -- newest N by event (N = `max_ckpts` when sole clause);
  * `max:<metric>` / `best:<metric>` -- top N by metric (e.g. `max:test_hist_acc`);
  * `min:<metric>` -- top N by a lower-is-better metric (e.g. `min:loss`);
  * e.g. `latest:1+max:test_hist_acc+max:test_curr_acc` keeps the last plus the
    best-historical and best-current snapshots.
* **Promotion** (`[model] deploy_rule`) -- a single-winner spec naming the
  checkpoint the `deployed` pointer file tracks (e.g. `max:test_hist_acc`). Kept
  separate from retention because you usually retain generously but deploy the
  single most robust model.

The default `ckpts_retention = "fifo"` reproduces the previous keep-newest-N
behavior.

## How this maps to the scaling goals

* **Memory management** -- windows and frozen eval sets are written to disk and
  memory-mapped, not held in RAM; `register_task` no longer copies.
* **Metadata management** -- the catalog makes "which datasets do I need for a
  historic load / full retrain" a query, not a directory walk.
* **Deployment management** -- checkpoint retention and promotion are rules over
  recorded metrics (best hist acc, best current acc, latest, ...).
* **Analysis of previous windows** -- immutable windows + per-event checkpoints
  make faithful offline replay possible (re-run a detector variant over a
  recorded metric trace, or re-adapt on stored windows to tune CL hyperparameters,
  then emit a proposed config delta). The storage foundation for this lands here;
  the offline advisor job itself is future work.

## Quick example

```python
import numpy as np
from apeiron.data import WindowStore

store = WindowStore("/data/windows")
for x, y, (t0, t1) in produce_partitions():           # your producer
    store.commit(x, y, val_fraction=0.1, t_start=t0, t_end=t1, detected=False)

# later, anywhere (even a different process):
store = WindowStore("/data/windows")
for handle in store.windows():
    loader = handle.loader("val", batch_size=256)      # memmap, not resident
    ...
ids = store.catalog.query(t_start_gte="2026-03-01")    # historic-load selection
```
