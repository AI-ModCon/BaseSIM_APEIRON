# Multi-node / multi-GPU (data-parallel)

Apeiron scales a monitoring run across processes with **data parallelism**: each
rank works on a contiguous shard of the window (inference and training), and the
ranks coordinate only at decision points. It is built for HPC (Frontier /
Perlmutter) but runs on CPU with the `gloo` backend for local testing.

## The `comm` facade

Everything distributed goes through `apeiron.distributed.comm`, a thin facade
over `torch.distributed`. Its defining property: **when the job was not launched
distributed (`world_size == 1`), every collective is an identity / no-op**. That
is what lets the engine, harness, and trainer call
`comm.all_gather_object` / `comm.broadcast_object` / `comm.all_reduce_mean_`
unconditionally while the single-process path stays byte-for-byte unchanged.

```python
from apeiron.distributed import comm
comm.init_from_env()          # detects the launch, creates the process group
comm.rank, comm.world_size    # this process's place in the job
comm.is_main                  # rank 0 -- owns all writes
comm.is_distributed           # world_size > 1
comm.shard()                  # (rank, world_size) or None single-process
comm.shutdown()
```

Launch detection supports the two ways these jobs start:

* **torchrun** — `RANK` / `WORLD_SIZE` / `LOCAL_RANK` (+ `MASTER_ADDR` / `MASTER_PORT`).
* **SLURM srun** — `SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID`; the
  rendezvous host is derived from `SLURM_NODELIST` if unset.

> A plain `sbatch` (no `srun`) sets `SLURM_NTASKS` but **not** `SLURM_PROCID`, so
> detection correctly reports single-process — the shipped example `.sbatch`
> files run one rank regardless of their `--ntasks-per-node`. You get multi-rank
> only by actually launching N processes with `srun` or `torchrun`.

Backend is `nccl` when CUDA/ROCm is available (ROCm's RCCL is exposed as
`nccl`), else `gloo`.

## What is sharded, what is not

Multi-node applies to `WindowedHarness` (the committed-window on-ramp). The
synthetic example harnesses (MNIST/CIFAR/ImageNet) are single-process demos and
do not shard.

| stream | sharded? | why |
|---|---|---|
| monitoring (`get_stream_dataloader`) | **yes** — `shard=(rank, world)` | parallel inference; the bottleneck |
| adaptation train split | **yes** | data-parallel updates (gradient all-reduce) |
| historical replay (train) | **yes** | parallel replay |
| validation splits (current + historical) | **no** (kept whole) | every rank computes the same eval metrics |

Contiguous (not strided) shards keep reads aligned with the parallel
filesystem's stripes. Sample order within a window does not matter for either
inference or data-parallel updates, so contiguous blocks are safe.

## Sharded monitoring

Each rank runs inference on its shard, then the ranks **all-gather** their
per-batch metrics into the window's global set (concatenating per-rank lists in
rank order reconstructs global order, because shards are contiguous). Rank 0 —
which owns the single, stateful detector — aggregates, decides, and logs; the
verdict is **broadcast** so every rank fires (and adapts) together.

Distributed runs use **one decision per window** (window cadence) rather than
reproducing single-process interval cadence. This makes the detector's decision
points invariant to the number of nodes. Consequently, detector verdicts are
deterministic **for a fixed world size**; when comparing a detector run against
its schedule/drift-only control arms, run them at the same world size.

## Data-parallel adaptation

On a fire, every rank runs the continual-learning loop together:

1. **Broadcast** rank 0's weights at the start of each adaptation, so all ranks
   train from an identical model (manual data parallel needs this).
2. Each rank computes gradients on its train shard; the (accumulated) gradients
   are **averaged across ranks** (`all_reduce_mean_`) before `optimizer.step()`,
   so every rank takes the same step on an effective batch of
   `world_size × per-rank` samples.

This is manual data parallelism (gradient all-reduce), chosen over the
`DistributedDataParallel` wrapper because it needs no changes to the updater /
model plumbing. The model therefore stays identical across ranks throughout.

## Rank-0-only writes

To avoid duplicate/racing writes, only rank 0 performs side effects: the metrics
logger and CSV (other ranks get a silent logger), checkpoints, and the durable
task-record JSONL. Every rank keeps the *in-memory* task records so BWT agrees
across ranks; only rank 0 persists them.

## Launching

Device binding is automatic: under a distributed launch each rank binds
`cuda:LOCAL_RANK` (skipping the single-process nvidia-smi picker, which would
otherwise fight the launcher's per-rank GPU assignment).

```bash
# One node, N GPUs (torchrun):
torchrun --nproc_per_node=N --nnodes=1 -m src.main --config <windowed.toml>

# Multi-node under SLURM (srun spawns one rank per task):
srun --ntasks-per-node=8 --nodes=2 python -m src.main --config <windowed.toml>
```

## Testing without a cluster

`tests/dist_smoke.py` runs the full path on CPU with `gloo`. The `slow`-marked
`tests/test_distributed_smoke.py` launches it with two processes and asserts the
shards cover the window and the parameters stay in sync across ranks:

```bash
poetry run pytest -m slow tests/test_distributed_smoke.py
# or directly:
DIST_SMOKE_STORE=/tmp/store poetry run torchrun \
    --nproc_per_node=2 --nnodes=1 tests/dist_smoke.py
```
