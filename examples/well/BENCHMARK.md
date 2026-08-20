# Benchmark recipe — measuring the memory-scaling branch

A sequential runbook. Steps 0–2 are one-time setup + building the data. Steps 3–6
are the four measurements. Each measurement writes a CSV; step 7 combines/plots.

Everything works on a laptop (CPU/`gloo`, small sizes) and on Perlmutter
(GPU/`nccl`, real sizes). The only difference is scale and the launcher.

> **Key constraint for multi-rank runs:** each window's *train split* is sharded
> across ranks, so it must have at least `world_size` samples — comfortably
> `world_size × train.batch_size`. The converter stacks **all trajectories** of a
> file into each window (`turbulent_radiative_layer_2D` = 8/file), so a window
> holds `trajectories × (window_steps − 1)` samples — e.g. `--window-steps 100`
> gives ~792, feeding ~18 ranks at batch 32. Raise `--window-steps` (and lower
> `--batch-size`) if a rank gets an empty shard; `--max-trajectories` caps
> stacking.

---

## 0. Setup (once)

```bash
git clone <your-fork> && cd BaseSIM_APEIRON && git checkout memory-scaling
poetry install                       # installs h5py, psutil, torch, ...
poetry run pytest -q -m "not slow"   # sanity: 319 passed
```

On **Perlmutter**, do this on a login node; use the repo's
`src/apeiron/deployment/perlmutter/install_venv.sh` to build the venv against the
system CUDA torch.

---

## 1. Build the window store(s)

The benchmark reads a committed `WindowStore`. Build it **once** (login node or a
single-task job) before any multi-rank run.

**Fixture (no download, fully size-knobbed — best for controlled scaling):**

```bash
# grid 128x384 (real resolution), 7 regimes, 40 steps each:
poetry run python -m examples.well.fixture data/wellsrc \
    --n-time 40 --height 128 --width 384
poetry run python -m examples.well.convert --files "data/wellsrc/*.hdf5" \
    --out data/wellstore --window-steps 20 --val-fraction 0.25
```

**Real Well data (authenticity run):**

```bash
poetry run python -m examples.well.convert --dataset turbulent_radiative_layer_2D \
    --split train --max-files 8 --out data/wellstore_real --window-steps 20
```

`convert` prints the window count, channels, grid, and regime order. Bigger knobs
→ bigger store: `--height/--width` grow each field (memory), `--n-time` /
`--max-files` grow the window count (work).

For **weak scaling** build a family of stores whose size scales with the rank
count, e.g. `wellstore_1x` … `wellstore_8x` (more `--n-time` or `--max-files`).

---

## 2. Pick your launcher

| | single process | multi-rank |
|---|---|---|
| laptop | `poetry run python -m ...` | `poetry run torchrun --nproc_per_node=N -m ...` |
| Perlmutter | `python -m ...` | `srun --ntasks-per-node=4 --gpus-per-node=4 python -m ...` |

> On Perlmutter, **do not** set `--gpu-bind`/`--gpus-per-task`: leave all 4 A100s
> visible per node so each rank binds `cuda:LOCAL_RANK` (which `comm` reads from
> `SLURM_LOCALID`). `comm` also derives `MASTER_ADDR` from `SLURM_NODELIST`; set
> `MASTER_PORT` if 29500 is taken.

---

## 3. Memory A/B — the capability headline (single node)

Pointer (new) vs copy (pre-Phase-1) task eval sets as retained-task count grows.

```bash
poetry run python -m examples.well.benchmark memory \
    --store data/wellstore --out out/mem.csv \
    --task-counts 1,10,50,200,500
```

**Read `out/mem.csv`:** `rss_added_mb` stays ~**0** for `mode=pointer` at every
task count; for `mode=copy` it grows ≈ `val_split_mb × tasks` (see
`predicted_copy_mb`). At real resolution the copy column reaches GBs and will OOM
the node at high task counts — that is the headline: *pointer makes
full-history BWT feasible where copy cannot*. Push `--task-counts` until copy
OOMs to find the ceiling.

---

## 4. Throughput / scaling (multi-rank)

One run per world size; each writes its own CSV.

**Strong scaling** — fixed `data/wellstore`, vary ranks:

```bash
for N in 1 2 4; do                                   # laptop / one node
  poetry run torchrun --nproc_per_node=$N --master_port=29500 \
    -m examples.well.benchmark throughput \
    --store data/wellstore --out out/strong_w$N.csv --width 128 --batch-size 16
done
```

**Weak scaling** — store size ∝ ranks:

```bash
for N in 1 2 4; do
  poetry run torchrun --nproc_per_node=$N -m examples.well.benchmark throughput \
    --store data/wellstore_${N}x --out out/weak_w$N.csv --width 128 --batch-size 16
done
```

**Perlmutter (sbatch, one node = 4 ranks; bump `--nodes` for more):**

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 0:30:00
#SBATCH -o out/scale_%j.log
source .venv/bin/activate
export WANDB_MODE=offline
srun python -m examples.well.benchmark throughput \
    --store $SCRATCH/wellstore --out out/scale_w${SLURM_NTASKS}.csv \
    --width 256 --batch-size 32
```

**Read the CSVs:** `wall_s` and `samples_per_s` give the scaling curves —
speedup `T(1)/T(N)`, efficiency `speedup/N` (strong); flat `wall_s` (weak). The
`comm_all_gather_s` / `comm_broadcast_s` / `comm_all_reduce_s` columns are the
**Amdahl breakdown** — `all_reduce` (gradient sync) dominates and grows with model
size and N; if it swamps compute, scale `--width` up so each rank does more work
per step.

> On CPU/`gloo` (laptop) more ranks is *slower* (oversubscription) — that only
> validates the mechanism + records the comm split. Real speedup needs GPUs/NCCL.

---

## 5. Frontier preservation — the correctness guardrail

Final error vs adaptation budget, at each world size. Confirms scaling didn't
move the science.

```bash
# W=1:
poetry run python -m examples.well.benchmark frontier \
    --store data/wellstore --out out/frontier_w1.csv --budgets 0,1,3,7,14
# W=2 (and 4, ...): same, under the launcher
poetry run torchrun --nproc_per_node=2 -m examples.well.benchmark frontier \
    --store data/wellstore --out out/frontier_w2.csv --budgets 0,1,3,7,14
```

**Read:** plot `final_vrmse` vs `triggers`. It should fall monotonically (more
adaptation → lower error) and the curves for W=1,2,4 should overlap within seed
noise. Non-overlap means the distributed path changed behavior — investigate.

---

## 6. Resumability — the durability guardrail (single node)

```bash
poetry run python -m examples.well.benchmark resume \
    --store data/wellstore --out out/resume.csv --ckpt-dir out/ck
```

**Read:** `tasks_reloaded == tasks_before` and `diagonals_match=True` — a fresh
process recovered the full forgetting history + model from disk.

---

## 7. Collect + plot

Combine the per-world-size throughput CSVs and plot:

```python
import glob, pandas as pd, matplotlib.pyplot as plt

scale = pd.concat([pd.read_csv(f) for f in glob.glob("out/strong_w*.csv")])
scale = scale.sort_values("world_size")
base = scale.iloc[0]["wall_s"]
scale["speedup"] = base / scale["wall_s"]
scale["efficiency"] = scale["speedup"] / scale["world_size"]
print(scale[["world_size", "wall_s", "samples_per_s", "speedup", "efficiency",
             "comm_all_reduce_s", "final_vrmse"]])

mem = pd.read_csv("out/mem.csv")
for mode, g in mem.groupby("mode"):
    plt.plot(g["tasks"], g["rss_added_mb"], marker="o", label=mode)
plt.xlabel("retained tasks"); plt.ylabel("RSS added (MB)"); plt.legend()
plt.savefig("out/memory_ab.png")
```

The deliverables: **memory_ab** (pointer flat vs copy diverging/OOM), **strong &
weak scaling** tables, **Amdahl** column (`comm_all_reduce_s` fraction), and
**frontier overlap** across world sizes — memory capability + throughput +
correctness in one report.
