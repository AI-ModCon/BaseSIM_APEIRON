# MATEY Example Harness

BaseSim harness for the [MATEY](https://github.com/FusionFM/MATEY) multiscale transformer codebase.

## Harness overview

| Config | Script | Model | Drift signal | Will ADWIN fire? |
|--------|--------|-------|--------------|------------------|
| `matey_outer_loop.toml` | `run_outer_loop.sh` | Placeholder L2 | Synthetic noise → `input_l2` | **Yes** (by design) |
| `matey_inference_drift.toml` | `run_inference_drift.sh` | Real TurBT ViT | NRMSE on real forward passes | Only if NRMSE shifts |
| `matey.toml` | `python -m src.main ...` | Real ViT + CL | NRMSE | Depends on data |

### How drift detection works

1. `ContinuousMonitor` evaluates batches from the validation loader.
2. Every `detection_interval` batches, metrics are aggregated (`mean`, `median`, or `last`).
3. **ADWIN** receives the monitored metric (`metric_index`: 0 = first eval metric).
4. If ADWIN detects a statistically significant change → continual learning dispatch (when `update_mode != "none"`).

**Outer loop** injects increasing Gaussian noise each stream, so `input_l2` jumps and drift fires reliably.

**Inference drift** uses real ViT NRMSE. Drift fires only when the metric actually changes between streams (e.g. different machines/shots). Re-shuffling the same three KSTAR example files produces ~flat NRMSE (~0.07) and **no drift** — that is expected, not a failure.

---

## Frontier (OLCF) quick start

### Environment

On Frontier, use the shared **matey-env** (ROCm 6.3.1). The run scripts source it automatically:

```bash
source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
export PYTHONPATH="/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:$(pwd)/src:$(pwd)"
```

For wandb v1 API keys (`wandb_v1_*`, ~86 chars), upgrade once on a login node:

```bash
./examples/matey/setup_wandb.sh
```

### GPU allocation

Real ViT inference requires a GPU compute node:

```bash
srun --account=lrn097 --partition=batch \
     --nodes=1 --ntasks=1 --gpus=1 --time=1:00:00 \
     --pty bash
```

Login node is fine for staging data, syncing wandb, and plotting CSVs.

### Shared paths (Frontier)

| Resource | Path |
|----------|------|
| SOLPS2D baseline | `/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps` |
| March 2026 checkpoint | `.../models/Dev_Fusion_Demo_March2026_Final/demo_nbatchsloc100/training_checkpoints/best_ckpt.tar` |
| Alternate SOLPS copy (lrn097) | `/lustre/orion/lrn097/proj-shared/fusionMT-data/solps` |

**Data format note:** Only `solps-kstar_example-*.nc` files (dimension `nt`) work with the current `SOLPS2D` loader. Raw `b2time.nc` under `SOLPS2DwION/D3D|KSTAR|SPARC` uses a different schema and fails with `KeyError: 'nt'` until a SOLPS2DwION loader is wired in.

---

## Running

### 1. Outer loop — validate drift pipeline (CPU ok)

Synthetic noise increases each stream; ADWIN should detect drift and log `Drift check #N: ... detected=True`.

```bash
cd BaseSIM_APEIRON
./examples/matey/run_outer_loop.sh
# optional: ENABLE_WANDB=1 with WANDB_API_KEY set
```

**Expect:** console drift-check lines every 5 batches; `output/matey_outer_loop.csv`; optional wandb project `matey-outer-loop-drift`.

### 2. Inference drift — real ViT NRMSE (GPU required)

**Sanity check (same domain, no drift expected):**

```bash
export BASELINE=/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps
export SHIFT="${BASELINE}"
export CKPT=/lustre/orion/fus183/proj-shared/MATEY/models/Dev_Fusion_Demo_March2026_Final/demo_nbatchsloc100/training_checkpoints/best_ckpt.tar
export WANDB_MODE=offline
export ENABLE_WANDB=1
# read -rs WANDB_API_KEY && export WANDB_API_KEY

./examples/matey/run_inference_drift.sh "${BASELINE}" "${SHIFT}" "${CKPT}" \
  --set drift_detection.max_stream_updates=6 \
  --set drift_detection.detection_interval=5
```

**Expect:** streams alternate `(baseline domain)` / `(shift domain)`; NRMSE logged to CSV/wandb; message `CL dispatch did not run because no drift was detected`.

**Weak cross-file shift (pipeline test, usually no ADWIN trigger):**

```bash
./examples/matey/stage_solps_shift.sh
# then set SHIFT to the printed scratch path and re-run run_inference_drift.sh
```

**Strong drift** needs genuinely different SOLPS2D data (e.g. `lrn037` SOLPS if you have access) or future SOLPS2DwION loader support.

Extra CLI overrides are forwarded after the third argument:

```bash
./examples/matey/run_inference_drift.sh "${BASELINE}" "${SHIFT}" "${CKPT}" \
  --set verbosity=INFO:2 \
  --set continual_learning.update_mode=base
```

### 3. Full continual learning

```bash
poetry run python -m src.main --config examples/matey/matey.toml
```

---

## Metrics and visualization

Per-batch **eval metrics** (`eval/nrmse`, `eval/rmse`, `eval/loss`) go to **CSV and wandb**, not the terminal. The terminal shows stream labels, tqdm progress, and (at `verbosity=INFO:1+`) drift-check summaries every `detection_interval` batches:

```text
Drift check #12: metric_0=0.073142, detected=False, score=0.0000
```

| Output | Path |
|--------|------|
| Metrics CSV | `output/matey_inference_drift.csv` or `output/matey_outer_loop.csv` |
| Offline wandb | `wandb/offline-run-*` |

### wandb on Frontier

- **Compute node:** set `WANDB_MODE=offline` (no outbound internet).
- **Login node:** sync with upgraded wandb (do **not** use bare `wandb` on PATH — it picks matey-env 0.19.x):

```bash
cd BaseSIM_APEIRON
unset PYTHONPATH
export PYTHONPATH="${HOME}/.local/23.11.0-0/lib/python3.10/site-packages"
read -rs WANDB_API_KEY && echo && export WANDB_API_KEY
/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rcom631/bin/python -m wandb sync wandb/offline-run-*
```

Plot NRMSE locally from CSV:

```bash
python3 << 'EOF'
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("output/matey_inference_drift.csv")
n = df[df.metric == "eval/nrmse"].copy()
n["value"] = pd.to_numeric(n["value"])
n.plot(x="step", y="value", figsize=(12, 4), title="NRMSE")
plt.ylabel("NRMSE"); plt.tight_layout()
plt.savefig("output/matey_inference_drift_dashboard.png", dpi=150)
print("Saved output/matey_inference_drift_dashboard.png")
EOF
```

---

## Poetry setup (non-Frontier)

1. Install BaseSim (from repo root):

   ```bash
   poetry install
   ```

2. Install the optional MATEY example dependency (pinned commit):

   ```bash
   poetry install --extras matey
   ```

   Pinned commit: `4e615bb5c86024632e386153bfbed028b38a8262`

---

## Configuration

Edit the TOML files to adjust training, drift detection, and data paths.

`[data].path` must point to a SOLPS root with `train/` and `valid/` subdirectories. The harness builds a deterministic file-level split `[0.8, 0.1, 0.1]` and caches staged views under:

```text
output/matey_split_cache/<fingerprint>/{train,val,test}
```

Key drift settings (`matey_inference_drift.toml`):

- `drift_detection.metric_index = 0` → **NRMSE**
- `detection_interval = 5` → check every 5 batches
- `aggregation = "last"` → use last batch metric in window
- `max_stream_updates = 10` → number of stream reloads before stop

Outer loop uses `adwin_delta = 0.05` (more sensitive); inference drift defaults to `0.01`.

See also `src/notes/BaseSIM_APEIRON/INTEGRATION_PLAN.md` for the full integration roadmap.

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Real ViT harness — checkpoint-aware TurBT loading |
| `model_outer_loop.py` | Placeholder L2 model + synthetic noise streams |
| `model_inference_drift.py` | Real ViT + baseline/shift domain toggle |
| `run_outer_loop.sh` | Run outer-loop drift demo |
| `run_inference_drift.sh` | Run real ViT inference drift (sources matey-env, wandb, MIOpen cache) |
| `stage_solps_shift.sh` | Stage a SOLPS2D shift domain under scratch |
| `setup_wandb.sh` | Upgrade wandb in user site-packages for v1 API keys |
| `matey.toml` / `matey_outer_loop.toml` / `matey_inference_drift.toml` | Experiment configs |
