# Prometheus Example

This example integrates an AGN-201 reactor time-series prediction model into the
BaseSim continual learning framework. The model is a Stacked LSTM that predicts
reactor output from control rod positions and period measurements.

## Folder structure

```
examples/
└── prometheus/             # PyTorch version — runs inside BaseSim
    ├── model.py            # StackedLSTM architecture + PrometheusHarness
    ├── utils.py            # CSV loading, normalization, sequence dataset
    ├── prometheus.toml     # Experiment configuration
    └── TemporalPredict.py  # Original TensorFlow source (reference only)

```

## What was done

### TensorFlow → PyTorch conversion (`examples/prometheus/`)

The original `TemporalPredict.py` was a single-file TensorFlow script. It was
converted to a PyTorch `BaseModelHarness` subclass so it can be driven by the
BaseSim `ContinuousMonitor` loop with drift detection and continual learning.

| TF original | PyTorch equivalent |
|---|---|
| `tf.keras.Sequential` (LSTM × 2 → Dropout → Dense × 2) | `StackedLSTM(nn.Module)` in `model.py` |
| `ModelGeneration.repeated_trainer` (optimizer + fit loop) | `PrometheusHarness.get_optmizer()` + framework training loop |
| Per-CSV DataFrames fed sequentially | `SequenceDataset` + `DataLoader` in `utils.py` |
| Single training run over all CSV batches | `update_data_stream()` advancing one CSV chunk per call |
| No replay | `get_hist_data_loaders()` returns `ConcatDataset` of prior task data |
| MSE + Adam (lr=0.001) | `nn.MSELoss` + `torch.optim.Adam` — same hyperparameters |

**Architecture** (identical to TF):
```
LSTM(n_features → 64, tanh)
LSTM(64 → 64, tanh)
Dropout(0.1)
Linear(64 → 32, tanh)
Linear(32 → n_targets)          # sigmoid for single target, linear for multiple
```

**Drift simulation**: CSV files under `data.path` are split into `NUM_TASKS`
chunks. Each call to `update_data_stream()` loads the next chunk, simulating
new operational data arriving in successive batches.

## Running the experiment

```bash
poetry run python -m src.main --config examples/prometheus/prometheus.toml
```

### Data layout

Place operational CSV files in the directory set by `data.path`
(default: `data/prometheus/`). All files must share the same column schema.

```
data/prometheus/
├── 2025-02-27.csv
├── 2025-03-12.csv
└── ...
```

## Configuration

Key settings in `prometheus.toml`:

```toml
[data]
path = "data/prometheus"   # folder containing operational CSV files

[train]
batch_size = 1             # use 1 for small test datasets (current: 4 CSVs)
                           # increase to 32–64 when more operational data is available

[drift_detection]
max_stream_updates = 10    # should match NUM_TASKS in the harness
metric_index = 0           # monitors MSE loss
```

> **TODO**: `batch_size` is currently set to `1` for the small 4-file test
> dataset. **Change this to `32` (original TF default) before running on the
> full operational dataset.** Set it in `prometheus.toml` under `[train]`.

### Feature / target columns

Confirmed from the production usage of `ModelGeneration`:

| Role | Columns |
|---|---|
| Features (12) | `NRAD_RX_REG_POS`, `NRAD_RX_SHIM1_POS`, `NRAD_RX_SHIM2_POS`, `total_rod_position`, `NRAD_RX_PERIOD_Inverse`, `NRAD_RX_REG_POS_dt`, `NRAD_RX_REG_POS_dt2`, `NRAD_RX_SHIM1_POS_dt`, `NRAD_RX_SHIM1_POS_dt2`, `NRAD_RX_SHIM2_POS_dt`, `NRAD_RX_SHIM2_POS_dt2`, `NRAD_RX_NMP1_PWR_integral` |
| Target (1) | `NRAD_RX_NMP1_PWR` |

These are set as class-level constants in `PrometheusHarness`
(`examples/prometheus/model.py`) and can be changed there if needed:

```python
FEATURE_VARIABLES: List[str] = [
    "NRAD_RX_REG_POS", "NRAD_RX_SHIM1_POS", "NRAD_RX_SHIM2_POS",
    "total_rod_position", "NRAD_RX_PERIOD_Inverse",
    "NRAD_RX_REG_POS_dt", "NRAD_RX_REG_POS_dt2",
    "NRAD_RX_SHIM1_POS_dt", "NRAD_RX_SHIM1_POS_dt2",
    "NRAD_RX_SHIM2_POS_dt", "NRAD_RX_SHIM2_POS_dt2",
    "NRAD_RX_NMP1_PWR_integral",
]
TARGET_VARIABLES: List[str] = ["NRAD_RX_NMP1_PWR"]
```

Other tunable constants in the same class:

| Constant | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | `30` | Timesteps per LSTM input window |
| `NUM_TASKS` | `10` | Number of CSV chunks (drift iterations) |
| `VAL_RATIO` | `0.2` | Fraction of each chunk held out for validation |
