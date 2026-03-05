# MATEY Example Harness

BaseSim harness for the [MATEY](https://github.com/FusionFM/MATEY) multiscale transformer codebase.

## Setup

1. Install BaseSim (from repo root):
   ```bash
   poetry install
   ```

2. Install the optional MATEY example dependency (pinned commit):
   ```bash
   poetry install --extras matey
   ```
   This uses GitHub SSH auth for the private MATEY repo, so your SSH key must
   have access to `FusionFM/MATEY`.

   This extra is pinned to:
   `4e615bb5c86024632e386153bfbed028b38a8262`

   Equivalent pip command:
   ```bash
   pip install "matey @ git+ssh://git@github.com/FusionFM/MATEY.git@4e615bb5c86024632e386153bfbed028b38a8262"
   ```

3. (Optional) Install heavy/system-dependent packages as needed for your environment:
   ```bash
   MAX_JOBS=4 NINJA_STATUS="[%f/%t] " pip install -vv --progress-bar on --no-build-isolation flash-attn
   pip install dadaptation==3.1                             # for DAdaptAdam optimizer
   pip install mpi4py                                       # requires MPI C library
   pip install netCDF4                                      # requires HDF5/netCDF C libs
   pip install git+https://github.com/sandialabs/exodusii.git # not on PyPI
   ```

   Alternative: install flash attention without screen output 
   
   ``` 
   MAX_JOBS=4 pip install flash-attn --no-build-isolation  # requires CUDA toolkit + nvcc
   ```
## Running

```bash
poetry run python -m src.main --config examples/matey/matey.toml
```

Outer-loop drift demo (L2 placeholder model + input-noise stream updates):

```bash
poetry run python -m src.main --config examples/matey/matey_outer_loop.toml
# or
./examples/matey/run_outer_loop.sh
```

## Configuration

Edit [matey.toml](matey.toml) to adjust training parameters, drift detection, and data paths.

The `[data].path` should point to your local SOLPS dataset root
(must contain `train/` and `valid/` directories).
This data is expected to be user-provided and is not tracked in git.

For the SOLPS example, the harness builds a deterministic file-level split of
`[0.7, 0.15, 0.15]` and materializes staged views under:

```text
output/matey_split_cache/<fingerprint>/{train,val,test}
```

The cache is reused when source files, split ratios, and seed are unchanged.

The example TOML is tuned for short smoke runs to make drift-triggered continual
learning dispatch easier to observe (`detection_interval=5`, `aggregation="last"`,
`adwin_delta=0.05`, `max_stream_updates=10`).

For the outer-loop harness, use [matey_outer_loop.toml](matey_outer_loop.toml):
- `data.name = "matey_outer_loop"` selects `model_outer_loop.py`.
- `data.path` points at `examples/matey/dump/SOLPS2DwION`.
- `continual_learning.update_mode = "none"` disables parameter updates.
- `drift_detection.metric_index = 0` monitors the `input_l2` metric.

## Files

| File | Description |
|---|---|
| `model.py` | `MATEYHarness` -- adapts MATEY models/data to BaseSim's `BaseModelHarness` interface |
| `matey.toml` | Experiment config |
| `model_outer_loop.py` | Outer-loop drift harness with L2 placeholder model and noisy input stream |
| `matey_outer_loop.toml` | Outer-loop experiment config |
| `run_outer_loop.sh` | Convenience run script for `matey_outer_loop.toml` |
| `pyproject.toml` | Optional example dependency manifest |
