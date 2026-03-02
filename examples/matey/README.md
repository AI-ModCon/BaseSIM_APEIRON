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
   This extra is pinned to:
   `4e615bb5c86024632e386153bfbed028b38a8262`

   Equivalent pip command:
   ```bash
   pip install "matey @ git+https://github.com/FusionFM/MATEY.git@4e615bb5c86024632e386153bfbed028b38a8262"
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

## Files

| File | Description |
|---|---|
| `model.py` | `MATEYHarness` -- adapts MATEY models/data to BaseSim's `BaseModelHarness` interface |
| `matey.toml` | Experiment config |
| `pyproject.toml` | Optional example dependency manifest |
