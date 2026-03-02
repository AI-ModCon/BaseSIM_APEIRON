# MATEY Example Harness

BaseSim harness for the [MATEY](https://code.ornl.gov/matey/matey) multiscale transformer codebase.

## Setup

1. Install BaseSim (from repo root):
   ```bash
   poetry install
   ```

2. Install MATEY dependencies:
   ```bash
   pip install -e examples/matey/
   ```

3. (Optional) Install heavy/system-dependent packages as needed:
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

The `[data].path` should point to your local MATEY checkout (default: `examples/matey/MATEY`).

For the SOLPS example, BaseSim treats `MATEY/` as a read-only third-party checkout.
The harness collects SOLPS files from configured `train_data_paths` and
`valid_data_paths`, applies a deterministic file-level split of `[0.7, 0.15, 0.15]`,
and materializes staged views under:

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
| `pyproject.toml` | Dependency manifest (decoupled from BaseSim core) |
| `MATEY/` | MATEY source checkout |
