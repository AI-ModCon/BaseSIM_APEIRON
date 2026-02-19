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
   MAX_JOBS=4 pip install flash-attn --no-build-isolation  # requires CUDA toolkit + nvcc
   pip install dadaptation==3.1                             # for DAdaptAdam optimizer
   pip install mpi4py                                       # requires MPI C library
   pip install netCDF4                                      # requires HDF5/netCDF C libs
   pip install git+https://github.com/sandialabs/exodusii.git # not on PyPI
   ```

## Running

```bash
poetry run python -m src.main --config examples/matey/matey.toml
```

## Configuration

Edit [matey.toml](matey.toml) to adjust training parameters, drift detection, and data paths.

The `[data].path` should point to your local MATEY checkout (default: `examples/matey/MATEY`).

## Files

| File | Description |
|---|---|
| `model.py` | `MATEYHarness` -- adapts MATEY models/data to BaseSim's `BaseModelHarness` interface |
| `matey.toml` | Experiment config |
| `pyproject.toml` | Dependency manifest (decoupled from BaseSim core) |
| `MATEY/` | MATEY source checkout |
