#!/bin/bash

module load PrgEnv-gnu
module load python/3.13.0
module load gcc/12.2.0
module load rocm/7.1.0 # torch's ROCm wheels need ROCm 7.1; check `module avail rocm`

command -v uv >/dev/null || python -m pip install --user uv # Install uv into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Create .venv with exactly the versions in uv.lock, using the module's Python.
# --no-cache keeps uv's download cache out of the limited $HOME quota.
uv sync --locked --no-cache --python "$(command -v python)"
source ./.venv/bin/activate # Activate environment

# Swap in the ROCm builds of torch and torchvision at the versions pinned in
# uv.lock. After this, don't use `uv run` or `uv sync`: both re-sync .venv to
# uv.lock and put the CUDA builds back. Use the activated environment instead.
ROCM=7.1
TORCH=$(uv export --locked --no-hashes --no-annotate --no-header --no-emit-project | sed -n 's/^torch==\([^ ;]*\).*/\1/p')
TORCHVISION=$(uv export --locked --no-hashes --no-annotate --no-header --no-emit-project | sed -n 's/^torchvision==\([^ ;]*\).*/\1/p')
uv pip install --no-cache \
    "torch==${TORCH}+rocm${ROCM}" \
    "torchvision==${TORCHVISION}+rocm${ROCM}" \
    --index-url "https://download.pytorch.org/whl/rocm${ROCM}"
unset ROCM TORCH TORCHVISION
