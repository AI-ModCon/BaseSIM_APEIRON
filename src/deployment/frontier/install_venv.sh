#!/bin/bash

module load PrgEnv-gnu
module load python/3.13.0
module load gcc/12.2.0
module load rocm/6.4.2

python -m venv .venv # Create a virtual environment
source ./.venv/bin/activate # Activate environment
pip install poetry # Install poetry
poetry lock # Sync poetry
poetry install --no-cache # Install poetry

poetry run pip install --force-reinstall \
    torch==2.9.1+rocm6.4 \
    torchvision==0.24.1+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4
