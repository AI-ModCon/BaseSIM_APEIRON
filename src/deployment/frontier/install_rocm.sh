#!/bin/bash

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.4.2

poetry lock
poetry install
poetry run pip install --force-reinstall \
    torch==2.9.1+rocm6.4 \
    torchvision==0.24.1+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4
