---
name: Bug report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''

---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

Steps to reproduce the behavior:
1. Config used (paste the TOML, or link the example under `examples/`)
2. Command run (e.g. `poetry run python -m src.main --config examples/mnist/mnist.toml`)
3. Point in the run where it goes wrong (during monitoring, on drift detection, during the CL update, ...)
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happens instead. Include the traceback if there is one.

## Screenshots

If applicable, add screenshots to help explain your problem. W&B or MLflow charts are often the
clearest way to show a drift-detection or adaptation problem.

## Environment

Please complete the following information:
- OS: [e.g. Linux, Windows, macOS]
- Python Version: [e.g. 3.13]
- Apeiron Version: [e.g. 0.1.0, or the commit SHA]
- Accelerator: [e.g. NVIDIA A100 / AMD MI250X / CPU only]
- CUDA or ROCm Version: [e.g. CUDA 12.4, ROCm 6.2, N/A]
- PyTorch Version: [e.g. 2.5.1]
- HPC system, if applicable: [e.g. Frontier, Perlmutter]

## Additional Context

Add any other context about the problem here.
