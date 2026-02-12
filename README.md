# SIM: Self Improving Model framework
[![Build Status](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml/badge.svg)](https://github.com/AI-ModCon/BaseSim_Framework/actions/workflows/build-test.yml)
[![Coverage Status](https://codecov.io/gh/AI-ModCon/BaseSim_Framework/badge.svg?branch=main)](https://codecov.io/gh/AI-ModCon/BaseSim_Framework?branch=main)

A PyTorch framework for continuous learning that automatically detects concept drift in data streams and adapts models through JVP regularized retraining.

## Overview
This repo trains models on continuously changing data streams. The system monitors model performance in real-time, detects when the data distribution (concept drift) shifts, and automatically triggers adaptive learning to maintain performance without catastrophic forgetting.

Concept drift occurs when the statistical relationship between inputs and outputs changes overtime, causing model performance to degrade. This framework detects drift by monitoring performance metrics (like accuracy or loss) and applies JVP-regularized learning only when needed.

## Adaptive Learning 
- Evaluates model performance on incoming batches, tracking metrics in real-time.
- Monitors metrics and identifies when data distribution shift significantly.
- When drift is detected, monitoring pauses, JVP regularization prevent catastrophic forgetting, and model updates balance new patterns and old knowledge.
- Monitoring resumes with updated model weights.

## Prerequisites
This project uses [Poetry](https://python-poetry.org/) for dependency management. You will need to have Poetry installed.

Install the project dependencies with:
```bash
poetry install
```
`torchvision` downloads MNIST to `data/` the first time the experiment is run.

## Running the Experiment
To run the experiment, execute the following command from the project root:
```bash
poetry run python -m src.main --config examples/mnist/mnist.toml
poetry run python -m src.main --config examples/cifar10/cifar10_vit.toml
poetry run python -m src.main --config examples/imagenet/imagenet_vit.toml
```
The script uses CUDA automatically when it is available; otherwise it falls back to CPU.

## Visualizing Performance
To visualize the training and testing continous learning metrics, execute the following command from the project root:
```bash
poetry run python -m src.visualize --config examples/mnist/mnist.toml
```

## Running Tests
To run the project's tests, execute the following command from the project root:
```bash
poetry run pytest
```

## What `main.py` Does
- Builds the `DummyCNN_MNIST` model defined in `src/model/DummyCNN_MNIST.py`, a cross-entropy loss, and an Adam optimizer.
- Loads the MNIST training split, stacks the tensors, and iterates over 10 tasks (digits 0–9). Each task applies random rotation and translation to encourage continual adaptation.
- Maintains replay buffers (`memory_image`, `memory_label`, etc.) so past samples remain available for rehearsal while training new tasks.
- Calls `CL(...)` to assemble task-specific dataloaders and drive the `One_task_CL` loop. The loop trains for five epochs, records loss/accuracy metrics, and prints periodic progress reports.
- Computes sensitivity scores with `src/validation/validation_utils/return_score` after each task; you can repurpose these values for analysis or adaptive triggers.

## Tuning Tips
- Change the number of epochs by editing `n_epoch` inside `CL`.
- Adjust replay/adversarial update counts through the `params` dictionaries in `One_task_CL` and `util.update_CL_`.
- Experiment with different transforms or task definitions by modifying `data.py`.
- Update batch sizes by changing the `batch_size` parameter used when constructing the dataloaders.

## Output
Training logs report the task id, training/test accuracy, and replay-memory accuracy every five epochs. Accuracy is computed via `test(...)` on both the current task and the accumulated memory set.
