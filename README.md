# Continual Learning on MNIST

This project trains a convolutional neural network on a sequence of continually changing MNIST tasks. Each task is built by sampling one digit, applying random affine transformations, and training while replaying everything seen so far.

## Prerequisites
Install the Python packages listed in `requirements.txt` (Python 3.8+):
```bash
pip install -r requirements.txt
```
`torchvision` downloads MNIST to `CL_modcon/data/` the first time `main.py` runs.

## Running the Experiment
From the `CL_modcon` directory execute:
```bash
python ./cl_only.py --config examples/mnist/mnist.toml
```
The script uses CUDA automatically when it is available; otherwise it falls back to CPU.

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
