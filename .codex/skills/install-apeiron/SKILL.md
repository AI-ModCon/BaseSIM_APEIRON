---
name: install-apeiron
description: Install the apeiron continual-learning package into an existing Python project so `import apeiron` works. Use when the user wants to add apeiron to another project or training framework, set it up as a path or git dependency, or fix imports in an external codebase. Handles package-manager detection, Python compatibility checks, and CPU vs CUDA PyTorch selection. Do not use for developing inside the apeiron repo itself; that path is just the repo's normal development install.
metadata:
  short-description: Install apeiron into another Python project
---

# Install Apeiron

Install apeiron as a dependency in the user's own Python project.

## Inputs

- Target project directory: use the path the user gives, otherwise the current working directory.
- Optional git URL: when the user asks for a git dependency, use that URL instead of a local path dependency.

Do not assume missing values. Read them from the repo when possible, and ask only when the source or target cannot be discovered safely.

## Success Criteria

From inside the target project's environment, this command must pass:

```bash
python -c "from apeiron import BaseModelHarness, ContinuousMonitor, build_config; print('apeiron OK')"
```

## Procedure

### 1. Resolve The Target

- Confirm the target directory contains `pyproject.toml`, `requirements.txt`, or `setup.py`.
- Detect the manager:
  - Poetry when `[tool.poetry]` or `poetry.lock` is present.
  - Otherwise use the existing pip or uv workflow.
- If no dependency-management files are present, ask the user how dependencies are managed.

### 2. Resolve The Apeiron Source

- If the user provided a git URL, use it as the dependency source.
- Otherwise prefer a local checkout.
- The current repo is apeiron when its `pyproject.toml` identifies the package as `apeiron`. Confirm this from the file before using the current repo path.
- If no local checkout can be found and no git URL was provided, ask for the apeiron path or git URL.

### 3. Check Python Compatibility

- Read apeiron's Python requirement from its `pyproject.toml`; do not hardcode it.
- Check the target project's interpreter with `python --version` or `poetry env info --python`.
- If the interpreter is outside apeiron's required range, stop and give exact remediation steps, such as installing a matching CPython and pointing Poetry at it with `poetry env use <path>`.
- Do not silently install or switch interpreters.

### 4. Ensure Poetry When Needed

- For Poetry targets, check `command -v poetry`.
- If Poetry is missing and installing it is necessary, request permission before running package-install commands such as `pipx install poetry` or `pip install --user poetry`.
- Re-check `poetry --version` before continuing.

### 5. Select The PyTorch Backend

- Probe for an NVIDIA GPU with `nvidia-smi -L` when available.
- If a GPU is present, use the default torch resolution and report that CUDA-capable wheels will be used.
- If no GPU is present, prefer CPU-only PyTorch wheels.
- For Poetry targets, add or preserve an explicit PyTorch CPU source before locking:

```toml
[[tool.poetry.source]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"

[tool.poetry.dependencies]
torch = { source = "pytorch-cpu" }
```

Then run `poetry lock`.

### 6. Add The Dependency

From the target project directory:

- Poetry local path: `poetry add --editable <absolute_apeiron_path>`
- Poetry git URL: `poetry add "git+<url>"`
- pip or uv local path: `pip install -e <absolute_apeiron_path>`
- pip or uv git URL: `pip install "apeiron @ git+<url>"`

For a CPU-only pip install, install torch from `https://download.pytorch.org/whl/cpu` before installing apeiron.

### 7. Verify And Report

- Run the import check from the success criteria inside the target environment.
- For Poetry targets, use `poetry run python -c ...`.
- Report:
  - apeiron source used, path or git URL
  - target Python version
  - package manager used
  - compute backend selected, CPU or CUDA
- Suggest `explore-examples` for a bundled demo or `integrate-apeiron` for wiring apeiron into an existing training loop.

## Troubleshooting

- `ModuleNotFoundError: apeiron`: re-run the dependency add from the target project directory.
- CUDA wheels on a CPU machine: make sure the CPU-only torch source was added before locking or installing.
- Python version conflict: point the target environment at a compatible interpreter instead of changing apeiron's requirement.
