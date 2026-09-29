---
name: install-apeiron
description: Install the apeiron continual-learning package into an existing Python project so `import apeiron` works. Use when the user wants to add apeiron to another project or training framework, set it up as a path or git dependency, or fix imports in an external codebase. Handles package-manager detection, Python compatibility checks, and CPU vs CUDA PyTorch selection. Do not use for developing inside the apeiron repo itself; that is just `uv sync`.
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
  - uv when `uv.lock` or a `[tool.uv]` table is present.
  - Poetry when `[tool.poetry]` or `poetry.lock` is present.
  - Otherwise use the existing pip workflow.
- If no dependency-management files are present, ask the user how dependencies are managed.

### 2. Resolve The Apeiron Source

- Never install apeiron from PyPI (`pip install apeiron`, `uv add apeiron`, `poetry add apeiron`). The `apeiron` name on PyPI belongs to an unrelated project.
- If the user provided a git URL, use it as the dependency source.
- Otherwise prefer a local checkout.
- The current repo is apeiron when its `pyproject.toml` identifies the package as `apeiron`. Confirm this from the file before using the current repo path.
- If no local checkout can be found and no git URL was provided, ask for the apeiron path or git URL.

### 3. Check Python Compatibility

- Read apeiron's Python requirement from its `pyproject.toml`; do not hardcode it.
- Check the target project's interpreter with `uv run python --version` (uv), `poetry env info --python` (Poetry), or `python --version`.
- If the interpreter is outside apeiron's required range, stop and give exact remediation steps: for uv, `uv python install 3.13` and `uv python pin 3.13`; for Poetry, install a matching CPython and point Poetry at it with `poetry env use <path>`.
- Do not silently install or switch interpreters.

### 4. Ensure The Manager When Needed

- For uv targets, check `command -v uv`; for Poetry targets, check `command -v poetry`.
- If the manager is missing and installing it is necessary, request permission before running package-install commands such as `pipx install uv`, `pip install --user uv`, `pipx install poetry`, or `pip install --user poetry`.
- Re-check `uv --version` or `poetry --version` before continuing.

### 5. Select The PyTorch Backend

- Probe for an NVIDIA GPU with `nvidia-smi -L` when available.
- If a GPU is present, use the default torch resolution and report that CUDA-capable wheels will be used.
- If no GPU is present, prefer CPU-only PyTorch wheels. Configure this before adding apeiron.
- For uv targets, add an explicit index and route torch and torchvision to it:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

Then run `uv add torch torchvision` so both are direct dependencies. uv applies `[tool.uv.sources]` only to a project's direct dependencies; a torch reached only through apeiron would still come from PyPI with the CUDA stack. Keep `explicit = true`, or uv consults the PyTorch index first for every package and fails on the old versions it mirrors.

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

- uv local path: `uv add --editable <absolute_apeiron_path>`
- uv git URL: `uv add "git+<url>"` (add `--tag <tag>` or `--branch <branch>` to pin a ref)
- Poetry local path: `poetry add --editable <absolute_apeiron_path>`
- Poetry git URL: `poetry add "git+<url>"`
- pip local path: `pip install -e <absolute_apeiron_path>`
- pip git URL: `pip install "apeiron @ git+<url>"`

For a CPU-only pip install, install torch from `https://download.pytorch.org/whl/cpu` before installing apeiron.

### 7. Verify And Report

- Run the import check from the success criteria inside the target environment.
- For uv targets, use `uv run python -c ...`; for Poetry targets, use `poetry run python -c ...`.
- Report:
  - apeiron source used, path or git URL
  - target Python version
  - package manager used
  - compute backend selected, CPU or CUDA
- Suggest `explore-examples` for a bundled demo or `integrate-apeiron` for wiring apeiron into an existing training loop.

## Troubleshooting

- `ModuleNotFoundError: apeiron`: re-run the dependency add from the target project directory.
- An unrelated `apeiron` package was installed: it came from PyPI. Uninstall it and re-add apeiron from its git URL or local path.
- CUDA wheels on a CPU machine: make sure the CPU-only torch source was added before locking or installing. For uv, torch and torchvision must also be direct dependencies of the target project.
- `invalid peer certificate: UnknownIssuer` from uv: the network intercepts TLS with a certificate uv's bundled roots don't trust. Re-run with `--system-certs`, or set `UV_SYSTEM_CERTS=1`.
- Python version conflict: point the target environment at a compatible interpreter instead of changing apeiron's requirement.
