---
name: install-apeiron
description: |
  Install the apeiron continual-learning package as a dependency into an
  existing Python project so the user can `import apeiron`. Use when the user
  wants to add apeiron to their own project or training framework, set it up as
  a path/git dependency, or get `from apeiron import ...` working in another
  codebase. Handles Poetry presence, Python 3.13 verification, and automatic
  GPU-vs-CPU PyTorch selection. SKIP for developing inside THIS repo itself —
  that is just `poetry install`.
argument-hint: "[target_project_dir] [--git <url>]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Glob
  - Grep
---

Install apeiron as a dependency in the user's own Python project, hands-off.

## Arguments
- `$1`: Target project directory (the project that will depend on apeiron). Defaults to the current working directory.
- `--git <url>`: Optional. Install apeiron from this git URL instead of a local path. If omitted, prefer a local path dependency (see step 2).

Do not assume any value not given — read it from the repo or ask.

## Goal
After this skill runs, the following must succeed from inside the target project's environment:
```bash
python -c "from apeiron import BaseModelHarness, ContinuousMonitor, build_config; print('apeiron OK')"
```

## Procedure

### 1. Resolve the target project and its package manager
- Target dir = `$1` or the current working directory. Confirm it contains a `pyproject.toml` (Poetry/PEP 621) or `requirements.txt`/`setup.py` (pip). If none, ask the user how they manage dependencies.
- Detect the manager: Poetry if `[tool.poetry]` or `poetry.lock` is present; otherwise pip/uv. Poetry is the primary path below; a pip fallback is in step 6.

### 2. Resolve the apeiron source (do not hardcode versions or paths)
- If `--git <url>` was given, use it as a git dependency.
- Else look for a local apeiron checkout. The current repo IS apeiron when its `pyproject.toml` has `name = "apeiron"`. Confirm with:
  ```bash
  grep -m1 'name = "apeiron"' pyproject.toml && pwd
  ```
  Use that absolute path as a **path (develop) dependency**. If the current repo is not apeiron and no `--git` was given, ask the user for the apeiron path or git URL.

### 3. Verify Python (guide, don't auto-manage interpreters)
- Read apeiron's required range dynamically rather than assuming it:
  ```bash
  grep 'requires-python' <apeiron_pyproject>
  ```
- Check the interpreter the target project will use (`python --version`, or `poetry env info --python`). If it is outside the range, stop and give the user exact instructions (e.g. install the matching CPython via pyenv/uv and point Poetry at it with `poetry env use <path>`). Do not silently install or switch interpreters.

### 4. Ensure Poetry is available (auto-install if missing)
- `command -v poetry` — if missing, install it: `pipx install poetry` (preferred) or `pip install --user poetry`, then re-check `poetry --version`.

### 5. Detect compute backend and select the PyTorch wheel
- Probe for an NVIDIA GPU:
  ```bash
  nvidia-smi -L 2>/dev/null && echo "GPU_PRESENT" || echo "NO_GPU"
  ```
- **GPU present:** do nothing special — the default CUDA-enabled torch wheels resolve normally. Report that CUDA wheels will be used.
- **No GPU:** pin torch to the CPU-only index so the install is smaller and portable. For a Poetry target, add an explicit source and route torch to it before adding apeiron:
  ```toml
  [[tool.poetry.source]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  priority = "explicit"

  [tool.poetry.dependencies]
  torch = { source = "pytorch-cpu" }
  ```
  Then `poetry lock`. Report that CPU-only wheels will be used.

### 6. Add the dependency
From the target project directory:
- **Poetry, local path:** `poetry add --editable <absolute_apeiron_path>`
- **Poetry, git:** `poetry add "git+<url>"`
- **pip/uv fallback, local path:** `pip install -e <absolute_apeiron_path>` (for the no-GPU case, first run `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- **pip/uv fallback, git:** `pip install "apeiron @ git+<url>"`

### 7. Verify and report
- Run the import check from step **Goal** inside the target environment (`poetry run python -c ...` for Poetry).
- On success, report: the apeiron source used (path/git), Python version, compute backend chosen (CUDA/CPU), and the manager the dependency was added to.
- Suggest next steps: `/run-experiment` to try a bundled example, or the integration skill if they are wiring apeiron into an existing training loop.

## Troubleshooting
- **`ModuleNotFoundError: apeiron`** after install — the editable/path link didn't register; re-run the add in the *target* project dir, not the apeiron repo.
- **torch pulls CUDA wheels on a CPU box** — the explicit `pytorch-cpu` source from step 5 was not applied before `poetry lock`; re-lock after adding it.
- **Python version conflict** — apeiron pins a narrow CPython range (see step 3); the target project must use a matching interpreter. Point Poetry at it with `poetry env use`.
