---
name: install-apeiron
description: |
  Install the apeiron continual-learning package as a dependency into an
  existing Python project so the user can `import apeiron`. Use when the user
  wants to add apeiron to their own project or training framework, set it up as
  a path/git dependency, or get `from apeiron import ...` working in another
  codebase. Handles uv/Poetry/pip targets, Python 3.13 verification, and
  automatic GPU-vs-CPU PyTorch selection. SKIP for developing inside THIS repo
  itself — that is just `uv sync`.
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
- Target dir = `$1` or the current working directory. Confirm it contains a `pyproject.toml` or `requirements.txt`/`setup.py`. If none, ask the user how they manage dependencies.
- Detect the manager:
  - **uv** if `uv.lock` or a `[tool.uv]` table is present.
  - **Poetry** if `[tool.poetry]` or `poetry.lock` is present.
  - **pip** otherwise.
- uv is the primary path below; Poetry and pip paths follow in each step.

### 2. Resolve the apeiron source (do not hardcode versions or paths)
- **Never install apeiron from PyPI** (`pip install apeiron`, `uv add apeiron`, `poetry add apeiron`). The `apeiron` name on PyPI belongs to an unrelated project.
- If `--git <url>` was given, use it as a git dependency.
- Else look for a local apeiron checkout. The current repo IS apeiron when its `pyproject.toml` has `name = "apeiron"`. Confirm with:
  ```bash
  grep -m1 'name = "apeiron"' pyproject.toml && pwd
  ```
  Use that absolute path as an **editable path dependency**. If the current repo is not apeiron and no `--git` was given, ask the user for the apeiron path or git URL.

### 3. Verify Python (guide, don't auto-manage interpreters)
- Read apeiron's required range dynamically rather than assuming it:
  ```bash
  grep 'requires-python' <apeiron_pyproject>
  ```
- Check the interpreter the target project will use (`uv run python --version` for uv, `poetry env info --python` for Poetry, `python --version` otherwise). If it is outside the range, stop and give the user exact instructions. For uv: `uv python install 3.13` then `uv python pin 3.13`, and widen the target's `requires-python` if it excludes 3.13. For Poetry: install a matching CPython and point Poetry at it with `poetry env use <path>`. Do not silently install or switch interpreters.

### 4. Ensure the manager is available (auto-install if missing)
- **uv:** `command -v uv` — if missing, install it with `pipx install uv` (preferred) or `pip install --user uv`, then re-check `uv --version`.
- **Poetry:** `command -v poetry` — if missing, `pipx install poetry` (preferred) or `pip install --user poetry`, then re-check `poetry --version`.

### 5. Detect compute backend and select the PyTorch wheel
- Probe for an NVIDIA GPU:
  ```bash
  nvidia-smi -L 2>/dev/null && echo "GPU_PRESENT" || echo "NO_GPU"
  ```
- **GPU present:** do nothing special — the default CUDA-enabled torch wheels resolve normally. Report that CUDA wheels will be used.
- **No GPU:** route torch to the CPU-only index so the install is smaller and portable. Do this **before** adding apeiron.
  - **uv target:** add an *explicit* index and point torch and torchvision at it:
    ```toml
    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true

    [tool.uv.sources]
    torch = [{ index = "pytorch-cpu" }]
    torchvision = [{ index = "pytorch-cpu" }]
    ```
    Then make them **direct** dependencies with `uv add torch torchvision`. uv only applies `[tool.uv.sources]` to the project's own direct dependencies, so a torch pulled in only through apeiron would still come from PyPI with the full CUDA stack. Keep `explicit = true`; without it uv consults the PyTorch index first for *every* package and fails on the old versions it mirrors.
  - **Poetry target:**
    ```toml
    [[tool.poetry.source]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    priority = "explicit"

    [tool.poetry.dependencies]
    torch = { source = "pytorch-cpu" }
    ```
    Then `poetry lock`.
  - Report that CPU-only wheels will be used.

### 6. Add the dependency
From the target project directory:
- **uv, local path:** `uv add --editable <absolute_apeiron_path>`
- **uv, git:** `uv add "git+<url>"` (add `--tag <tag>` or `--branch <branch>` to pin a ref)
- **Poetry, local path:** `poetry add --editable <absolute_apeiron_path>`
- **Poetry, git:** `poetry add "git+<url>"`
- **pip, local path:** `pip install -e <absolute_apeiron_path>` (for the no-GPU case, first run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)
- **pip, git:** `pip install "apeiron @ git+<url>"`

### 7. Verify and report
- Run the import check from step **Goal** inside the target environment (`uv run python -c ...` for uv, `poetry run python -c ...` for Poetry).
- On success, report: the apeiron source used (path/git), Python version, compute backend chosen (CUDA/CPU), and the manager the dependency was added to.
- Suggest next steps: `/run-experiment` to try a bundled example, or the integration skill if they are wiring apeiron into an existing training loop.

## Troubleshooting
- **`ModuleNotFoundError: apeiron`** after install — the editable/path link didn't register; re-run the add in the *target* project dir, not the apeiron repo.
- **An unrelated `apeiron` package was installed** — it came from PyPI. Uninstall it and re-add apeiron from its git URL or local path (step 6).
- **torch pulls CUDA wheels on a CPU box** — for uv, check that torch and torchvision are *direct* dependencies of the target project and that `[tool.uv.sources]` routes both to the CPU index; for Poetry, the explicit `pytorch-cpu` source was not applied before `poetry lock`. Re-lock after fixing.
- **`invalid peer certificate: UnknownIssuer`** from uv — the network intercepts TLS with a certificate uv's bundled roots don't trust. Re-run with `--system-certs`, or `export UV_SYSTEM_CERTS=1`.
- **Python version conflict** — apeiron pins a narrow CPython range (see step 3); the target project must use a matching interpreter.
