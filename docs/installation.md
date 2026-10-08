# Installation

Apeiron requires **Python `>=3.13,<3.14`** and uses [uv](https://docs.astral.sh/uv/)
for dependency management. See the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to
install it.

## Developing inside this repository

Clone the repository and create the environment, including the dev tools
(pytest, ruff, mypy):

```bash
git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
uv sync
```

`uv sync` creates `.venv` with exactly the versions pinned in `uv.lock`,
downloading a matching Python first if you don't have one. Verify the install:

```bash
uv run pytest -m "not slow"
uv run python -c "import apeiron; print(apeiron.__doc__)"
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit the
updated `uv.lock`. CI fails if the lock is out of date.

## Using Apeiron as a dependency in your own project

The installable package lives under `src/apeiron/` and is imported as `apeiron`.
Install it from GitHub:

```{warning}
Don't run `pip install apeiron` or `uv add apeiron`. The `apeiron` name on PyPI
belongs to an unrelated project.
```

```bash
# A uv project, pinned to a release tag
uv add "git+https://github.com/AI-ModCon/BaseSIM_APEIRON" --tag v0.1.0

# Or an editable local checkout, while developing both side by side
uv add --editable ../BaseSIM_APEIRON

# Or with pip
pip install "apeiron @ git+https://github.com/AI-ModCon/BaseSIM_APEIRON@v0.1.0"
```

Then import the public API:

```python
from apeiron import BaseModelHarness, ContinuousMonitor, build_config
from apeiron.drift_detection import ADWINDetector
from apeiron.training.updater import BaseUpdater
```

See {doc}`api/index` for everything the package exports.

```{note}
On Linux, `uv sync` installs the CUDA build of PyTorch from PyPI. For a CPU-only
or ROCm machine, install the matching build over it with `uv pip install`,
following the [PyTorch install matrix](https://pytorch.org/get-started/locally/).
Afterwards, run commands in the activated environment (`source .venv/bin/activate`)
or with `uv run --no-sync`: plain `uv run` and `uv sync` re-sync `.venv` to
`uv.lock`, which puts the default build back.

To choose a PyTorch build in your own project instead, see uv's
[PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
For HPC systems see {doc}`deployment`.
```

## Development commands

```bash
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # type checks
```

The same checks are wrapped in the `Makefile`; run `make help` to list them.

## Building these docs locally

The docs are built with Sphinx and MyST-Markdown. Heavy runtime dependencies are
mocked, so a docs build does not need torch installed:

```bash
make docs
open docs/_build/html/index.html
```

`make docs` runs Sphinx in a throwaway environment built from
`docs/requirements.txt`, without installing the project. Without uv, the
equivalent is:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
