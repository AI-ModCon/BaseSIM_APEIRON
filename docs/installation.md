# Installation

Apeiron requires **Python `>=3.13,<3.14`** and uses [Poetry](https://python-poetry.org/)
for dependency management.

## Developing inside this repository

Clone the repository and install the full environment, including the dev tools
(pytest, ruff, mypy):

```bash
git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
poetry install
```

Verify the install:

```bash
poetry run pytest -m "not slow"
poetry run python -c "import apeiron; print(apeiron.__doc__)"
```

## Using Apeiron as a dependency in your own project

The installable package lives under `src/apeiron/` and is imported as `apeiron`.

```toml
# pyproject.toml
[tool.poetry.dependencies]
apeiron = "^0.1.0"  # once published to PyPI

# Or as a path dependency during development
apeiron = { path = "../BaseSIM_APEIRON/", develop = true }

# Or straight from git
apeiron = { git = "https://github.com/AI-ModCon/BaseSIM_APEIRON.git", branch = "main" }
```

Then import the public API:

```python
from apeiron import BaseModelHarness, ContinuousMonitor, build_config
from apeiron.drift_detection import ADWINDetector
from apeiron.training.updater import BaseUpdater
```

See {doc}`api/index` for everything the package exports.

```{note}
PyTorch resolution differs between CPU-only and CUDA/ROCm machines. If Poetry
picks the wrong wheel, install the matching `torch` build first (following the
[PyTorch install matrix](https://pytorch.org/get-started/locally/)) and then run
`poetry install`. For HPC systems see {doc}`deployment`.
```

## Development commands

```bash
poetry run pytest                  # tests
poetry run ruff check .            # lint
poetry run ruff format --check .   # formatting
poetry run mypy .                  # type checks
```

## Building these docs locally

The docs are built with Sphinx and MyST-Markdown. Heavy runtime dependencies are
mocked, so a docs build does not need torch installed:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```
