# Deployment

## NERSC Perlmutter

### Setup

Clone the repo into your scratch directory and run the install script:

```bash
cd $SCRATCH
git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
source ./src/apeiron/deployment/perlmutter/install_venv.sh
```

`install_venv.sh` installs [uv](https://docs.astral.sh/uv/) for your user if it isn't already available, then uses it to create a virtual environment with exactly the dependency versions pinned in `uv.lock`. The environment is saved to `.venv` in the project root. The script runs the following:

```bash
module load python/3.13-26.1.0

command -v uv >/dev/null || python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked --no-cache --python "$(command -v python)"
source .venv/bin/activate
```

> **Note:** The MNIST example requires to the dataset, which is downloaded on first run. Download it before submitting a batch job:
>
> ```bash
> uv run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
> ```

### Submitting a Job

The virtual environment can be sourced directly at the top of your SLURM script (`source .venv/bin/activate`), so uv is not needed at runtime — jobs run against the installed environment.

From the project root:

```bash
sbatch -A amsc002 src/apeiron/deployment/perlmutter/mnist_example.sbatch
```

### Troubleshooting

- **`uv sync --locked` fails because the lockfile needs to be updated** — `pyproject.toml` was changed without re-locking. Don't re-lock on the cluster; pull the latest `uv.lock` from the repository, or run `uv lock` on your workstation and commit it.
- **Disk quota errors in `$HOME`** — uv's cache defaults to `~/.cache/uv`. The install script already passes `--no-cache`; if you run `uv` commands yourself, add `--no-cache` or point `UV_CACHE_DIR` at scratch storage (for example `export UV_CACHE_DIR=$SCRATCH/.cache/uv`).
- **`uv: command not found`** — `pip install --user` puts uv in `~/.local/bin`; make sure it is on your `PATH`.
