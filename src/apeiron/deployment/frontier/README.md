# Deployment

## OLCF Frontier

### Setup

Clone the repo into your scratch directory and run the install script:

```bash
cd $MEMBERWORK
git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
source ./src/apeiron/deployment/frontier/install_venv.sh
```

`install_venv.sh` installs [uv](https://docs.astral.sh/uv/) for your user if it isn't already available, then uses it to create a virtual environment with exactly the dependency versions pinned in `uv.lock`. It then replaces the default CUDA builds of `torch` and `torchvision` with the ROCm builds of the same versions, read from `uv.lock`. The environment is saved to `.venv` in the project root and left activated. The script runs the following:

```bash
module load PrgEnv-gnu
module load python/3.13.0
module load gcc/12.2.0
module load rocm/7.1.0

command -v uv >/dev/null || python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked --no-cache --python "$(command -v python)"
source ./.venv/bin/activate

ROCM=7.1
TORCH=$(uv export --locked --no-hashes --no-annotate --no-header --no-emit-project | sed -n 's/^torch==\([^ ;]*\).*/\1/p')
TORCHVISION=$(uv export --locked --no-hashes --no-annotate --no-header --no-emit-project | sed -n 's/^torchvision==\([^ ;]*\).*/\1/p')
uv pip install --no-cache \
    "torch==${TORCH}+rocm${ROCM}" \
    "torchvision==${TORCHVISION}+rocm${ROCM}" \
    --index-url "https://download.pytorch.org/whl/rocm${ROCM}"
```

> **Warning:** After the install, don't run `uv run` or `uv sync` in this environment. Both re-sync `.venv` to `uv.lock` and replace the ROCm builds of torch with the CUDA builds. Use the activated environment (`source .venv/bin/activate`) instead, or `uv run --no-sync`. To pick up a dependency change, re-run `install_venv.sh`.

Prior to running experiments, test ROCM support from the project root, in the activated environment:
```bash
pytest tests/test_rocm.py
```

### Submitting a Job

> **Note:** The MNIST example requires to the dataset, which is downloaded on first run. Download it before submitting a batch job:
>
> ```bash
> python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
> ```

The virtual environment can be sourced directly at the top of your SLURM script (`source .venv/bin/activate`), so uv is not needed at runtime — jobs run against the installed environment.

From the project root:

```bash
sbatch -A xxx src/apeiron/deployment/frontier/mnist_example.sbatch
```

### Troubleshooting

- **`module load rocm/7.1.0` fails** — the ROCm 7.1 module on Frontier may use a different patch version. Run `module avail rocm` and load the 7.1.x module it lists. The PyTorch ROCm wheels for the locked `torch` need ROCm 7.1.
- **`test_rocm.py` fails or `torch.version.hip` is `None`** — the CUDA build of torch is installed, usually because `uv run` or `uv sync` was used after the install. Re-run `install_venv.sh`.
- **`uv sync --locked` fails because the lockfile needs to be updated** — `pyproject.toml` was changed without re-locking. Don't re-lock on the cluster; pull the latest `uv.lock` from the repository, or run `uv lock` on your workstation and commit it.
- **Disk quota errors in `$HOME`** — uv's cache defaults to `~/.cache/uv`. The install script already passes `--no-cache`; if you run `uv` commands yourself, add `--no-cache` or point `UV_CACHE_DIR` at project storage.
- **`invalid peer certificate: UnknownIssuer`** — the network intercepts TLS with a certificate uv's bundled roots don't trust. Set `export UV_SYSTEM_CERTS=1` and re-run the script.
- **`uv: command not found`** — `pip install --user` puts uv in `~/.local/bin`; make sure it is on your `PATH`.
