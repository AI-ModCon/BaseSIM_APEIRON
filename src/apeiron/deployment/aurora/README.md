# Deployment

## ALCF Aurora

### Setup

Clone the repo into your project directory on Flare and run the install script.
Do this on a login node; compute nodes have no outbound network.

```bash
cd /lus/flare/projects/<PROJECT>/$USER
git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
source ./src/apeiron/deployment/aurora/install_venv.sh
```

`install_venv.sh` loads the `frameworks` module (which provides [uv](https://docs.astral.sh/uv/)), fetches a Python 3.13 interpreter through uv, and creates a virtual environment with exactly the dependency versions pinned in `uv.lock`. The environment is saved to `.venv` in the project root. The script runs the following:

```bash
# Login nodes reach the internet only through the ALCF proxy.
if [[ "${HOSTNAME}" =~ aurora-uan ]]; then
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
fi

module use /soft/modulefiles
module load frameworks
command -v uv >/dev/null || python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# Keep uv's cache and interpreters off the $HOME quota.
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"

uv python install 3.13
uv sync --locked --python 3.13
source .venv/bin/activate
```

> **Note:** Aurora has no `python/3.13` module. uv downloads a standalone CPython 3.13 into `.uv-python` inside the project. The frameworks module's own Python is not used by the venv.

> **Note:** The MNIST example requires the dataset, which is downloaded on first run. Download it on a login node before submitting a batch job:
>
> ```bash
> uv run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
> ```

> **GPU note:** `uv.lock` resolves the PyPI build of `torch`, which is CPU-only on Aurora. Intel's XPU build ships only in the `frameworks` conda environment. The MNIST example runs on CPU as-is. For GPU runs, either build the venv on top of the frameworks Python (`uv venv --system-site-packages --python "$(command -v python)"`, which drops the 3.13 pin) or add Intel's XPU index to `pyproject.toml` and maintain an Aurora-specific lock.

### Submitting a Job

Aurora uses PBS Pro, not Slurm. Jobs are allocated whole nodes (6 GPUs, 12 tiles). The virtual environment is sourced directly at the top of the PBS script (`source .venv/bin/activate`), so uv is not needed at runtime.

From the project root:

```bash
mkdir -p output
qsub -A <PROJECT> src/apeiron/deployment/aurora/mnist_example.pbs
```

The script sets `ZE_AFFINITY_MASK=0.0` to expose a single GPU tile. Remove it for multi-GPU runs.

Useful commands: `qstat -u $USER` (job status), `qdel <jobid>` (cancel), `qstat -Q` (queue limits).

### Troubleshooting

- **`uv sync --locked` fails because the lockfile needs to be updated**: `pyproject.toml` was changed without re-locking. Don't re-lock on the cluster; pull the latest `uv.lock` from the repository, or run `uv lock` on your workstation and commit it.
- **Downloads hang or time out**: the proxy variables aren't set. Source the install script or export `https_proxy=http://proxy.alcf.anl.gov:3128` manually. Compute nodes have no network; do all downloads from a login node.
- **Disk quota errors in `$HOME`**: uv's cache defaults to `~/.cache/uv` and interpreters to `~/.local/share/uv`. The install script redirects both into the project directory on Flare. If you run `uv` commands outside the script, export `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` the same way.
- **Job held with a filesystem error**: the PBS script must declare every filesystem it touches via `-l filesystems=home:flare`. Add `daos_user_fs` if using DAOS.
- **`output/` files missing**: PBS drops `-o`/`-e` into the submit directory if the target directory doesn't exist. Create `output/` before submitting.
- **`uv: command not found`**: load `frameworks` (`module use /soft/modulefiles; module load frameworks`). If a frameworks release drops uv, the script falls back to `pip install --user uv`, which lands in `~/.local/bin`; make sure that's on your `PATH`.
