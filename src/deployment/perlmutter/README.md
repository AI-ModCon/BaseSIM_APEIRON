# Deployment

## NERSC Perlmutter

### Setup

Clone the repo into your scratch directory and run the install script:

```bash
cd $SCRATCH
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
cd BaseSim_Framework
source ./src/deployment/perlmutter/install_venv.sh
```

`install_venv.sh` creates a virtual environment, installs Poetry, and uses it to resolve and install project dependencies. The environment is saved to `.venv` in the project root. The script runs the following:

```bash
module load python/3.13-26.1.0
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry lock
poetry install --no-cache
```

> **Note:** The MNIST example requires to the dataset, which is downloaded on first run. Download it before submitting a batch job:
>
> ```bash
> poetry run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
> ```

### Submitting a Job

The virtual environment can be sourced directly at the top of your SLURM script (`source .venv/bin/activate`), so Poetry is not needed at runtime — jobs run against the installed environment.

From the project root:

```bash
sbatch -A amsc002 src/deployment/perlmutter/mnist_example.sbatch
```

### Troubleshooting

- **`poetry install` fails to connect to PyPI** — Run `poetry lock` first, then retry. The lock file caches package download specs and may be stale on a new host.
- **`poetry install` fails with disk quota errors** — Poetry's default cache is in the home directory, which has limited space. Retry with `poetry install --no-cache` or free up space in `$HOME`.
