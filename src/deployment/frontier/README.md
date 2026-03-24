# Deployment

## OLCF Froniter

### Setup

Clone the repo into your scratch directory and run the install script:

```bash
cd $MEMBERWORK
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
cd BaseSim_Framework
source ./src/deployment/frontier/install_venv.sh
```

`install_venv.sh` creates a virtual environment, installs Poetry, and uses it to resolve and install project dependencies. The environment is saved to `.venv` in the project root. The script runs the following:

```bash
module load PrgEnv-gnu
module load python/3.13.0
module load gcc/12.2.0
module load rocm/6.4.2

python -m venv .venv # Create a virtual environment
source ./.venv/bin/activate # Activate environment
pip install poetry # Install poetry
poetry lock # Sync poetry
poetry install --no-cache # Install poetry

poetry run pip install --force-reinstall \
     torch==2.9.1+rocm6.4 \
     torchvision==0.24.1+rocm6.4 \
     --index-url https://download.pytorch.org/whl/rocm6.4
```

Prior to running experiments, test ROCM support from the project root:
```bash
poetry run pytest tests/test_rocm.py
```

### Submitting a Job

> **Note:** The MNIST example requires to the dataset, which is downloaded on first run. Download it before submitting a batch job:
>
> ```bash
> poetry run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
> ```

The virtual environment can be sourced directly at the top of your SLURM script (`source .venv/bin/activate`), so Poetry is not needed at runtime — jobs run against the installed environment.

From the project root:

```bash
sbatch -A xxx src/deployment/frontier/mnist_example.sbatch
```

### Troubleshooting

- **`poetry install` fails to connect to PyPI** — Run `poetry lock` first, then retry. The lock file caches package download specs and may be stale on a new host.
- **`poetry install` fails with disk quota errors** — Poetry's default cache is in the home directory, which has limited space. Retry with `poetry install --no-cache` or free up space in `$HOME`.
