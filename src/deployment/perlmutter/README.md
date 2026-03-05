# Deployment

## NERSC's Perlmutter

### Setup
First, within your scratch directory, the clone the repo, and run install script.:

```bash
cd $SCRATCH # User scratch space
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
cd ./BaseSim_Framework
source ./src/deployment/perlmutter/install_venv.sh
```

`install_venv.sh` creates a virtual enviroment, installs poetry, then poetry installs
the project dependencies. The virtual enviroment is saved under `.venv` in the root directory.

The following commands are run in `install_venv.sh`:
```bash
module load python/3.13-26.1.0 # Load supported python version
python -m venv .venv # Create a virtual environment
source ./venv/bin/activate
pip install poetry
poetry lock
poetry install --no_cache
```

> Note: Testing model harness and jvp update requires MNIST dataset download on first run.
> Download the dataset before submitting the run using:

```bash
poetry run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
```

### Submit Job
To submit run from project root:

```bash
sbatch -A amsc002 src/deployment/perlmutter/mnist_example.sbatch
```

### Common Issues

1. If running `poetry install` produces errors connecting to PyPi, run `poetry lock` then retry `poetry install`.
Poetry's lock file contains the packages download spec. It may be stale and needs an update for the new host.

2. If running `poetry install` produces errors regarding disk space quota limits, there is not enough space in
poetry's default cache location (home directory). Retry with `poetry install --no-cache` or free up space in
home directory.
