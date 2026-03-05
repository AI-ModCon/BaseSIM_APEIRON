# Deployment

## NERSC's Perlmutter

### Setup
First, create a local virtual enviroment in scratch directory and clone repo:

```bash
cd $SCRATCH # User scratch space
module load python/3.13-26.1.0 # Load stable python
python -m venv my_env # Create a virtual environment
source ./my_env/bin/activate
pip install poetry
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
cd ./BaseSim_Framework
poetry install
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
