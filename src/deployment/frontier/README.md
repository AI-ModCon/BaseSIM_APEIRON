# Deployment

## OLCF's Frontier

### Setup
First, create a local virtual enviroment in scratch directory and clone repo:

```bash
cd $MEMBERWORK # User scratch space
module load python # Load stable python
python -m venv my_env # Create a virtual environment
source ./my_env/bin/activate
pip install poetry
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
```

To install dependencies and torch libraries with ROCM support (6.4.2), run from the project root:

```bash
cd ./BaseSim_Framework
source ./src/deployment/frontier/install_rocm.sh
```

Prior to running experiments, test ROCM support from the project root:
> Pass project account via PROJECT_ACCOUNT
```bash
poetry run pytest tests/test_rocm.py
```


### Submit Job

> Note: Requires MNIST dataset download on first run.
> Download the dataset before submitting the run using:

```bash
poetry run python -c "from examples.mnist.utils import get_mnist_data; get_mnist_data()"
```

Submit run from project root:

```bash
SLURM_ACCOUNT=lrnxxx sbatch src/deployment/frontier/mnist_example.sbatch
```
