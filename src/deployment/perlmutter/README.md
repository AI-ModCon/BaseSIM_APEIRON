# Deployment

## NERSC's Perlmutter

### Setup
First, create a local virtual enviroment in scratch directory and clone repo:

```bash
cd $SCRATCH # User scratch space
module load python # Load stable python
python -m venv my_env # Create a virtual environment
source ./my_env/bin/activate
pip install poetry
git clone https://github.com/AI-ModCon/BaseSim_Framework.git
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
