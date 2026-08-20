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
mkdir -p output
sbatch -A amsc002 src/apeiron/deployment/perlmutter/mnist_example.sbatch
```

### Well scaling benchmark

The Well example (`examples/well/`) is the data-parallel scaling benchmark; the
full recipe is in `examples/well/BENCHMARK.md`. On Perlmutter:

1. **Build the WindowStore on a login node** — compute nodes have no internet, so
   the HuggingFace download must happen before you submit any job (this is the
   Well analog of the MNIST download note above):

   ```bash
   poetry run python -m examples.well.convert --dataset turbulent_radiative_layer_2D \
       --split train --max-files 8 --out $SCRATCH/wellstore --window-steps 24
   ```

2. **Throughput / scaling** — submit `well_benchmark.sbatch` (4 GPUs = 4 ranks per
   node) and sweep the world size with `--nodes`; each run writes
   `output/well_scale_w<N>.csv`:

   ```bash
   mkdir -p output
   sbatch -A amsc002 src/apeiron/deployment/perlmutter/well_benchmark.sbatch            #  4 ranks
   sbatch -A amsc002 --nodes=2 src/apeiron/deployment/perlmutter/well_benchmark.sbatch  #  8 ranks
   ```

3. **Memory / frontier / resume** are single-process — run them directly on a
   login node (no `srun`), per `examples/well/BENCHMARK.md`.

### Troubleshooting

- **`poetry install` fails to connect to PyPI** — Run `poetry lock` first, then retry. The lock file caches package download specs and may be stale on a new host.
- **`poetry install` fails with disk quota errors** — Poetry's default cache is in the home directory, which has limited space. Retry with `poetry install --no-cache` or free up space in `$HOME`.
