#
module load python/3.13-26.1.0 # Load supported python version
python -m venv .venv # Create a virtual environment
source ./.venv/bin/activate # Activate environment
pip install poetry # Install poetry
poetry lock # Sync poetry
poetry install --no-cache # Install poetry
