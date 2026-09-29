#
module load python/3.13-26.1.0 # Load supported python version

command -v uv >/dev/null || python -m pip install --user uv # Install uv into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Create .venv with exactly the versions in uv.lock, using the module's Python.
# --no-cache keeps uv's download cache out of the limited $HOME quota.
uv sync --locked --no-cache --python "$(command -v python)"
source .venv/bin/activate # Activate environment
