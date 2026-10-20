#!/bin/bash
# Aurora setup for uv.lock-pinned env

# Login nodes need the proxy for any download (uv, pip, python-build-standalone).
if [[ "${HOSTNAME}" =~ aurora-uan ]]; then
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
fi

module use /soft/modulefiles
module load frameworks                      # provides uv (and the frameworks Python)

# Keep uv's caches off $HOME quota. Flare is faster than --no-cache for repeat syncs.
export UV_CACHE_DIR=/lus/flare/projects/<proj>/$USER/.uv-cache
export UV_PYTHON_INSTALL_DIR=/lus/flare/projects/<proj>/$USER/.uv-python

# No python/3.13 module on Aurora. Let uv fetch a 3.13 interpreter.
uv python install 3.13
uv sync --locked --python 3.13
source .venv/bin/activate
