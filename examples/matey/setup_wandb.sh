#!/usr/bin/env bash
# Upgrade wandb for matey-env (supports new wandb_v1_* API keys, 86 chars).
#
# matey-env site-packages is not writable; install goes to ~/.local and must
# be prepended on PYTHONPATH (run_inference_drift.sh does this when ENABLE_WANDB=1).

set -euo pipefail

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"

# shellcheck disable=SC1090
source "${MATEY_ENV}"

echo "Current wandb (matey-env default):"
python -c "import wandb; print(wandb.__version__, wandb.__file__)" || true

echo "Installing wandb>=0.22.3 to user site-packages..."
pip install --user --upgrade --ignore-installed "wandb>=0.22.3"

USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"

echo "Upgraded wandb:"
python -c "import wandb; print(wandb.__version__, wandb.__file__)"

cat <<EOF

Done. Before running with wandb:

  export WANDB_API_KEY='wandb_v1_...'   # from https://wandb.ai/authorize

Then:

  ENABLE_WANDB=1 ./examples/matey/run_inference_drift.sh "\${BASELINE}" "\${BASELINE}" "\${CKPT}"

EOF
