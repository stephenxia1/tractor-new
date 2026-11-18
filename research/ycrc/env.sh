# Sourced by every job script (and by hand on a login node):
#   source research/ycrc/env.sh
# TRACTOR_ENV: the conda environment setup.sh made.
# TRACTOR_ROOT: where runs, labels and records live (project storage).
: "${TRACTOR_ENV:=$HOME/project/tractor-env}"
: "${TRACTOR_ROOT:=$HOME/project/tractor-runs}"
export TRACTOR_ENV TRACTOR_ROOT
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO
module purge >/dev/null 2>&1 || true
module load miniconda >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TRACTOR_ENV"
export PYTHONPATH="$REPO/research/py${PYTHONPATH:+:$PYTHONPATH}"
# The evaluator starts `npx tsx research/src/evalhost.ts` from the repo root.
export PATH="$REPO/node_modules/.bin:$PATH"
mkdir -p "$TRACTOR_ROOT"
