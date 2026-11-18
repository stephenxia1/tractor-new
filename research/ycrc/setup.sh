#!/usr/bin/env bash
# One-time setup on a Bouchet login node (a few minutes, mostly downloads):
#   bash research/ycrc/setup.sh
# A conda environment at $TRACTOR_ENV with Python 3.11, Node 20 and the
# research stack's pins (torch 2.6.0, as on Modal), then the npm workspace
# the evaluator needs. Put it in project storage: home quota is small.
set -euo pipefail
: "${TRACTOR_ENV:=$HOME/project/tractor-env}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
module purge >/dev/null 2>&1 || true
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$TRACTOR_ENV" ]; then
  conda create -y -p "$TRACTOR_ENV" -c conda-forge python=3.11 nodejs=20
fi
conda activate "$TRACTOR_ENV"
pip install --quiet "numpy>=1.26" "torch==2.6.0" pytest
cd "$REPO" && npm ci --include=dev
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
node --version
cd "$REPO/research/py" && python -m pytest -q tractor/tests/test_cluster.py
echo "setup done: TRACTOR_ENV=$TRACTOR_ENV"
