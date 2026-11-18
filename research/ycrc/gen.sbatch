#!/usr/bin/env bash
# Expert-iteration labels: one shard per array task, each on a GPU.
#   sbatch -p gpu --gpus=1 --array=0-5 research/ycrc/gen.sbatch \
#     --checkpoint CKPT --out-dir "$TRACTOR_ROOT/distill/dst2-5000" --hands 1250 --seed 2
# A shard that already exists is skipped, so a rerun fills only the gaps.
#SBATCH --job-name=tractor-gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
# Slurm runs a copy of this script, so env.sh is found from where the job
# was submitted: always submit from the repository root.
source "$SLURM_SUBMIT_DIR/research/ycrc/env.sh"
cd "$REPO/research/py"
python -m tractor.cluster gen --shard "${SLURM_ARRAY_TASK_ID:-0}" "$@"
