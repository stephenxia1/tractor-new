#!/usr/bin/env bash
# One checkpoint against the fair bots, 770 duplicate deals each, with its
# own declarations and burials (the forwards want a GPU; the bots want the
# CPUs). Records append to --out in modal_eval.py's layout.
#   sbatch -p gpu --gpus=1 research/ycrc/eval.sbatch \
#     --checkpoint "$TRACTOR_ROOT/zero2-dst3/milestones/step_0005250.pt" \
#     --out "$TRACTOR_ROOT/evals/zero2-dst3-5250-policy.jsonl"
# Add --search-worlds 8 --search-confident 0.99 for search.
#SBATCH --job-name=tractor-eval
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a copy of this script, so env.sh is found from where the job
# was submitted: always submit from the repository root.
source "$SLURM_SUBMIT_DIR/research/ycrc/env.sh"
cd "$REPO"
python -m tractor.cluster eval --workers "$SLURM_CPUS_PER_TASK" "$@"
