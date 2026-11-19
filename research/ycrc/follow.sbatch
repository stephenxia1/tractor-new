#!/usr/bin/env bash
# Expert-iteration labels on CPUs, following a training run. Every core
# reloads the run's newest checkpoint as it appears and writes small label
# shards into a rolling directory; the run rereads the newest of them
# (--distill-reload-every), so the GPU only trains.
#   sbatch -p <cpu partition> --array=0-1 research/ycrc/follow.sbatch \
#     --run-dir "$TRACTOR_ROOT/zero2-live" --out-dir "$TRACTOR_ROOT/distill/zero2-live"
# One array task per node's worth of cores. Each exits once the run has
# written final.pt; at the time limit it requeues and carries on.
#SBATCH --job-name=tractor-follow
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x-%A_%a.out
set -uo pipefail
# Slurm runs a copy of this script, so env.sh is found from where the job
# was submitted: always submit from the repository root.
source "$SLURM_SUBMIT_DIR/research/ycrc/env.sh"
trap 'echo "time limit near: requeueing"; scontrol requeue "$SLURM_JOB_ID"' USR1
# One single-threaded process per core (train/labels.py).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO/research/py"
python -m tractor.cluster follow --workers "$SLURM_CPUS_PER_TASK" \
  --task "${SLURM_ARRAY_TASK_ID:-0}" "$@" &
wait $!
