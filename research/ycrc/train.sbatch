#!/usr/bin/env bash
# A training run on one GPU. Pass the partition and GPU on the command
# line, and tractor.cluster's flags after the script:
#   sbatch -p gpu --gpus=1 research/ycrc/train.sbatch --name zero2-dst3 --resume CKPT ...
# 20 CPUs for 16 rollout workers, as on Modal's H100. At the time limit the
# job requeues itself; the requeued job keeps its id, so tractor.cluster
# resumes from the run's newest checkpoint (written every 50 steps).
#SBATCH --job-name=tractor-train
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -uo pipefail
# Slurm runs a copy of this script, so env.sh is found from where the job
# was submitted: always submit from the repository root.
source "$SLURM_SUBMIT_DIR/research/ycrc/env.sh"
trap 'echo "time limit near: requeueing"; scontrol requeue "$SLURM_JOB_ID"' USR1
nvidia-smi -L
cd "$REPO/research/py"
python -m tractor.cluster train --root "$TRACTOR_ROOT" "$@" &
wait $!
