#!/usr/bin/env bash
# Expert iteration round 2 (README.md §4) as one chain of Slurm jobs:
#   1. labels from zero2-dst2 step 5000: six CPU shards of 1,250 hands;
#   2. zero2-d2c, plain PPO from step 5000 to 6000 (starts at once);
#   3. zero2-dst3, the same with distillation on the labels (after 1);
#   4. each arm's milestones against the three fair bots once it finishes:
#      policy at 5250 / 5500 / 5750 / 6000, random pre-play at 5500 / 6000,
#      and step 6000 + search (8 x 3, confidence 0.99) on both seeds.
# Run on a login node from the repository root:
#   research/ycrc/round2.sh --gpu <gpu partition> --cpu <cpu partition>
# Options: --worlds 8 --top-k 3 for 8 x 3 labels (default 12 x 4, as
# planned); --eval-on gpu to evaluate on GPUs (default: the CPU partition,
# --procs 16); --dry-run to print the sbatch commands without submitting.
# Job ids go to $TRACTOR_ROOT/round2-jobs.txt.
set -euo pipefail
GPU="" CPU="" WORLDS=12 TOPK=4 EVAL_ON=cpu DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --worlds) WORLDS="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --eval-on) EVAL_ON="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU" ] && [ -n "$CPU" ] || { echo "need --gpu and --cpu partitions" >&2; exit 2; }
case "$EVAL_ON" in cpu|gpu) ;; *) echo "--eval-on cpu or gpu" >&2; exit 2 ;; esac
[ -f research/ycrc/train.sbatch ] || { echo "run from the repository root" >&2; exit 2; }

# As env.sh sets it; the jobs source env.sh themselves.
: "${TRACTOR_ROOT:=$HOME/project/tractor-runs}"
CK=$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
LABELS=$TRACTOR_ROOT/distill/dst2-5000-w${WORLDS}k${TOPK}
EVALS=$TRACTOR_ROOT/evals
if [ "$DRY" = 0 ]; then
  [ -f "$CK" ] || { echo "missing $CK (README.md §2)" >&2; exit 1; }
  mkdir -p "$EVALS" "$LABELS"
fi

N=0
submit() {  # VAR sbatch-arguments...: VAR gets the job id (--dry-run: a stand-in)
  # Locals named so that they cannot shadow the caller's VAR.
  local _submit_var=$1 _submit_id
  shift
  if [ "$DRY" = 1 ]; then
    N=$((N + 1))
    _submit_id="DRY$N"
    echo "$_submit_id: sbatch $*" >&2
  else
    _submit_id=$(sbatch --parsable "$@" | cut -d';' -f1)
  fi
  printf -v "$_submit_var" '%s' "$_submit_id"
}

COMMON="--arch v5 --hands 1024 --lr 1e-4 --length-buckets 4 --checkpoint-every 50 \
  --keep-last 12 --rollout-workers 16 --eval-every 0 --fixed-shapes --cuda-graphs \
  --rollout-forwards compiled --bury-policy --random-burial-frac 0.2 --bury-samples 4 \
  --no-bury-scale-minibatch --bury-reward playout --bury-landlords 512 \
  --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 --declare-hands 1024"

# shellcheck disable=SC2086
submit GEN -p "$CPU" -c 32 --mem=64G -t 6:00:00 --array=0-5 research/ycrc/gen.sbatch \
  --checkpoint "$CK" --out-dir "$LABELS" --hands 1250 --seed 2 \
  --worlds "$WORLDS" --top-k "$TOPK" --confident 0.99 --workers 32
# shellcheck disable=SC2086
submit D2C -p "$GPU" --gpus=1 research/ycrc/train.sbatch \
  --name zero2-d2c --resume "$CK" --steps 6000 $COMMON
# shellcheck disable=SC2086
submit DST3 -p "$GPU" --gpus=1 --dependency="afterok:$GEN" research/ycrc/train.sbatch \
  --name zero2-dst3 --resume "$CK" --steps 6000 $COMMON \
  --distill-paths "$LABELS/shard_*.npz" \
  --distill-coef 1.0 --distill-lr 1e-4 --distill-steps 8 --distill-minibatch 512

if [ "$EVAL_ON" = gpu ]; then
  EVAL_RES=(-p "$GPU" --gpus=1) EVAL_ARGS=()
else
  EVAL_RES=(-p "$CPU") EVAL_ARGS=(--procs 16)
fi
EVAL_IDS=()
evaluate() {  # run train-job step out-name [cluster eval flags...]
  local run=$1 after=$2 step=$3 name=$4; shift 4
  local ck id
  ck=$TRACTOR_ROOT/$run/milestones/$(printf 'step_%07d.pt' "$step")
  submit id "${EVAL_RES[@]}" --dependency="afterok:$after" research/ycrc/eval.sbatch \
    --checkpoint "$ck" --out "$EVALS/$name.jsonl" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} "$@"
  EVAL_IDS+=("$id")
}
for arm in "zero2-d2c $D2C" "zero2-dst3 $DST3"; do
  set -- $arm  # shellcheck disable=SC2086
  run=$1 job=$2
  for step in 5250 5500 5750 6000; do
    evaluate "$run" "$job" "$step" "$run-$step-policy"
  done
  for step in 5500 6000; do
    evaluate "$run" "$job" "$step" "$run-$step-random" --pre-play random
  done
  evaluate "$run" "$job" 6000 "$run-6000-policy-search8-c99" \
    --search-worlds 8 --search-top 3 --search-confident 0.99 --seed 20260923
  evaluate "$run" "$job" 6000 "$run-6000-policy-search8-c99-seed2" \
    --search-worlds 8 --search-top 3 --search-confident 0.99 --seed 20261114
done

{
  echo "# round 2, submitted $(date -u +%FT%TZ); labels ${WORLDS}x${TOPK} in $LABELS"
  echo "labels $GEN"
  echo "zero2-d2c $D2C"
  echo "zero2-dst3 $DST3"
  echo "evals ${EVAL_IDS[*]}"
} | if [ "$DRY" = 1 ]; then cat; else tee "$TRACTOR_ROOT/round2-jobs.txt"; fi
echo "records land in $EVALS; check with: squeue --me" >&2
