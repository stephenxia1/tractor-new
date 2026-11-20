#!/usr/bin/env bash
# Expert iteration round 3 (README.md §4, "Continuous expert iteration"):
# distillation on labels made from the run's own newest checkpoint while it
# trains, against plain PPO, both from zero2-dst2 step 5000 to 6000.
#   1. zero2-live: train.sbatch with the distillation of zero2-dst3, but
#      reading a rolling label directory (--distill-reload-every);
#   2. its followers: follow.sbatch on CPUs, from when the trainer starts
#      until it writes final.pt, labelling from its newest checkpoint
#      (the step-5000 one until it has its own);
#   3. the control: zero2-d2c, already trained with the same COMMON flags
#      (round 2), or a fresh one with --fresh-control NAME;
#   4. both arms' milestones at 5250 / 5500 / 5750 / 6000 against the three
#      fair bots, 770 deals, seed 20260923, with today's eval.sbatch, whose
#      records keep per-deal levels (the control's round-2 records do not,
#      so it is measured again, under new names).
# Run on a login node from the repository root:
#   research/ycrc/round3.sh --gpu <gpu partition> --cpu <cpu partition> [--submit]
# Without --submit it prints the sbatch commands and submits nothing.
# Options: --worlds/--top-k for the labels (default 8 x 3, as zero2-dst3's;
# 12 x 4 measured no better at ~1.45x the cost), --account to charge.
# Job ids go to $TRACTOR_ROOT/round3-jobs.txt.
set -euo pipefail
GPU="" CPU="" WORLDS=8 TOPK=3 ACCOUNT="" CONTROL=zero2-d2c FRESH=0 DRY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --worlds) WORLDS="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --fresh-control) CONTROL="$2"; FRESH=1; shift 2 ;;
    --submit) DRY=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU" ] && [ -n "$CPU" ] || { echo "need --gpu and --cpu partitions" >&2; exit 2; }
[ -f research/ycrc/train.sbatch ] || { echo "run from the repository root" >&2; exit 2; }
ACCT=(); [ -n "$ACCOUNT" ] && ACCT=(-A "$ACCOUNT")

: "${TRACTOR_ROOT:=$HOME/project/tractor-runs}"
CK=$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
RUN=zero2-live
LABELS=$TRACTOR_ROOT/distill/$RUN
EVALS=$TRACTOR_ROOT/evals
if [ "$DRY" = 0 ]; then
  [ -f "$CK" ] || { echo "missing $CK (README.md §2)" >&2; exit 1; }
  [ -e "$TRACTOR_ROOT/$RUN" ] && { echo "$TRACTOR_ROOT/$RUN exists; not training over it" >&2; exit 1; }
  if [ "$FRESH" = 0 ]; then
    for s in 5250 5500 5750 6000; do
      f=$TRACTOR_ROOT/$CONTROL/milestones/$(printf 'step_%07d.pt' "$s")
      [ -f "$f" ] || { echo "missing control milestone $f" >&2; exit 1; }
    done
  fi
  mkdir -p "$EVALS" "$LABELS"
fi

N=0
submit() {  # VAR sbatch-arguments...: VAR gets the job id (dry run: a stand-in)
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
# zero2-dst3's distillation, on a rolling window of fresh labels.
DISTILL="--distill-coef 1.0 --distill-lr 1e-4 --distill-steps 8 --distill-minibatch 512"

# shellcheck disable=SC2086
submit LIVE -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 research/ycrc/train.sbatch \
  --name "$RUN" --resume "$CK" --steps 6000 $COMMON $DISTILL \
  --distill-paths "$LABELS/s*.npz" --distill-reload-every 25 \
  --distill-window-rows 200000 --distill-max-age 150
# Two tasks of 32 cores, from when the trainer starts (it writes launch.json,
# which --initial-checkpoint auto reads); each exits once final.pt exists.
submit FOLLOW -p "$CPU" ${ACCT[@]+"${ACCT[@]}"} --array=0-1 --dependency="after:$LIVE" \
  research/ycrc/follow.sbatch \
  --run-dir "$TRACTOR_ROOT/$RUN" --out-dir "$LABELS" --hands 16 \
  --worlds "$WORLDS" --top-k "$TOPK" --confident 0.99 --keep 100 --initial-checkpoint auto
CTL=""
if [ "$FRESH" = 1 ]; then
  # shellcheck disable=SC2086
  submit CTL -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 research/ycrc/train.sbatch \
    --name "$CONTROL" --resume "$CK" --steps 6000 $COMMON
fi

EVAL_IDS=()
evaluate() {  # run after-job(or "") step out-name
  local run=$1 after=$2 step=$3 name=$4 ck id
  local dep=(); [ -n "$after" ] && dep=(--dependency="afterok:$after")
  ck=$TRACTOR_ROOT/$run/milestones/$(printf 'step_%07d.pt' "$step")
  submit id -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 ${dep[@]+"${dep[@]}"} \
    research/ycrc/eval.sbatch --checkpoint "$ck" --out "$EVALS/$name.jsonl"
  EVAL_IDS+=("$id")
}
for step in 5250 5500 5750 6000; do
  evaluate "$RUN" "$LIVE" "$step" "$RUN-$step-policy"
  evaluate "$CONTROL" "$CTL" "$step" "$CONTROL-$step-policy-r3"
done

{
  echo "# round 3, $( [ "$DRY" = 1 ] && echo "dry run" || echo "submitted") $(date -u +%FT%TZ); labels ${WORLDS}x${TOPK} in $LABELS"
  echo "$RUN $LIVE"
  echo "followers $FOLLOW"
  echo "control $CONTROL ${CTL:-(already trained)}"
  echo "evals ${EVAL_IDS[*]}"
  echo "# report, once every record exists (B - A per milestone, then pooled):"
  printf 'python research/scripts/tab_evals.py --paired'
  for step in 5250 5500 5750 6000; do
    printf ' \\\n  %s %s' "$EVALS/$CONTROL-$step-policy-r3.jsonl" "$EVALS/$RUN-$step-policy.jsonl"
  done
  echo
} | if [ "$DRY" = 1 ]; then cat; else tee "$TRACTOR_ROOT/round3-jobs.txt"; fi
