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
# 12 x 4 measured no better at ~1.45x the cost), --account to charge,
# --resume CK and --name RUN to start the in-run arm elsewhere (default
# zero2-dst2 step 5000 and zero2-live; pair with --fresh-control).
# --seed N for both trainers and the followers (default: TrainConfig's,
# and follow's 7); --agree for labels two searches agree on (about twice
# the follower cost per label, so with more --followers N, default 2).
# --control NAME --no-control pairs against a control trained and
# measured by another submission, and submits none.
# --labels-root DIR puts the rolling label shards there (default
# $TRACTOR_ROOT/distill; scratch is the place for them). Each follower
# process keeps its newest --keep N shards (default 20: with 64 processes
# ~1,300, about 1.5x a 200,000-row window) and deletes them once the run
# has finished. (Keeping 100 each filled the class's project quota on
# 2026-11-21: round 3's cache alone was 124 GB.)
# Job ids go to $TRACTOR_ROOT/round3-jobs.txt (round3-<name>-jobs.txt
# with --name).
set -euo pipefail
GPU="" CPU="" WORLDS=8 TOPK=3 ACCOUNT="" CONTROL=zero2-d2c FRESH=0 DRY=1
RESUME="" RUN=zero2-live SEED="" AGREE=0 NFOLLOW=2 NOCTL=0 LROOT="" KEEP=20
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --worlds) WORLDS="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --fresh-control) CONTROL="$2"; FRESH=1; shift 2 ;;
    --control) CONTROL="$2"; FRESH=0; shift 2 ;;
    --no-control) NOCTL=1; shift ;;
    --resume) RESUME="$2"; shift 2 ;;
    --name) RUN="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --agree) AGREE=1; shift ;;
    --followers) NFOLLOW="$2"; shift 2 ;;
    --labels-root) LROOT="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --submit) DRY=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU" ] && [ -n "$CPU" ] || { echo "need --gpu and --cpu partitions" >&2; exit 2; }
[ -f research/ycrc/train.sbatch ] || { echo "run from the repository root" >&2; exit 2; }
if [ "$NOCTL" = 1 ] && [ "$FRESH" = 1 ]; then
  echo "--no-control and --fresh-control together" >&2; exit 2
fi
ACCT=(); [ -n "$ACCOUNT" ] && ACCT=(-A "$ACCOUNT")
SEEDA=(); [ -n "$SEED" ] && SEEDA=(--seed "$SEED")
AGREEA=(); [ "$AGREE" = 1 ] && AGREEA=(--agree)

: "${TRACTOR_ROOT:=$HOME/project/tractor-runs}"
CK=${RESUME:-$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt}
LABELS=${LROOT:-$TRACTOR_ROOT/distill}/$RUN
EVALS=$TRACTOR_ROOT/evals
JOBS=$TRACTOR_ROOT/round3-jobs.txt
[ "$RUN" = zero2-live ] || JOBS=$TRACTOR_ROOT/round3-$RUN-jobs.txt
if [ "$DRY" = 0 ]; then
  [ -f "$CK" ] || { echo "missing $CK (README.md §2)" >&2; exit 1; }
  [ -e "$TRACTOR_ROOT/$RUN" ] && { echo "$TRACTOR_ROOT/$RUN exists; not training over it" >&2; exit 1; }
  if [ "$FRESH" = 1 ]; then
    if [ -e "$TRACTOR_ROOT/$CONTROL" ]; then
      echo "$TRACTOR_ROOT/$CONTROL exists; not training over it" >&2; exit 1
    fi
  elif [ "$NOCTL" = 0 ]; then
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
  --distill-window-rows 200000 --distill-max-age 150 ${SEEDA[@]+"${SEEDA[@]}"}
# Tasks of 32 cores (two by default), from when the trainer starts (it writes
# launch.json, which --initial-checkpoint auto reads); each exits once
# final.pt exists.
submit FOLLOW -p "$CPU" ${ACCT[@]+"${ACCT[@]}"} --array=0-$((NFOLLOW - 1)) \
  --dependency="after:$LIVE" research/ycrc/follow.sbatch \
  --run-dir "$TRACTOR_ROOT/$RUN" --out-dir "$LABELS" --hands 16 \
  --worlds "$WORLDS" --top-k "$TOPK" --confident 0.99 --keep "$KEEP" --cleanup \
  --initial-checkpoint auto ${SEEDA[@]+"${SEEDA[@]}"} ${AGREEA[@]+"${AGREEA[@]}"}
CTL=""
if [ "$FRESH" = 1 ]; then
  # shellcheck disable=SC2086
  submit CTL -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 research/ycrc/train.sbatch \
    --name "$CONTROL" --resume "$CK" --steps 6000 $COMMON ${SEEDA[@]+"${SEEDA[@]}"}
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
  if [ "$NOCTL" = 0 ]; then
    evaluate "$CONTROL" "$CTL" "$step" "$CONTROL-$step-policy-r3"
  fi
done

if [ "$NOCTL" = 1 ]; then CTL_NOTE="(trained and measured elsewhere)"
else CTL_NOTE=${CTL:-(already trained)}; fi
{
  echo "# round 3, $( [ "$DRY" = 1 ] && echo "dry run" || echo "submitted") $(date -u +%FT%TZ); from $CK; labels ${WORLDS}x${TOPK}$( [ "$AGREE" = 1 ] && echo ", agreed" || true)${SEED:+, seed $SEED} in $LABELS"
  echo "$RUN $LIVE"
  echo "followers $FOLLOW"
  echo "control $CONTROL $CTL_NOTE"
  echo "evals ${EVAL_IDS[*]}"
  echo "# report, once every record exists (B - A per milestone, then pooled):"
  printf 'python research/scripts/tab_evals.py --paired'
  for step in 5250 5500 5750 6000; do
    printf ' \\\n  %s %s' "$EVALS/$CONTROL-$step-policy-r3.jsonl" "$EVALS/$RUN-$step-policy.jsonl"
  done
  echo
} | if [ "$DRY" = 1 ]; then cat; else tee "$JOBS"; fi
