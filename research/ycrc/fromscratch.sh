#!/usr/bin/env bash
# A clean run from scratch (research/README.md, "Expert iteration in the
# run, and a clean round"): every label made in the run, nothing resumed
# from a frozen-label lineage.
#   1. the trunk, plain PPO from random initialisation on zero2's recipe:
#      steps 0-3000 at lr 3e-4 (PPOConfig's default), then 3000-4000 at
#      1e-4 in the same run directory (as zero2-lr1e4 resumed zero2).
#      One difference from zero2: the burial's play-out return from step 0
#      (zero2 took it from step 900; its earlier settings were not saved);
#   2. the trunk's milestones at 3000 and 4000 against the fair bots, to
#      set beside zero2 step 3000 and zero2-lr1e4 step 4000;
#   3. from its step 4000, two pairs of arms to step 5000, each pair an
#      in-run distilled arm and a plain-PPO control on one training seed
#      (round3.sh --to 5000 --after <trunk>), with their milestones.
# Run on a login node from the repository root:
#   research/ycrc/fromscratch.sh --gpu <gpu partition> --cpu <cpu partition> \
#     --labels-root <scratch dir> [--account A] [--name clean1] [--seed 3] [--submit]
# Without --submit it prints the sbatch commands and submits nothing.
# Job ids go to $TRACTOR_ROOT/fromscratch-<name>-jobs.txt.
set -euo pipefail
GPU="" CPU="" ACCOUNT="" NAME=clean1 SEED=3 LROOT="" DRY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --labels-root) LROOT="$2"; shift 2 ;;
    --submit) DRY=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU" ] && [ -n "$CPU" ] && [ -n "$LROOT" ] || {
  echo "need --gpu, --cpu and --labels-root" >&2; exit 2; }
[ -f research/ycrc/train.sbatch ] || { echo "run from the repository root" >&2; exit 2; }
ACCT=(); [ -n "$ACCOUNT" ] && ACCT=(-A "$ACCOUNT")

: "${TRACTOR_ROOT:=$HOME/project/tractor-runs}"
RUN=$TRACTOR_ROOT/$NAME
EVALS=$TRACTOR_ROOT/evals
JOBS=$TRACTOR_ROOT/fromscratch-$NAME-jobs.txt
ms() { printf '%s/milestones/step_%07d.pt' "$RUN" "$1"; }
if [ "$DRY" = 0 ]; then
  for r in "$NAME" "$NAME-live" "$NAME-ppo" "$NAME-live-s2" "$NAME-ppo-s2"; do
    if [ -e "$TRACTOR_ROOT/$r" ]; then
      echo "$TRACTOR_ROOT/$r exists; not training over it" >&2; exit 1
    fi
  done
  mkdir -p "$EVALS"
fi

N=0
submit() {  # VAR sbatch-arguments...: VAR gets the job id (dry run: a stand-in)
  local _submit_var=$1 _submit_id
  shift
  if [ "$DRY" = 1 ]; then
    N=$((N + 1))
    _submit_id="TRUNK$N"
    echo "$_submit_id: sbatch $*" >&2
  else
    _submit_id=$(sbatch --parsable "$@" | cut -d';' -f1)
  fi
  printf -v "$_submit_var" '%s' "$_submit_id"
}

# round3.sh's COMMON without --lr: zero2's settings.
ZERO2="--arch v5 --hands 1024 --length-buckets 4 --checkpoint-every 50 \
  --keep-last 12 --rollout-workers 16 --eval-every 0 --fixed-shapes --cuda-graphs \
  --rollout-forwards compiled --bury-policy --random-burial-frac 0.2 --bury-samples 4 \
  --no-bury-scale-minibatch --bury-reward playout --bury-landlords 512 \
  --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 --declare-hands 1024"

# shellcheck disable=SC2086
submit A -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 research/ycrc/train.sbatch \
  --name "$NAME" --steps 3000 --seed "$SEED" $ZERO2
# shellcheck disable=SC2086
submit B -p "$GPU" ${ACCT[@]+"${ACCT[@]}"} --gpus=1 --dependency="afterok:$A" \
  research/ycrc/train.sbatch --name "$NAME" --resume "$(ms 3000)" --steps 4000 \
  --seed "$SEED" --lr 1e-4 $ZERO2
submit E3 -p "$CPU" ${ACCT[@]+"${ACCT[@]}"} --dependency="afterok:$A" \
  research/ycrc/eval.sbatch --procs 16 --checkpoint "$(ms 3000)" \
  --out "$EVALS/$NAME-3000-policy.jsonl"
submit E4 -p "$CPU" ${ACCT[@]+"${ACCT[@]}"} --dependency="afterok:$B" \
  research/ycrc/eval.sbatch --procs 16 --checkpoint "$(ms 4000)" \
  --out "$EVALS/$NAME-4000-policy.jsonl"

{
  echo "# from scratch, $( [ "$DRY" = 1 ] && echo "dry run" || echo "submitted") $(date -u +%FT%TZ); seed $SEED"
  echo "trunk $NAME $A (0-3000, lr 3e-4) $B (3000-4000, lr 1e-4)"
  echo "trunk evals $E3 $E4"
} | if [ "$DRY" = 1 ]; then cat; else tee "$JOBS"; fi

# The branches: one pair a training seed, from the trunk's step 4000.
SUB=(); [ "$DRY" = 0 ] && SUB=(--submit)
for pair in "1 " "2 -s2"; do
  read -r k sfx <<<"$pair" || true
  research/ycrc/round3.sh --gpu "$GPU" --cpu "$CPU" ${ACCOUNT:+--account "$ACCOUNT"} \
    --resume "$(ms 4000)" --after "$B" --to 5000 --labels-root "$LROOT" \
    --name "$NAME-live${sfx:-}" --fresh-control "$NAME-ppo${sfx:-}" \
    --seed "$((SEED * 10 + k))" ${SUB[@]+"${SUB[@]}"} \
    | if [ "$DRY" = 1 ]; then cat; else tee -a "$JOBS"; fi
done
