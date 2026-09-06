#!/usr/bin/env bash
# Ships this working tree to a rented box over SSH and brings training up
# there, without the box ever holding a GitHub credential.
#
#   bash research/scripts/remote_train.sh --host 1.2.3.4 --port 22 --user root
#   bash research/scripts/remote_train.sh --host 1.2.3.4 --port 40022 --stage sweep
#   bash research/scripts/remote_train.sh --host 1.2.3.4 --stage train --workers 20 --games 52 --threads 10
#
# Stages, run in this order:
#   ship    tar the source up (no node_modules, no .git) and install deps,
#           then run dupcheck / selftest / return tests on the box
#   sweep   size the actor/torch core split for that machine
#   train   start the long run, detached, surviving disconnection
#   status  print the latest metrics
#   fetch   copy checkpoints and metrics back here
#   stop    kill the training process (does NOT terminate the instance)
#
# `ship` is what the other stages assume has already happened.

set -euo pipefail

HOST=""
PORT=22
USER_NAME="root"
KEY="${HOME}/.ssh/id_rsa"
STAGE="ship"
REMOTE_DIR="/root/tractor-new"
RUN_NAME="cloud1"
WORKERS=""
GAMES=""
THREADS=""
MINUTES=1440
SWEEP_SECONDS=70

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --games) GAMES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --sweep-seconds) SWEEP_SECONDS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$HOST" ]] || { echo "--host is required" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_OPTS=(-p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
TARGET="${USER_NAME}@${HOST}"

remote() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }

case "$STAGE" in

ship)
  echo "==> shipping source to ${TARGET}:${REMOTE_DIR}"
  remote "mkdir -p '$REMOTE_DIR'"
  # tar over ssh rather than rsync: rsync is not standard in Git Bash on
  # Windows, tar and ssh are. Excludes keep this to a few hundred KB.
  tar czf - -C "$REPO_ROOT" \
    --exclude='node_modules' --exclude='.git' --exclude='.next' \
    --exclude='research/checkpoints' --exclude='research/runs' \
    --exclude='.claude/worktrees' \
    package.json package-lock.json tsconfig.base.json rules.md \
    apps packages research \
    | remote "tar xzf - -C '$REMOTE_DIR'"

  echo "==> installing and verifying on the box"
  remote "cd '$REMOTE_DIR' && SKIP_CLONE=1 SKIP_VENV=1 WORKDIR='$REMOTE_DIR' \
    bash research/scripts/bootstrap_vm.sh --minutes 2 --run-name ship_smoke"
  ;;

sweep)
  echo "==> sweeping configurations (~$((SWEEP_SECONDS * 7 / 60)) min)"
  remote "cd '$REMOTE_DIR/research' && PYTHONPATH=py python3 py/sweep_throughput.py --seconds $SWEEP_SECONDS"
  ;;

train)
  [[ -n "$WORKERS" && -n "$GAMES" && -n "$THREADS" ]] \
    || { echo "train needs --workers, --games and --threads (from the sweep)" >&2; exit 2; }
  echo "==> starting $RUN_NAME: ${WORKERS}x${GAMES}, ${THREADS} threads, ${MINUTES} min"
  remote "cd '$REMOTE_DIR/research' && mkdir -p runs checkpoints && \
    PYTHONPATH=py nohup python3 py/train_dmc.py \
      --workers $WORKERS --games $GAMES --threads $THREADS --groups 2 \
      --device auto --minutes $MINUTES --run-name $RUN_NAME \
      --log-every 200 --checkpoint-every 20000 \
      > runs/${RUN_NAME}_stdout.log 2> runs/${RUN_NAME}_stderr.log & \
    echo \$! > runs/${RUN_NAME}.pid; sleep 25; \
    echo pid \$(cat runs/${RUN_NAME}.pid); \
    tail -2 runs/${RUN_NAME}.jsonl 2>/dev/null || tail -5 runs/${RUN_NAME}_stderr.log"
  ;;

status)
  remote "cd '$REMOTE_DIR/research' && \
    (nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true) && \
    tail -3 runs/${RUN_NAME}.jsonl 2>/dev/null || echo 'no metrics yet'"
  ;;

fetch)
  echo "==> fetching results"
  mkdir -p "$REPO_ROOT/research/checkpoints" "$REPO_ROOT/research/runs"
  scp "${SSH_OPTS[@]}" \
    "${TARGET}:${REMOTE_DIR}/research/checkpoints/${RUN_NAME}_latest.pt" \
    "$REPO_ROOT/research/checkpoints/" || echo "(no checkpoint yet)"
  scp "${SSH_OPTS[@]}" \
    "${TARGET}:${REMOTE_DIR}/research/runs/${RUN_NAME}.jsonl" \
    "$REPO_ROOT/research/runs/" || echo "(no metrics yet)"
  ;;

stop)
  remote "cd '$REMOTE_DIR/research' && kill \$(cat runs/${RUN_NAME}.pid) 2>/dev/null && echo stopped || echo 'not running'"
  echo "note: the instance is still billing. Terminate it in the provider console."
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  exit 2
  ;;
esac
