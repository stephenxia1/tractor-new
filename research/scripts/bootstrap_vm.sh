#!/usr/bin/env bash
# Brings a bare Ubuntu/Debian VM to a running training job, and leaves it
# running after you disconnect. Everything is idempotent, so re-running it
# after a reboot or a failed step is safe.
#
#   curl -fsSL https://raw.githubusercontent.com/stephenxia1/tractor-new/rl-research/research/scripts/bootstrap_vm.sh | bash -s -- --minutes 1440
#
# or, with the repo already cloned:
#   bash research/scripts/bootstrap_vm.sh --workers 24 --games 64 --threads 12
#
# Sizing: the loop is roughly half learner, half acting forward pass, with
# the Node workers needing about one core each. On an N-core box, start
# with workers = N-4 and threads = 4-8, then confirm with
# `python3 py/sweep_throughput.py` -- the best split is machine-specific
# and the wrong one costs 2-3x (see RESULTS.md).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/stephenxia1/tractor-new.git}"
BRANCH="${BRANCH:-rl-research}"
WORKDIR="${WORKDIR:-$HOME/tractor-new}"
RUN_NAME="${RUN_NAME:-cloud1}"

# Defaults sized for a 16-core box; override on the command line.
WORKERS=12
GAMES=64
THREADS=8
GROUPS=2
MINUTES=1440

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --games) GAMES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --groups) GROUPS="$2"; shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "==> installing system packages"
if ! command -v node >/dev/null 2>&1 || [[ "$(node -v | cut -c2-3)" -lt 20 ]]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends git python3 python3-pip python3-venv

echo "==> fetching $BRANCH"
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch --quiet origin "$BRANCH"
  git -C "$WORKDIR" checkout --quiet "$BRANCH"
  git -C "$WORKDIR" reset --hard --quiet "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "==> installing node workspace deps"
npm ci --include=dev --silent

echo "==> installing python deps"
python3 -m venv "$WORKDIR/.venv" 2>/dev/null || true
# shellcheck disable=SC1091
source "$WORKDIR/.venv/bin/activate"
pip install --quiet --upgrade pip
if [[ -n "${CUDA_INDEX:-}" ]]; then
  pip install --quiet numpy "torch>=2.0" --index-url "$CUDA_INDEX"
else
  pip install --quiet numpy "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
fi

cd "$WORKDIR/research"
export PYTHONPATH="$WORKDIR/research/py"

echo "==> verifying the stack before spending money on it"
npx tsx src/dupcheck.ts --hands 3 --games 1
HANDS=10 npx tsx src/selftest.ts | tail -3
python3 py/test_returns.py | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DEVICE_ARG=""
if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE_ARG="--device cuda"
  echo "==> CUDA available; training on GPU"
fi

mkdir -p runs checkpoints
echo "==> starting training ($RUN_NAME): $WORKERS workers x $GAMES games, $THREADS threads, $MINUTES min"
nohup python3 py/train_dmc.py \
  --workers "$WORKERS" --games "$GAMES" --threads "$THREADS" --groups "$GROUPS" \
  --minutes "$MINUTES" --run-name "$RUN_NAME" \
  --log-every 200 --checkpoint-every 20000 $DEVICE_ARG \
  > "runs/${RUN_NAME}_stdout.log" 2> "runs/${RUN_NAME}_stderr.log" &

echo "$!" > "runs/${RUN_NAME}.pid"
sleep 25
echo "==> pid $(cat "runs/${RUN_NAME}.pid"); first metrics:"
tail -2 "runs/${RUN_NAME}.jsonl" 2>/dev/null || tail -5 "runs/${RUN_NAME}_stderr.log"
cat <<EOF

Training is running detached. Useful next commands on this box:

  tail -f research/runs/${RUN_NAME}.jsonl          # metrics stream
  bash research/scripts/publish_progress.sh ${RUN_NAME}   # push results home
  kill \$(cat research/runs/${RUN_NAME}.pid)        # stop

EOF
