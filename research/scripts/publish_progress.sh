#!/usr/bin/env bash
# Pushes a training run's results back to GitHub so another machine can see
# them, since checkpoints and metrics are gitignored on the working branch.
#
#   bash research/scripts/publish_progress.sh cloud1
#
# Everything lands on a dedicated `training-progress` branch rather than on
# the code branch: checkpoints are ~2.7MB binaries and do not belong in the
# history someone will later read or bisect. Pull them elsewhere with:
#
#   git fetch origin training-progress
#   git checkout origin/training-progress -- research/checkpoints
#
# Requires push credentials on the box (a deploy key, a PAT in the remote
# URL, or `gh auth login`).

set -euo pipefail

RUN_NAME="${1:-cloud1}"
PROGRESS_BRANCH="${PROGRESS_BRANCH:-training-progress}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

METRICS="research/runs/${RUN_NAME}.jsonl"
CKPT="research/checkpoints/${RUN_NAME}_latest.pt"

if [[ ! -f "$METRICS" ]]; then
  echo "no metrics at $METRICS -- is the run name right?" >&2
  exit 1
fi

# A one-line human summary alongside the raw metrics, so the branch is
# readable without a JSON parser.
SUMMARY=$(python3 - "$METRICS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
if not rows:
    print("no rows yet")
else:
    r = rows[-1]
    print(
        f"{r['elapsed_s']/3600:.2f}h  hands {r['hands']}  decisions {r['decisions']}  "
        f"steps {r['learner_steps']}  dec/s {r['decisions_per_s']:.0f}  "
        f"loss {r['loss']}  atk_pts {r['attacker_points_mean']}"
    )
PY
)

echo "==> $RUN_NAME: $SUMMARY"

STASH_REF=""
if ! git diff --quiet || ! git diff --cached --quiet; then
  STASH_REF=$(git stash create "publish_progress wip") || true
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch --quiet origin "$PROGRESS_BRANCH" 2>/dev/null || true
if git show-ref --quiet "refs/remotes/origin/$PROGRESS_BRANCH"; then
  git checkout --quiet -B "$PROGRESS_BRANCH" "origin/$PROGRESS_BRANCH"
else
  # Orphan branch: progress artifacts share no history with the code.
  git checkout --quiet --orphan "$PROGRESS_BRANCH"
  git rm -rq --cached . 2>/dev/null || true
fi

mkdir -p "research/runs" "research/checkpoints"
cp "$METRICS" "research/runs/${RUN_NAME}.jsonl"
[[ -f "$CKPT" ]] && cp "$CKPT" "research/checkpoints/${RUN_NAME}_latest.pt"
{
  echo "# Training progress"
  echo
  echo "Written by research/scripts/publish_progress.sh. Code lives on the"
  echo "\`rl-research\` branch; this branch carries artifacts only."
  echo
  echo "## ${RUN_NAME}"
  echo
  echo "    $SUMMARY"
} > PROGRESS.md

git add -f PROGRESS.md "research/runs/${RUN_NAME}.jsonl" 2>/dev/null || true
[[ -f "research/checkpoints/${RUN_NAME}_latest.pt" ]] \
  && git add -f "research/checkpoints/${RUN_NAME}_latest.pt"

if git diff --cached --quiet; then
  echo "==> nothing new to publish"
else
  git -c user.name="tractor-train" -c user.email="noreply@localhost" \
    commit -q -m "progress(${RUN_NAME}): ${SUMMARY}"
  git push -q origin "$PROGRESS_BRANCH"
  echo "==> pushed to origin/$PROGRESS_BRANCH"
fi

git checkout --quiet "$CURRENT_BRANCH"
if [[ -n "$STASH_REF" ]]; then
  git stash apply --quiet "$STASH_REF" || true
fi
