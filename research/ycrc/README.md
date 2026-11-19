# Running the research stack on YCRC Bouchet

Modal's credits ran out on 2026-11-17. This runs the same training,
search labels and evaluation as Slurm jobs on Bouchet, through
`python -m tractor.cluster` (`research/py/tractor/cluster.py`), which takes
the Modal launchers' flags and writes the same files.

Driven from a Claude Code session on your own computer (a cloud session
cannot reach YCRC), over the `bouchet` SSH alias with connection sharing,
so one Duo approval covers many commands.

## 0. Your computer

`~/.ssh/config`:

```
Host bouchet
  HostName bouchet.ycrc.yale.edu
  User <netid>
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
```

Open the shared connection once (`ssh bouchet`, approve Duo) and keep that
terminal open. `ssh bouchet hostname` must then answer without Duo.

## 1. Once, on Bouchet

```bash
ssh bouchet
mkdir -p ~/project && cd ~/project          # project storage; home quota is small
git clone -b rl-research https://github.com/stephenxia1/tractor-new
cd tractor-new
bash research/ycrc/setup.sh                 # conda env, torch 2.6.0, Node 20, npm ci
sinfo -s                                    # which GPU partitions you can use
```

`~/project` is the YCRC project-storage link; if it does not exist, use the
path `mydirectories` or `getquota` shows and set `TRACTOR_ENV` and
`TRACTOR_ROOT` to directories there before `setup.sh` (see `env.sh`).

## 2. Checkpoints

The best checkpoints are on Modal's volume, whose storage still answers
without credits, and in a draft GitHub release ("Zero checkpoints").
From Bouchet, with a Modal token (`pip install modal && modal token new`,
or copy `~/.modal.toml` from your computer):

```bash
source research/ycrc/env.sh && pip install --quiet modal
mkdir -p $TRACTOR_ROOT/ckpt
modal volume get tractor-rl zero2-dst2/milestones/step_0005000.pt $TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
```

## 3. A smoke test first

```bash
srun -p <gpu partition> --gpus=1 -c 8 --mem=32G -t 30:00 --pty bash
source research/ycrc/env.sh && cd research/py
python -m tractor.cluster train --root $TRACTOR_ROOT --name smoke --overwrite \
  --steps 2 --hands 16 --rollout-workers 4 --eval-every 0 --checkpoint-every 1 \
  --warmup-steps 1 --play-phase-steps 1 --declare-phase-steps 1 --declare-hands 16 \
  --bury-policy --resume $TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
```

## 4. Wider search, then expert iteration round 2

The questions (2026-11-19): does search at 12 worlds x top 4 beat 8 x 3
enough to be worth its cost, and does distillation keep paying when the
labels come from the wider search? Confidence 0.99 and look-ahead to the
end of the trick throughout. Submit from the repository root.

```bash
CK=$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
mkdir -p $TRACTOR_ROOT/evals
```

**4a. The cost and the gain of 12 x 4** (two jobs, one per seed; the 8 x 3
records for the same checkpoint and deals are in
`research/evals/2026-11-16/zero2-dst2-5000-policy-search8-c99*.jsonl`):

```bash
for SEED in 20260923 20261114; do
  sbatch -p <gpu partition> --gpus=1 research/ycrc/eval.sbatch --checkpoint $CK \
    --search-worlds 12 --search-top 4 --search-confident 0.99 --seed $SEED \
    --out $TRACTOR_ROOT/evals/zero2-dst2-5000-policy-search12x4-c99-$SEED.jsonl
done
```

A platform check alongside (one bot, no search, about half an hour): its
per-deal results should match the Modal record
`zero2-dst2-5000-policy.jsonl` deal for deal. If they do, the paired
comparison above measures the search width alone.

```bash
sbatch -p <gpu partition> --gpus=1 research/ycrc/eval.sbatch --checkpoint $CK \
  --bots smart-v18 --out $TRACTOR_ROOT/evals/zero2-dst2-5000-policy-ycrc.jsonl
```

**4b. Labels** from the 12 x 4 search, six shards of 1,250 hands. On a
CPU partition, 32 cores a shard (about 10 minutes each, §5b):

```bash
sbatch -p <cpu partition> -c 32 --mem=64G -t 6:00:00 --array=0-5 research/ycrc/gen.sbatch \
  --checkpoint $CK --out-dir $TRACTOR_ROOT/distill/dst2-5000-w12k4 --hands 1250 --seed 2 \
  --worlds 12 --top-k 4 --confident 0.99 --workers 32
```

or on GPUs as before (drop `--workers`, use `-p <gpu partition> --gpus=1`).

**4c. Two arms from step 5,000 to 6,000**, identical except for
distillation, on `zero2-dst2`'s settings:

```bash
COMMON="--arch v5 --hands 1024 --lr 1e-4 --length-buckets 4 --checkpoint-every 50 \
  --keep-last 12 --rollout-workers 16 --eval-every 0 --fixed-shapes --cuda-graphs \
  --rollout-forwards compiled --bury-policy --random-burial-frac 0.2 --bury-samples 4 \
  --no-bury-scale-minibatch --bury-reward playout --bury-landlords 512 \
  --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 --declare-hands 1024"
```

1. The control, plain PPO (does not depend on the labels; can start at once):
   ```bash
   sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
     --name zero2-d2c --resume $CK --steps 6000 $COMMON
   ```
2. Distillation on the new labels, once all six shards exist, as `dst2`
   (coef 1.0, lr 1e-4, 8 minibatches of 512 a step):
   ```bash
   sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
     --name zero2-dst3 --resume $CK --steps 6000 $COMMON \
     --distill-paths "$TRACTOR_ROOT/distill/dst2-5000-w12k4/shard_*.npz" \
     --distill-coef 1.0 --distill-lr 1e-4 --distill-steps 8 --distill-minibatch 512
   ```

A step-1,000 run took about 5 hours on Modal's H100; slower GPUs take
longer. A job that reaches its time limit requeues itself and resumes
from its newest checkpoint.

## 5. Evaluation

Milestones land in `$TRACTOR_ROOT/<run>/milestones/` every 250 steps.
Each is measured with its own declarations and burials against the three
fair bots, 770 deals a bot (the standing benchmark):

```bash
sbatch -p <gpu partition> --gpus=1 research/ycrc/eval.sbatch \
  --checkpoint $TRACTOR_ROOT/zero2-dst3/milestones/step_0005250.pt \
  --out $TRACTOR_ROOT/evals/zero2-dst3-5250-policy.jsonl
```

Compare the arms at the same step, paired on the same deals (both records
use seed 20260923 by default), and report win rate, levels and points
together. Copy records into `research/evals/<date>/` and commit them to
`rl-research`.

## 5b. Labels on CPUs, and continuous expert iteration

Search runs on CPUs now (`eval/fastnet.py`; methodology in
`research/README.md`, *Search on CPUs*): measured at about 0.2 s per
searched decision per core at 12 worlds x top 4, all cores busy. A hand
is about 100 decisions, two thirds of them searched, so about 14 s of
one core and 65 label rows per hand. Bouchet's CPU partitions are the
place for it; `sinfo -s` lists them.

**A fixed shard on CPUs** (the §4b labels without a GPU): one process per
core, merged into one shard:

```bash
sbatch -p <cpu partition> -c 32 --mem=64G -t 6:00:00 --array=0-5 research/ycrc/gen.sbatch \
  --checkpoint $CK --out-dir $TRACTOR_ROOT/distill/dst2-5000-w12k4 --hands 1250 --seed 2 \
  --worlds 12 --top-k 4 --confident 0.99 --workers 32
```

(about 1,250 hands x 14 s / 32 cores = 10 minutes a shard, if Bouchet's
cores match the ones measured; an A10G took about an hour a shard at 8 x 3.)

**Continuous expert iteration.** The run distils from a rolling
directory that CPU jobs keep filling from its newest checkpoint; labels
trail the policy by roughly a checkpoint interval plus a shard's time,
and each refresh logs `ppo/distill/label_lag` (steps). Until the first
shards exist the run trains without distilling.

```bash
RUN=zero2-live
sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
  --name $RUN --resume $CK --steps 6000 $COMMON \
  --distill-paths "$TRACTOR_ROOT/distill/$RUN/s*.npz" --distill-reload-every 25 \
  --distill-window-rows 200000 \
  --distill-coef 1.0 --distill-lr 1e-4 --distill-steps 8 --distill-minibatch 512
sbatch -p <cpu partition> --array=0-1 research/ycrc/follow.sbatch \
  --run-dir $TRACTOR_ROOT/$RUN --out-dir $TRACTOR_ROOT/distill/$RUN \
  --hands 16 --worlds 12 --top-k 4 --confident 0.99 --keep 100
```

Sizing: `dst2`'s settings read 8 x 512 = 4,096 label rows a step. At
~18 s a step (Modal's H100; slower GPUs take longer) that is ~230 rows a
second, which ~50 cores at 12 x 4 produce, each row then read about
once. Two tasks of 32 cores cover it. The window of 200,000 rows (about
50 steps' worth) takes ~6 GB of the trainer's memory. A shard names the
checkpoint step it came from (`s0005250-t001-w07-00012.npz`), and each
process keeps only its own newest `--keep` shards.

## 6. E7: card by card, cold start

**On hold** (2026-11-19) until the zero2 work in §4 is under way.

`research/E7.md`: every play chosen card by card, the Large SwiGLU
networks (`--arch v7`), from random initialisation. `zero2`'s
cold-start recipe (1024 hands a step, warm-up 300 steps, blocks of 100
pre-play and 200 play steps, leave-one-out over 4 burials with
played-out returns), with play-outs from the start and without the
compiled or graph-captured forwards, which card mode does not do yet.

Smoke first (2 steps, a few minutes on a GPU):

```bash
python -m tractor.cluster train --root $TRACTOR_ROOT --name e7-smoke --overwrite \
  --arch v7 --action-space cards --steps 2 --hands 32 --rollout-workers 4 \
  --eval-every 0 --checkpoint-every 1 --warmup-steps 1 --play-phase-steps 1 \
  --declare-phase-steps 1 --declare-hands 32 --bury-policy --bury-reward playout \
  --bury-landlords 8 --bury-samples 4 --no-bury-scale-minibatch --fixed-shapes
```

The run:

```bash
sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
  --name e7-v7 --steps 4000 --arch v7 --action-space cards --hands 1024 \
  --lr 1e-4 --checkpoint-every 50 --keep-last 12 --rollout-workers 16 --eval-every 0 \
  --fixed-shapes --bury-policy --random-burial-frac 0.2 --bury-samples 4 \
  --no-bury-scale-minibatch --bury-reward playout --bury-landlords 512 \
  --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 --declare-hands 1024
```

`--lr 1e-4` departs from `zero2`'s 3e-4: the larger v6 networks
collapsed on `zero2`'s recipe (methodology doc §7, Exp 3) and 1e-4 is
what held `zero2`'s peak. Drop the flag for `zero2`'s exact settings.
Milestones are measured with `eval.sbatch` as in §5; card mode is picked
up from the checkpoint.

## Rules that hold here as on Modal

- **Zero.** Nothing may train on heuristic knowledge: labels come from the
  agent's own search, critic and the rules.
- **Benchmarks.** Fair bots only (`smart-v18`, `smart-v28`, `smart-v34`),
  770 deals a bot. Win rate, levels and points reported together.
- **Branches.** Commit to `rl-research`; never push to `main` (it deploys
  the game server).
- **Login nodes** are for small commands; anything heavier goes through
  `srun` or `sbatch`.
