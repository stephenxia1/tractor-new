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

## 4. Expert iteration, round 2

The question (methodology doc §11): does distillation keep paying when
the labels come from the stronger bot? Two arms from `zero2-dst2` step
5,000 to 6,000, identical except for the labels. Submit from the
repository root.

`zero2-dst2`'s settings:

```bash
COMMON="--arch v5 --hands 1024 --lr 1e-4 --length-buckets 4 --checkpoint-every 50 \
  --keep-last 12 --rollout-workers 16 --eval-every 0 --fixed-shapes --cuda-graphs \
  --rollout-forwards compiled --bury-policy --random-burial-frac 0.2 --bury-samples 4 \
  --no-bury-scale-minibatch --bury-reward playout --bury-landlords 512 \
  --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 --declare-hands 1024"
CK=$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
```

1. Labels, six shards of 1,250 hands (about an hour each on an A10G):
   ```bash
   sbatch -p <gpu partition> --gpus=1 --array=0-5 research/ycrc/gen.sbatch \
     --checkpoint $CK --out-dir $TRACTOR_ROOT/distill/dst2-5000 --hands 1250 --seed 2
   ```
2. The control, plain PPO (can start at once):
   ```bash
   sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
     --name zero2-d2c --resume $CK --steps 6000 $COMMON
   ```
3. Distillation on the new labels, once all six shards exist:
   ```bash
   sbatch -p <gpu partition> --gpus=1 research/ycrc/train.sbatch \
     --name zero2-dst3 --resume $CK --steps 6000 $COMMON \
     --distill-paths "$TRACTOR_ROOT/distill/dst2-5000/shard_*.npz" \
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

## 6. E7: card by card, cold start

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
