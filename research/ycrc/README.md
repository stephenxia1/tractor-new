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
  --steps 5002 --hands 16 --rollout-workers 4 --eval-every 0 --checkpoint-every 1 \
  --milestone-every 1 \
  --warmup-steps 1 --play-phase-steps 1 --declare-phase-steps 1 --declare-hands 16 \
  --bury-policy --resume $TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
```

`--steps` is the total, not a count from the checkpoint: resumed at step
5,000, `--steps 2` trains nothing and only writes `final.pt`. 5002 runs two
steps, each with a declaration phase, and `--milestone-every 1` copies both
to `milestones/`.

## 4. Wider search, then expert iteration round 2

The questions (2026-11-19): does search at 12 worlds x top 4 beat 8 x 3
enough to be worth its cost, and does distillation keep paying when the
labels come from the wider search? Confidence 0.99 and look-ahead to the
end of the trick throughout. Submit from the repository root.

```bash
CK=$TRACTOR_ROOT/ckpt/zero2-dst2-5000.pt
mkdir -p $TRACTOR_ROOT/evals
```

**4a. The cost and the gain of 12 x 4: done, no need to run.** Measured
in a cloud session on CPUs, both widths on the same platform, 770 deals
per fair bot, both seeds (`research/README.md`, "Wider search"): 12 x 4 -
8 x 3 is -0.003 +/- 0.009 win, -0.003 +/- 0.020 levels, +0.12 +/- 0.58
points over 9,240 paired hands, at ~1.45x the cost per searched decision.
The platform check came out at -0.001 +/- 0.001 win against the Modal
records. Nor are 12 x 4's labels cleaner: two independent searches of
the same position choose the same move 0.725 of the time at 12 x 4 and
0.719 at 8 x 3 (+0.007 +/- 0.013, 7,183 positions; `research/README.md`,
"Distillation flattened the policy"). **Before 4b, decide the label
width**: 12 x 4 as planned, or 8 x 3, which played as well for two thirds
of the cost. That section also lists label changes to consider (soft
targets, agreement-filtered labels, 12 x 3); none is implemented.

**All of 4b, 4c and §5's milestone evaluations in one command**, as a
chain of Slurm jobs (`round2.sh`; job ids in `$TRACTOR_ROOT/round2-jobs.txt`):

```bash
research/ycrc/round2.sh --gpu <gpu partition> --cpu <cpu partition>              # 12 x 4 labels
research/ycrc/round2.sh --gpu <gpu partition> --cpu <cpu partition> --worlds 8 --top-k 3   # or 8 x 3
```

It submits the six label shards (CPU), `zero2-d2c` at once, `zero2-dst3`
once every shard has succeeded (`afterok`), and, once each arm's training
job has finished, that arm's milestones: policy at 5250 / 5500 / 5750 /
6000, random pre-play at 5500 / 6000, and step 6000 + search (8 x 3,
0.99) on both seeds, 19 jobs in all. Evaluations run on the CPU partition
with `--procs 16`; `--eval-on gpu` puts them on GPUs. `--dry-run` prints
the `sbatch` lines and submits nothing. If a job fails, the jobs waiting
on it stay pending (`DependencyNeverSatisfied`): `scancel` them, fix, and
resubmit that part by hand from the commands below.

Expect more labels than round 1: `zero2-dst2` is far less sure of itself
than the policy round 1 labelled (search acts on 64-66% of its decisions,
against 29% for `zero2-lr1e4` step 4000), so a hand gives ~41 rows, not
~17: ~300k rows, ~9 GB of float16 for the six shards (held in the
training job's memory, which has 96 GB). A test shard of 16 hands at
12 x 4 took 50 s on 4 cores in a cloud session.

**4b. Labels** from the 12 x 4 search, six shards of 1,250 hands. On a
CPU partition, 32 cores a shard (about 10 minutes each, §5b):

```bash
sbatch -p <cpu partition> -c 32 --mem=64G -t 6:00:00 --array=0-5 research/ycrc/gen.sbatch \
  --checkpoint $CK --out-dir $TRACTOR_ROOT/distill/dst2-5000-w12k4 --hands 1250 --seed 2 \
  --worlds 12 --top-k 4 --confident 0.99 --workers 32
```

For 8 x 3 labels instead, the same with `--worlds 8 --top-k 3` and
`--out-dir $TRACTOR_ROOT/distill/dst2-5000-w8k3` (and that directory in
4c's `--distill-paths`).

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

Round 2's report once `round2.sh`'s records are in (`zero2-dst3` against
`zero2-d2c`, each milestone and then pooled, as round 1 was reported;
then step 6000 with search over both seeds):

```bash
E=$TRACTOR_ROOT/evals
python research/scripts/tab_evals.py --paired \
  $(for s in 5250 5500 5750 6000; do echo $E/zero2-d2c-$s-policy.jsonl $E/zero2-dst3-$s-policy.jsonl; done)
python research/scripts/tab_evals.py --paired \
  $E/zero2-d2c-6000-policy-search8-c99.jsonl $E/zero2-dst3-6000-policy-search8-c99.jsonl \
  $E/zero2-d2c-6000-policy-search8-c99-seed2.jsonl $E/zero2-dst3-6000-policy-search8-c99-seed2.jsonl
```

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
  --distill-window-rows 200000 --distill-max-age 150 \
  --distill-coef 1.0 --distill-lr 1e-4 --distill-steps 8 --distill-minibatch 512
sbatch -p <cpu partition> --array=0-1 research/ycrc/follow.sbatch \
  --run-dir $TRACTOR_ROOT/$RUN --out-dir $TRACTOR_ROOT/distill/$RUN \
  --hands 16 --worlds 12 --top-k 4 --confident 0.99 --keep 100 \
  --initial-checkpoint auto
```

`--initial-checkpoint auto` labels from the checkpoint the run resumed
from (its `launch.json`) until the run writes one of its own, a
checkpoint interval in; without it the followers wait and those first
steps train undistilled. `--distill-max-age 150` leaves out shards
labelled more than 150 steps ago (three checkpoint intervals), so a
follower that falls behind -- queued, requeued -- stops the distillation
rather than leaving it on old labels; each refresh logs
`ppo/distill/label_age_mean`, `label_age_max` and `shards_too_old`
beside `label_lag`, `rows` and `shards`. A requeued trainer rereads the
window at its first step, and its checkpoints carry the distiller's
optimiser (about 85 MB more each, live runs only), so the requeue
resumes its moments. With none of these flags a run is as before.

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
