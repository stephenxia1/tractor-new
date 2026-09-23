# Handoff — PPO track, 2026-09-23

Written at the end of a session that measured the critic, parallelised the
rollout, switched the reward to trick points, and trained `v2-points` to a
checkpoint that is at or above even against every rung of the bot ladder.

Read this, then `README.md` § *Measurements on the PPO track* for the tables.
Don't re-derive those numbers; extend them.

---

## 1. Where things stand

**Branch `rl-research`, working tree clean, 4 commits unpushed:**

```
18696d5  Train on an L40S, and print what the step is waiting on
7f7d7ba  Decompose the rollout's serial floor; widen the pool to 512
c3ac43e  Points-only reward; 1024-hand rounds, both measured
ea808a0  Measure the critic's generalisation; parallelise the rollout
```

`git push` has not been run — the user never asked for it. Ask before pushing.

**160 Python tests pass** (`py/` under the `tractor` conda env — see §5).
The TypeScript engine tests were **not** re-run this session; nothing in
`packages/engine` was touched after the last time they passed.

**The active track is `py/tractor/`** — PPO, enumerate-and-score actor,
privileged critic, driven by `modal_train.py` on Modal. The DMC/population
line (`py/train2.py`, `py/train_pop.py`, `py/tractor_rl/`) still exists but
is not where work happens.

---

## 2. The trained checkpoint

Run **`v2-points`**: 1250 steps × 1024 hands, points-only reward, 1 epoch,
7 rollout workers, pool 512, on an **L40S**. Finished in **7.5 h** at ~20 s
a step and ~3,400 decisions/s.

Checkpoint: `/vol/v2-points/checkpoints/final.pt` on the Modal volume
`tractor-rl`. Intermediate checkpoints every 125 steps, 12 kept.

Final ladder, 16 duplicate deals (32 hands) a rung:

| rung | winrate | 95% CI | levels | points |
| --- | --- | --- | --- | --- |
| random | 0.969 | [0.91, 1.03] | +1.344 | +50.0 |
| lowball | 0.969 | [0.91, 1.03] | +1.312 | +53.1 |
| point-hoarder | 0.969 | [0.91, 1.03] | +1.500 | +55.3 |
| aggressive | 0.938 | [0.85, 1.02] | +1.594 | +61.1 |
| smart | 0.656 | [0.49, 0.82] | +0.281 | +13.8 |
| smart-v2 | 0.656 | [0.49, 0.82] | +0.250 | +10.6 |
| smart-v4 | 0.594 | [0.42, 0.77] | +0.125 | +0.3 |
| smart-v9 | 0.531 | [0.36, 0.71] | −0.062 | −4.1 |
| smart-v18 | 0.594 | [0.42, 0.77] | −0.031 | −3.1 |
| smart-v28 | 0.594 | [0.42, 0.77] | +0.000 | −1.2 |
| smart-v34 | 0.594 | [0.42, 0.77] | +0.000 | −1.7 |

The previous run (`v1-selfplay-c`, 2500 steps, annealed reward) scored 0.375
against `smart-v34` and 0.281 against `smart-v4`.

### Do not state "beats the ladder" as established

At 32 hands a rung the standard error is **0.087**, and **no single rung is
individually significant**: `smart` is the strongest at z = +1.86, `smart-v34`
at z = +1.08. What carries the claim is that all seven smart rungs sit at or
above 0.53 together — under noise you would expect about half below.

**This is the top outstanding task** (§6.1).

---

## 3. What was built

| file | what it is |
| --- | --- |
| `py/tractor/train/critic_diag.py` | Holdout diagnostic: keeps a fraction of **hands** out of the fit, runs the real `PPOLearner.update`, scores both halves. Bootstrap clustered by hand. |
| `py/tractor/train/parallel.py` | `RolloutPool` — persistent worker processes running the simulator, one batched forward in the parent. |
| `py/tractor/train/shared.py` | The shared-memory buffer the workers and parent both write. The buffer *is* the transport. |
| `py/bench_ipc.py` | Standalone IPC benchmark that decided the transport design. |
| `py/tractor/train/ppo.py` | Gained `update(rows=)` (fit a subset) and `PPOConfig.detach_critic`. |
| `py/tractor/train/buffer.py` | Gained a pluggable `alloc` and `window(start, stop)`. |
| `py/tractor/train/loop.py` | `rollout_workers`, a `buffer_capacity` check, per-step progress printing, pool held across the run in a try/finally. |
| `modal_train.py` | `train_l40s`, `critic`, `critic_hands`, `rollout` entrypoints. |

New tests: `test_critic_diag.py`, `test_parallel.py`, `test_shared.py`, plus
additions to `test_ppo.py` and `test_loop.py`.

---

## 4. Decisions, and what is already refuted

**The reward is trick points only** (`anneal_steps=0`, `final_level_weight=0.0`).
The user's call, and independently where the critic generalises best: held-out
explained variance is **+0.405 at level weight 0 and −0.324 at 1**, where
negative is worse than predicting the mean.

Points are a **proxy** and this is the live risk. They cannot see the kitty
bonus at all (never in a trick record, up to 8× the kitty's points), and they
rate 119 points and 0 points as 119 apart when the game scores them
identically. **The ladder win rate is the instrument that catches this** — so
far win rate and point margin have moved together, so no sign of failure.

**No anneal is also a resumability decision.** A stationary reward means a
checkpoint means the same thing whenever you resume it.

**Two critic fixes are measurably inert — do not retry them:**

| change | held-out EV | vs baseline −0.324 |
| --- | --- | --- |
| `gamma = 0.99` | −0.328 | no effect |
| `detach_critic = True` | −0.320 | no effect |
| **1 epoch instead of 4** | **+0.030** | the only lever |

A hand is ~22 decisions a seat, so `0.99²² ≈ 0.80` barely separates rows that
share a terminal term; and the memorisation is in the critic head reading the
privileged view, not in the shared trunk.

**1024 hands a round.** Held-out EV across 128/256/512/1024/2048 runs
+0.023, +0.404, +0.378, +0.473, +0.479 — the rise stops at 1024, where the
calibration slope also peaks at 0.95. 2048 buys +0.006 for twice the rollout.
Rollout cost is exactly linear in hands.

**One epoch stays even on points-only.** The penalty shrinks but does not
vanish: paired on identical hands at 1024, one epoch holds out at +0.473
against +0.424 for four, and calibrates at 0.95 against 0.79.

**`v2-points` started from scratch**, not from `v1-selfplay-c/final.pt` —
that checkpoint's critic was fitted to the annealed level reward.

---

## 5. Operational traps

Each of these cost real time. They are not discoverable from the code.

- **Use the `tractor` conda env**:
  `/c/Users/xiast/anaconda3/envs/tractor/python.exe` — torch 2.6.0 +
  numpy 2.2.5, matching the Modal image. The anaconda **base** env's torch is
  broken (`WinError 1114` on `c10.dll`; the files verify clean against pip's
  RECORD hashes, so it is an OS-level break and a reinstall will not fix it).
  `pytest` was installed into `tractor` on 2026-09-23.
- **Git Bash mangles POSIX paths.** `--checkpoint /vol/...` becomes
  `C:/Program Files/Git/vol/...`. Prefix every Modal command with
  `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"`.
- **Two concurrent `modal run`s of the same app kill each other**
  (`ConflictError: APP_STATE_STOPPED`). Run them sequentially.
- **Editing anything under `research/` while a Modal image builds aborts the
  build** ("was modified during build process").
- **`modal app logs` returns only the last ~50 lines.** A monitor that counts
  matching lines saturates and reports false stalls — track the step *number*.
- **A GPU-share check must not divide by `step_s`.** Eval steps take ~250 s,
  so the share collapses and a naive threshold fires. Compute it from the
  spans: `(forward + update) / (collect + update)`.
- **`TaskStop` kills the shell wrapper, not its children.** A `find /` survived
  it and burned 2,413 s of CPU over 2h20m. Verify with `Get-Process`.
- **Heredocs mangle escaped newlines** in patch scripts. Write patch scripts
  with the Write tool instead.
- **A round wider than `buffer_capacity` does not raise** — `collect` silently
  returns fewer hands. Now a config-time error; keep
  `buffer_capacity ≥ hands_per_step × 100`.

---

## 6. Outstanding work, in priority order

### 6.1 Confirm the ladder result (highest value, ~30 min)

32 hands a rung cannot establish any single rung. Re-run the three rungs that
matter at 128 deals:

```bash
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" modal run modal_train.py::critic \
  --checkpoint /vol/v2-points/checkpoints/final.pt
```

…or better, a dedicated ladder run against `smart`, `smart-v9`, `smart-v34`
at `num_deals=128`. That halves the interval to ~±0.09 and settles whether
0.594 against `smart-v34` is real.

### 6.2 The history encoder (largest remaining speedup)

A step is ~71% GPU. The rollout's parent-side floor is ~10.1 s at 1024 hands,
**flat in worker count**, and **76% of it is the forward**. Proved to be
arithmetic, not launch overhead: tripling the batch (pool 128 → 512) cut the
forward only 9.5%.

`HistoryEncoder.forward` runs **all 128 positions densely** whatever the real
history length. Packing to actual lengths is the candidate. More cores will
not help — 16 reserved cores buy under 4% of a step for ~20% more cost.

### 6.3 Overlap the forward (a real 1.4× on collect)

`RolloutPool` is lockstep: the parent forwards while every worker idles.
Two batches in flight per worker would hide most of the 10.1 s floor behind
worker packing. Needs each worker to hold two independent pools.

### 6.4 Cheaper parent-side data movement (~1.9 s at 1024 hands)

Each worker's rows are contiguous, so the `buf.history[rows]` fancy-index
gather could be a slice. The copy-out into the caller's buffer is a second
full pass over the round's bytes and exists only because the learner does not
read the shared block directly.

### 6.5 Push the 4 commits

Ask the user first.

---

## 7. Commands

All from `research/`, all with the `tractor` env's python.

```bash
# Tests (~6 min)
/c/Users/xiast/anaconda3/envs/tractor/python.exe -m pytest py/tractor/tests -q

# Critic generalisation: the anneal sweep and the candidate fixes
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" modal run modal_train.py::critic \
  --checkpoint /vol/v2-points/checkpoints/final.pt

# Critic quality vs round width
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" modal run modal_train.py::critic_hands \
  --checkpoint /vol/v2-points/checkpoints/final.pt --epochs 1 --level-weight 0.0

# Serial vs parallel rollout, with the parent-phase breakdown
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" modal run modal_train.py::rollout \
  --checkpoint /vol/v2-points/checkpoints/final.pt --hands 1024 --workers 7,15 --cpus 16

# Resume training (no anneal, so a checkpoint resumes cleanly)
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" modal run --detach modal_train.py::launch \
  --gpu l40s --steps 1250 --name v3 --resume /vol/v2-points/checkpoints/final.pt \
  --eval-bots ladder --eval-every 250

# Watch a detached run (one line per step)
modal app logs <app-id>

# Pull a run's metrics
modal volume get tractor-rl v2-points/metrics.jsonl ./metrics.jsonl --force
```

The per-step line reads:

```
[354/1250]  20.1s  collect   8.9 (fwd  3.4)  update  10.8  gpu 70.9%   3360 dec/s  ev +0.331  lw 0.00
```

`gpu` is occupancy, not utilisation — it says the device was busy, not that it
was busy doing useful arithmetic. `lw 0.00` confirms no annealing.

---

## 8. Corrections made this session

Recorded so they are not re-derived or re-asserted:

- **γ<1 was recommended as the primary critic fix. It does nothing** (−0.328
  vs −0.324). `detach_critic` likewise (−0.320).
- **A "knee at 256 hands" was asserted off a 512-hand dip.** Wrong — 1024 is
  meaningfully better (+0.473 vs +0.378) and the plateau is at 1024.
- **The first parallel rollout spawned workers per round** and measured 1.08×,
  with the entire gain inside ~9.5 s of `import torch` per worker.
- **The first IPC design** sent observations through a queue per decision:
  ~1,150 decisions/s, *slower* than the 1,261/s loop it replaced.
- **`smart-v18` and `smart-v28` returning identical eval rows is not a bug** —
  v28 is v18 plus a conditional cost relaxation that did not bind on those
  deals. They diverge on other seeds.
