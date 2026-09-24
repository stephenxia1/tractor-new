# Handoff — PPO track, 2026-09-23

Written at the end of a session that measured the critic, parallelised the
rollout, switched the reward to trick points, and trained `v2-points` to a
checkpoint that is at or above even against every rung of the bot ladder.

Read this, then `README.md` § *Measurements on the PPO track* for the tables.
Don't re-derive those numbers; extend them.

---

## 00. Update, 2026-09-24 — the full hand. Read this first.

The agent now declares trump and buries the kitty itself (landlord-set
hands), trained in alternation with play. Design and measured step costs
are in `README.md` § *The full hand* and § *Where a full-hand step goes*.
What a new session needs beyond that:

- **Run `v4-full`** (Modal app `tractor-train`, detached): resumed from
  `v3-points/step_0002000.pt`; warm-up to step 2150 on uniform starts,
  then 5 blocks of a 50-step declaration phase + 200 play steps, to 3150.
  1024 hands, `bury_consider 12`, `random_burial_frac 0.2`,
  `length_buckets 4`. Checkpoints `/vol/v4-full/checkpoints/`; each
  carries the declaration network under `"declare"`. It was restarted
  at step 2100 to pick up the TF32 burial and length buckets, and at
  step 2550 to bury by the critic (`--bury-evaluator critic`,
  `--bury-consider 0`) after the actor-scored climb measured worse than
  its own heuristic starting point (README § *What the full hand
  bought*).
- **The critic changed shape** (`Critic.context_proj`). `load_model`
  loads older checkpoints with it at zero, and `load_checkpoint` extends
  an older optimiser state. Anything that calls `load_state_dict`
  directly on a pre-2026-09-24 checkpoint will now raise.
- **The reward changed**: the kitty bonus is credited with the last
  trick. Critic numbers from before are not comparable.
- **Evaluation modes** (`modal_eval.py --pre-play`): `random` (play only,
  the default), `bot` (the bots declare and bury for everyone), `agent`
  (the network declares with the checkpoint's declaration network and
  buries by the value climb; runs on a GPU container), `agent-pass`
  (same, but the network never declares). Each result is written to
  `/vol/evals/<key>.json` from inside its container; rerun the same
  command to collect what a dropped client missed.
- **Run separate apps side by side**: `TRACTOR_TRAIN_APP` and
  `TRACTOR_EVAL_APP` override the app names. Two `modal run`s of one
  app stop each other.
- **Not built**: the population of 8 (task 7 of the plan). Everything
  up to it is single-agent.

---

## 0. Update, late 2026-09-23

A second session re-measured everything at 770 deals a bot, changed how
evaluation is run, and added search at inference time. **Several claims
below are superseded by this section**, notably §2's "at or above even
against every rung".

### What changed in evaluation

- **Only fair bots.** `smart-v2` through `smart-v12` search on the true
  deal (v2 simulates each trick with the other seats' real cards; v3+ read
  opponent hands; see the `smart-v13` comment in
  `packages/engine/src/bot.ts`). They are out of `LADDER`
  (`py/tractor/eval/bots.py`) and out of `modal_eval.py`'s defaults. The
  fair smart bots are `smart`, `smart-v18`, `smart-v28`, `smart-v34`
  (v13+ sample hidden hands). The last three are near-copies; their results
  on the same deals are strongly correlated.
- **Random pre-play by default.** Before this, the opponent bot declared
  and buried for all four seats, the network's included. Now
  `GameSlot.randomPrePlay` (`src/env.ts`, `--pre-play random|bot` in
  `src/evalhost.ts`, `pre_play=` in `play_against_bot`) draws the
  declaration uniformly over legal (seat, declaration) pairs and the burial
  uniformly — the training generator's distribution (`gen/episode.py`) —
  from a stream keyed on the deal, so both duplicate halves still match.
  This also changes what the in-training ladder evaluation measures.
- **`modal_eval.py`** (separate Modal app, default `tractor-eval`,
  overridable with `TRACTOR_EVAL_APP`): one container per
  (checkpoint, bot), 770 deals (±0.025 on a win rate), per-hand results for
  paired comparisons, appended to a JSONL as each pair lands and resumed on
  rerun. Modal preempts containers; rerun the same command to fill gaps.

### What was measured (all 770 deals, 1,540 hands a bot)

The policy has been **flat since about step 500**. Paired on the same
hands, step 1250 and step 1625 differ from step 500 by +0.004 and +0.002
(±0.015) against v18/v28/v34. `v3-points` (resumed from `v2-points`) was
stopped at step 2003; its last checkpoint is
`/vol/v3-points/checkpoints/step_0002000.pt`.

v3 step 2000, random pre-play:

| opponent | | win rate | levels/deal | points/deal |
| --- | --- | --- | --- | --- |
| random | no search | 0.746 ± 0.022 | +0.850 ± 0.041 | +34.0 ± 1.6 |
| random | **search** | **0.762** | **+0.897** | **+36.5** |
| smart | no search | 0.518 ± 0.025 | +0.037 ± 0.037 | +1.7 ± 1.4 |
| smart | **search** | **0.539** | **+0.115** | **+4.6** |
| smart-v18 | no search | 0.499 | −0.003 | +0.4 |
| smart-v18 | **search** | **0.537** | **+0.080** | **+3.5** |
| smart-v28 | no search | 0.499 | +0.004 | +0.6 |
| smart-v28 | **search** | **0.537** | **+0.080** | **+3.3** |
| smart-v34 | no search | 0.498 | −0.006 | +0.2 |
| smart-v34 | **search** | **0.538** | **+0.088** | **+3.6** |

Search's paired gain against v18/v28/v34 is +0.038 to +0.040 win rate
(±0.022 each, every one excluding zero). Under the old bot pre-play the
policy scored ~0.49 against those three; the old table in §2 came from 16
deals a rung and is noise.

Against `random` search adds +0.016 ± 0.018 win rate and +2.6 ± 1.6
points a deal, paired (the random bot draws from `Math.random`, so its
own moves differ between the two runs).

Raw per-hand results: `evals/2026-09-23/*.jsonl` (copied from the
gitignored `runs/`). A deep search (16 worlds, top 4, 3 extra tricks,
confident 0.95) against the four smart bots was still in flight when
this was written and may be missing there.

### Inference-time search (`py/tractor/eval/search.py`)

For a decision where the policy's top move has under `confident` (0.9)
probability: sample `worlds` (8) deals of the unseen cards consistent with
the seat's view (own hand, the table, hand sizes, suits a seat has shown
itself out of, the kitty only if it buried it); play each of the policy's
`top_k` (3) moves in every world; let the greedy policy play all seats to
the end of the trick (+`extra_tricks`); score = points settled for the
searching team + 400 × the privileged critic's value in that world (+ the
kitty bonus if the hand ended); take the best mean. `tests/test_search.py`
re-deals the hidden cards and requires identical worlds and choices.

It must run on a GPU: a CPU forward is ~100 ms a row on one thread.
`modal_eval.py --search-worlds 8` routes to an A10G container; about 40
minutes a smart bot at 770 deals, a third of decisions searched.

### Next steps, in order

1. Read the deep-search result (or rerun it) to see if more search helps.
2. Train on realistic declarations and burials instead of uniform ones,
   and credit the kitty bonus in the reward (it is invisible to the
   points-only reward and the critic).
3. Opponent pool of past checkpoints instead of pure self-play (entropy has
   been flat at ~0.31 since step 500).
4. Distil search back into the policy.

### Corrections to the commands below

- `--steps` is the **absolute** step count. Resuming a step-1250
  checkpoint with `--steps 1250` trains nothing; use `--steps 2500` for
  1250 more. A `--resume` path that does not exist silently starts from
  step 0 — confirm the first log line reads `[1251/...]`.
- §6.1's `modal run modal_train.py::critic` is the critic diagnostic, not
  a ladder evaluation. Use `modal_eval.py`.
- Log lines carry ANSI colour codes, so `grep "^\[123/"` never matches.
- The `modal` CLI is `/c/Users/xiast/anaconda3/envs/tractor/Scripts/modal.exe`
  on the Windows machine; a cloud session needs its own Modal token
  (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) to run anything.

```bash
# The measurements above (from research/)
modal run modal_eval.py::smart --checkpoints /vol/v3-points/checkpoints/step_0002000.pt \
  --pre-play random --out runs/eval.jsonl                       # policy alone
modal run modal_eval.py::smart --checkpoints /vol/v3-points/checkpoints/step_0002000.pt \
  --pre-play random --search-worlds 8 --out runs/eval.jsonl     # with search
```

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
