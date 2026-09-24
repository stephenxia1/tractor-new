# Handoff — Tractor RL, PPO track, 2026-09-24 (afternoon, EDT)

For the cloud session taking over from the Windows machine. Rewritten
from scratch; the previous handoffs are in git history (`33b829d`,
`1c6b962`, `34e512f`) and nothing in them is needed beyond this file
and `README.md` § *Measurements on the PPO track*.

---

## 0. What is in flight, and the first thing to do

**Training run `v5-bury`** — Modal app `tractor-train`, id
`ap-fiXapAUx62STVFCwuAImWx`, detached, one L40S. At 15:41 it was at step
3966 of 4150 at ~18 s a step, so `final.pt` lands around **16:40 EDT**.
The last pre-play phase (the 5th of 5) has already run.

```
/vol/v5-bury/checkpoints/step_00NNNNN.pt   every 50 steps
/vol/v5-bury/checkpoints/final.pt          at 4150
/vol/v5-bury/metrics.jsonl
```

**Benchmarks of `v5-bury/final.pt`.** The Windows session left a
background script waiting for `final.pt`; when it appears it launches
four `modal_eval.py::smart` runs, each as its own app, in two waves:

| app name | `--pre-play` | what it measures |
| --- | --- | --- |
| `tractor-eval-v5-policy` | `agent-policy` | the full hand with the learned burial policy — **the question** |
| `tractor-eval-v5-heur` | `agent-heur` | same declarations, heuristic burial — the bar to beat |
| `tractor-eval-v5-critic` | `agent-critic` | same declarations, critic-scored climb (v4's method) |
| `tractor-eval-v5-random` | `random` | play only, for pairing against v4 final and v3 step 2000 |

770 deals per bot, bots `smart,smart-v18,smart-v28,smart-v34`, seed
20260923 (the default). A pair took 10–25 min for v4 final, so both
waves should be done within about an hour of `final.pt`. That script
lives only on the Windows machine and only as long as that session
does, so **check before launching anything**:

```bash
modal app list | grep tractor-eval-v5      # running? then wait for them to stop
```

- If those apps **are running**, wait until they stop. Do not run the
  same measurement from here meanwhile: a pair not yet stored on the
  volume would be launched a second time.
- If they **ran and stopped**, or **never started**, run the four
  commands below. Each one first pulls every result already stored in
  `/vol/evals/` (keyed on checkpoint path, bot, deals, seed, pre-play,
  search) and measures only what is missing, so the same command
  collects a finished run or starts a missing one.

```bash
# from research/. Separate app names so they don't stop each other.
CK=/vol/v5-bury/checkpoints/final.pt
TRACTOR_EVAL_APP=tractor-eval-v5-policy modal run --detach modal_eval.py::smart --checkpoints $CK --pre-play agent-policy --out runs/v5-final-policy.jsonl
TRACTOR_EVAL_APP=tractor-eval-v5-heur   modal run --detach modal_eval.py::smart --checkpoints $CK --pre-play agent-heur   --out runs/v5-final-heur.jsonl
# after those two finish: each agent-* launch holds 4 A10Gs (one per bot), and three at once would ask for 12 of the workspace's 10
TRACTOR_EVAL_APP=tractor-eval-v5-critic modal run --detach modal_eval.py::smart --checkpoints $CK --pre-play agent-critic --out runs/v5-final-critic.jsonl
TRACTOR_EVAL_APP=tractor-eval-v5-random modal run --detach modal_eval.py::smart --checkpoints $CK --pre-play random       --out runs/v5-final-random.jsonl
```

Then tabulate (`scripts/tab_evals.py`, reproduces the README tables):

```bash
python scripts/tab_evals.py runs/v5-final-*.jsonl                         # one line per bot + the v18/v28/v34 mean
python scripts/tab_evals.py runs/v5-final-heur.jsonl runs/v5-final-policy.jsonl --paired   # policy − heuristic, same deals
python scripts/tab_evals.py evals/2026-09-24/v4-final-random.jsonl runs/v5-final-random.jsonl --paired   # did play move?
```

Copy the four files to `evals/2026-09-24/v5-final-*.jsonl` (`runs/` is
gitignored), add a README section *Where v5-bury ended* in the style of
§ *Where v4-full ended*, and update §1 below.

**How to read it.** v4 final's numbers against the three fair search
bots (smart-v18/v28/v34, averaged) were 0.525 win with the critic climb
it trained with, **0.553 with the heuristic burial**, 0.566 critic climb
+ search. The burial policy has earned its place if `agent-policy`
matches or beats `agent-heur` on v5. Pay most attention to the *win as
landlord* column (`win LL`): burial only matters in the hands the
network is landlord, and that is where the value climb lost (0.454
against 0.510).

---

## 1. Where the project stands

The agent plays the whole of a landlord-set hand: it declares trump
during the deal, buries the kitty, and plays the tricks. Every piece is
trained. Against the fair smart bots, on 770 duplicate deals (±0.025):

| | win vs smart-v18/v28/v34 | vs smart | notes |
| --- | --- | --- | --- |
| v3 step 2000, play only | ~0.50 | 0.518 | flat from step 500 to 2000 |
| v3 step 2000, play only + search | ~0.537 | 0.539 | search adds ~+0.04 |
| v4-full final, play only | 0.574 | 0.619 | **+0.075 ± 0.013 paired over v3** |
| v4-full final, full hand (critic burial, as trained) | 0.525 | 0.706 | |
| v4-full final, full hand, heuristic burial | 0.553 | 0.718 | |
| v4-full final, full hand, critic burial + search | **0.566** | **0.729** | the best measured |
| v3 step 2000 against `random` | 0.746 (+0.85 levels, +34 pts) | | 0.762 with search |

"Play only" (`--pre-play random`) draws the declaration and burial
uniformly for all four seats, so bots and network start equal; "full
hand" (`--pre-play agent-*`) lets each side declare and bury for itself.
The two are different games, so their win rates are not comparable to
each other.

The play policy **plateaued again** at v4 step 2400: the last 750 steps
of v4 added nothing measurable (+0.005 ± 0.02).

**Branch `rl-research`** is pushed at the handoff commit. Before that the
branch was 10 commits ahead of `origin/rl-research`. The user's standing
rule is **ask before pushing**; this push was for the move to the cloud.
**190 Python tests passed** at `6db412e`. The TypeScript side has no test
script here (`npm run typecheck`); every benchmark runs the eval host end
to end.

---

## 2. The code, by what it does

Everything is under `research/`. The active track is `py/tractor/`. The
DMC/population line (`py/train2.py`, `py/train_pop.py`,
`py/tractor_rl/`) is superseded; the README's first half describes it
and is out of date. Its intro line "the agent currently learns in-trick
play only" is **stale**.

| area | files | what |
| --- | --- | --- |
| play simulator | `py/tractor/sim/state.py`, `actions.py` | the rules in numpy; parity with the TS engine is tested |
| deal + declaration env | `sim/deal.py` | `DealEnv`: 100 cards from the landlord, a decision per card when a declaration is legal (pass or one of 10), closing round until 4 passes, kitty flip. Legality matches the engine on 600 scenarios (`src/dumpDeclare.ts` → `tests/fixtures/declare.json`) |
| burial | `sim/bury.py` | `heuristic_burial`, `random_burial`; the value climb (`value_burials`, actor or critic scored); the policy's sampler `policy_burials` + `bury_context` + `BuryRows` |
| play network | `nn/model.py` | `ActorCritic` (10.7M): enumerate-and-score actor, privileged critic reading all hands + kitty + the context vector (`Critic.context_proj`, zero-init; `load_model` accepts checkpoints without it) |
| declaration network | `nn/declare.py` | `DeclareNet`, MLP, 11 actions, privileged critic |
| burial network | `nn/bury_net.py` | `BuryNet`: set transformer over 54 card types (2 layers, d 128), 8 sequential picks, masked logits, value head that reads the other hands |
| play training | `train/loop.py`, `ppo.py`, `rollout.py`, `parallel.py`, `shared.py` | PPO, GAE γ 1 λ 0.95, points-only reward; `RolloutPool` workers + batched forward in the parent |
| pre-play training | `train/starts.py`, `declare_loop.py`, `bury_loop.py` | `make_starts` builds play starts from the declaration and burial policies; `DeclareLearner` trains declaration (and burial, when given a `BuryLearner`) with play frozen |
| evaluation | `eval/bots.py`, `eval/search.py`, `src/evalhost.ts`, `src/env.ts` | games against the TS bots through a Node host; inference-time search |
| Modal | `modal_train.py`, `modal_eval.py`, `modal_perf.py`, `modal_bench.py` | training, benchmark, and profiling entrypoints |

**Reward.** Trick points only: each seat gets ±points/400 per trick for
its team, and the kitty bonus is credited with the last trick
(`rollout._credit_trick`). So the critic's value × 400 is the seat's
expected team point margin from that position.

**Checkpoint format** (`train/loop.py`): `model`, `optimizer`, `step`,
`config`, plus for full-hand runs `declare: {model, optimizer,
phases_done}` and, when the burial policy is on, `bury: {model,
optimizer}`.

---

## 3. The training schedule and the `v5-bury` run

`train/loop.py` flags (all on `modal_train.py::launch`):

- `--warmup-steps W`: play only, on uniform declarations and burials,
  until step W.
- then blocks of `--play-phase-steps` (200). Each block opens with
  `--declare-phase-steps` (100 for v5) of **pre-play training**: play
  frozen, the declaration policy (and the burial policy, with
  `--bury-policy`) trained by PPO on `--declare-hands` (1024) hands a
  step. Then the block's play steps start from the frozen pre-play
  policies.
- `--random-burial-frac 0.2`: a fifth of play starts are buried at random,
  so the critic stays accurate off the burial policy's path.
- `--restart-schedule`: count pre-play phases from zero, for resuming a
  checkpoint made under a different schedule.
- `--steps` is the **absolute** final step, not a count to add.

**Declaration reward:** each seat's privileged critic value at play
start (× `reward_scale` 10).

**Burial policy** (the user's design): the landlord picks 8 cards one at
a time from its 33. Every pick shares one return: the privileged
critic's value at the play start the burial produced, from the
landlord's seat, × 10. The user chose the critic as the reward
explicitly. It trains in the pre-play phase on the same hands as the
declaration policy. Training samples picks (Gumbel); evaluation takes
the argmax. It costs ~0.1 s per 1024 hands, where the value climb cost
~7 s.

`v5-bury` was launched as:

```bash
modal run --detach modal_train.py::launch --gpu l40s --steps 4150 --hands 1024 \
  --name v5-bury --resume /vol/v4-full/checkpoints/final.pt --warmup-steps 3150 \
  --restart-schedule --play-phase-steps 200 --declare-phase-steps 100 \
  --declare-hands 1024 --bury-policy --random-burial-frac 0.2 --length-buckets 4 \
  --checkpoint-every 50 --eval-bots ladder --eval-every 250 \
  --eval-bot-deals 48 --eval-bot-budget-s 1500
```

That makes five blocks from v4 final: 100 pre-play steps + 200 play
steps each. At the end of each pre-play phase the burial return (the
landlord's critic value at play start, ×10) was:

| pre-play phase | at step | bury return | entropy | kitty points buried |
| --- | --- | --- | --- | --- |
| 0 | 3150 | +0.765 | 1.43 | 2.4 |
| 3 | 3750 | +0.960 | 1.20 | — |
| 4 | 3950 | +0.982 | 1.14 | 1.9 |

For comparison, v4's critic-scored climb reached about +0.85 on the same
scale in its last phase. **This is the critic grading the thing trained
to please it**, so it cannot show the policy buries better. The play
benchmarks decide. The declaration phase also logs `nt 0.33–0.36`: a
third of declarations are no-trump, which is high against ~20% under the
bots' declarations. Worth a look if declaration becomes the weak point.

Per-step line:

```
[3966/4150]  17.9s  collect   8.4 (fwd  2.2)  update   8.6  gpu 60.6%   3410 dec/s  ev +0.533  lw 0.00
  [declare 4:100/100]   1.4s  rows  3281  return +0.552  declared 1.00  nt 0.33  ent 0.220  bury ret +0.982 ent 1.143 pts  1.9
```

---

## 4. Evaluation

`modal_eval.py::smart` runs one container per (checkpoint, bot), 770
deals by default. Each deal is played twice with the seats swapped
(duplicate), which gives 1,540 hands and ±0.025 on a win rate.
Per-hand results let runs on the same seed be paired. Every result is
written to `/vol/evals/<key>.json` from inside its container, and a
rerun of the same command fetches it, so a dropped client loses
nothing.

- **Only fair bots**: `smart`, `smart-v18`, `smart-v28`, `smart-v34`.
  `smart-v2`…`v12` read the true deal (they cheat) and are excluded from
  `LADDER` and the defaults. v18/v28/v34 are near-copies whose results
  correlate. The README's headline numbers average those three.
- **`--pre-play`**:
  - `random`: play only, uniform draw for all seats, on CPU.
  - `bot`: the bots declare and bury for everyone.
  - `agent`: declaration net + actor-scored climb.
  - `agent-critic`: declaration net + critic climb over 8 sampled deals.
  - `agent-heur`: declaration net + heuristic burial.
  - `agent-pass`: never declares, actor climb.
  - `agent-policy`: declaration net + the learned burial policy.

  Every `agent*` mode runs on an A10G (`play_gpu`, `max_containers=8`).
- **Search**: `--search-worlds 8` (top 3, confident 0.9) runs on the
  GPU, about 30 min a bot. It samples hidden hands consistent with voids
  shown, plays each candidate to the end of the trick, and scores with
  the critic. Deep search (16 worlds, top 4, 3 extra tricks) gains only
  +0.02 more at ~7× the cost.
- **Pairing**: `scripts/tab_evals.py --paired`. Same seed means the same
  deals. The difference is tightest between two play-only runs; between
  burial modes, positions differ after the burial, so the interval is
  wider.
- Raw per-hand files: `evals/2026-09-23/`, `evals/2026-09-24/`.

---

## 5. Decisions already made, and what is refuted

- **Points-only reward** (the user's call). The critic generalises best
  there: held-out explained variance was +0.405 at level weight 0 and
  −0.324 at weight 1.
- **One PPO epoch, 1024 hands a round.** Held-out EV stops rising at
  1024.
- **Inert critic fixes, don't retry**: γ 0.99 and `detach_critic`.
- **Winner's curse in the value climb.** Taking the best of ~100 noisy
  burial estimates picks the most overrated one. The actor-scored climb
  lost to its own heuristic starting point by 0.045 as landlord. Scoring
  with the critic over sampled deals recovered some of that, but not all
  (0.454 vs 0.510 landlord win at v4 final). This is why burial became a
  policy.
- **Prefetching play starts on a thread gains nothing** (GIL). It stays
  off.
- **Exact speedups kept**:
  - the history encoder runs only to the longest history in the batch;
  - the critic's empty-history fast path;
  - burial rows are built on the device;
  - `length_buckets 4` in the PPO update.

  See README § *Where a full-hand step goes*.
- **What broke the v3 plateau is unknown.** Three things changed
  together at v4: the critic's context input, the kitty bonus, and play
  from agent-made starts.

---

## 6. Next steps, in order

1. **Benchmark `v5-bury/final.pt`** (§0) and record it. If
   `agent-policy` beats `agent-heur` as landlord, burial is done for now.
   If it loses, the likely cause is the policy exploiting critic error,
   the failure the climb had. The options are:
   - return = the hand's actual outcome, or a mix of outcome and critic;
   - a KL penalty toward the heuristic;
   - more random burials.

   The critic reward was the user's explicit choice, so **propose the
   change; don't switch it silently**.
2. **Population of 8** (plan task 7, not started). The user's spec:
   - K = 8 randomly initialised agents;
   - one agent per seat, so partners can be different agents;
   - two tables;
   - seats reshuffled every N steps;
   - landlord-set hands only.

   Today `RolloutPool` batches one network for all four seats. It needs
   per-seat network routing in the parent's forward, one learner per
   agent, and the pre-play phase run per agent.
3. **Separate what broke the plateau**: one-change-at-a-time runs from v3
   step 2000.
4. **Distil search into the policy** (search is worth +0.04).

---

## 7. Operational traps

- **Modal auth**: a cloud session needs its own `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET`. The volume is `tractor-rl`, mounted at `/vol`.
- **Two `modal run`s of the same app stop each other**
  (`APP_STATE_STOPPED`). Use `TRACTOR_EVAL_APP=<unique>` for each eval
  launch, and `TRACTOR_TRAIN_APP` to train beside a live run.
- **App-create rate limit**: launching four apps at once had one
  refused. Space launches ~90 s apart.
- **GPU limit**: the workspace allows 10 GPUs at once. `play_gpu` caps
  itself at 8 per app, and extra pairs queue rather than fail.
- **`modal app logs <id>` returns only the last ~50 lines**, with ANSI
  colours: strip `\x1b\[[0-9;]*m` before grepping and don't anchor on
  `^`. Track the step number, not a count of lines. The full history is
  in `/vol/<run>/metrics.jsonl`
  (`modal volume get tractor-rl <run>/metrics.jsonl .`).
- **A `--resume` path that does not exist silently starts from step 0.**
  Confirm the first step line.
- **Editing files under `research/` while a Modal image builds aborts
  the build.**
- **Windows only**: Git Bash rewrites `/vol/...` arguments. Prefix Modal
  commands with `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"`. Python is
  the `tractor` conda env; the base env's torch is broken.
- **Tests**: `python -m pytest py/tractor/tests -q` from `research/`
  (~6 min, 190 tests; torch 2.6.0, numpy 2.2). The tests that play
  against the bots spawn `npx tsx src/evalhost.ts`, which needs `npm ci`
  at the repository root (an npm workspace; `modal_train.py`'s image
  shows the exact build).

---

## 8. The user's standing instructions

- Ask before `git push`.
- Commit messages end with the session's `Co-Authored-By` /
  `Claude-Session` lines.
- Report win rate, levels and points together. For benchmarks, 770 deals
  a bot (±0.025), fair bots only.
- The user makes the design calls on reward and architecture: the
  critic as burial reward, landlord-set hands only, a population of 8.
  Propose alternatives rather than substituting them.
