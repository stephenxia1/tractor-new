# Handoff — Tractor RL, PPO track, 2026-09-24 (afternoon, EDT)

For the cloud session taking over from the Windows machine. Rewritten
from scratch; the previous handoffs are in git history (`33b829d`,
`1c6b962`, `34e512f`) and nothing in them is needed beyond this file
and `README.md` § *Measurements on the PPO track*.

---

## 0. Nothing is in flight (updated 2026-09-28)

**`v5-bury` is finished and benchmarked.** `final.pt` landed at 16:46
EDT on 09-24, and all four benchmarks (16 pairs) were stored on the
volume by 20:06. No cloud session committed anything, so the results
were taken straight from `/vol/evals/` on 09-28. The local launcher had
in fact finished too: its `runs/bench-v5-final-*.jsonl` match them hand
for hand. They are in
`evals/2026-09-24/v5-final-{policy,heur,critic,random}.jsonl` and
written up in README § *Where v5-bury ended*.

**The learned burial earned its place.** Against smart-v18/v28/v34,
averaged: learned burial 0.566 (0.523 as landlord), heuristic 0.561
(0.513), critic climb 0.552 (0.495). Paired, learned − heuristic is
+0.005 ± 0.011 and learned − critic climb is +0.014 ± 0.010. By §6's
criterion burial is done for now. Play only did not move (+0.006 ±
0.012 over v4 final).

To rerun or extend a benchmark, `modal_eval.py::smart` with the same
arguments first fetches every result already on the volume, then
measures only what is missing:

```bash
# from research/. A unique app name per launch; see §7.
CK=/vol/v5-bury/checkpoints/final.pt
TRACTOR_EVAL_APP=tractor-eval-v5-policy modal run --detach modal_eval.py::smart --checkpoints $CK --pre-play agent-policy --out runs/v5-final-policy.jsonl
python scripts/tab_evals.py evals/2026-09-24/v5-final-*.jsonl
python scripts/tab_evals.py evals/2026-09-24/v5-final-heur.jsonl evals/2026-09-24/v5-final-policy.jsonl --paired
```

`modal app list` forgets stopped apps after a few days. To find out
whether something ran, list the volume (`modal volume ls tractor-rl
evals`) rather than the apps.

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
| v4-full final, full hand, critic burial + search | 0.566 | 0.729 | |
| v5-bury final, play only | 0.580 | 0.632 | +0.006 ± 0.012 paired over v4 final |
| v5-bury final, full hand, learned burial (as trained) | **0.566** | 0.719 | **the best measured without search** |
| v5-bury final, full hand, heuristic burial | 0.561 | 0.713 | |
| v5-bury final, full hand, critic burial | 0.552 | 0.725 | |
| v3 step 2000 against `random` | 0.746 (+0.85 levels, +34 pts) | | 0.762 with search |

"Play only" (`--pre-play random`) draws the declaration and burial
uniformly for all four seats, so bots and network start equal; "full
hand" (`--pre-play agent-*`) lets each side declare and bury for itself.
The two are different games, so their win rates are not comparable to
each other.

The play policy **plateaued again** at v4 step 2400: the last 750 steps
of v4 added nothing measurable (+0.005 ± 0.02), and neither did v5's
thousand (+0.006 ± 0.012). v5 moved the burial, not the play.

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
| play network | `nn/model.py` | `ActorCritic`: enumerate-and-score actor (10.68M at v5), privileged critic reading all hands + kitty + the context vector (`Critic.context_proj`, zero-init; `load_model` accepts checkpoints without it). v6: `ffn="swiglu"`, `critic_kind="cards"` (`CardCritic`) |
| architectures | `nn/arch.py` | `ARCHES` (`v5`, `v6`; `launch --arch`), rebuilding a checkpoint's declaration and burial nets from their saved configs, `check_saved`. Every config defaults to v5; `tests/test_arch.py` pins v5's layouts and outputs |
| declaration network | `nn/declare.py` | `DeclareNet`, MLP, 11 actions, privileged critic |
| burial network | `nn/bury_net.py` | `BuryNet`: set transformer over 54 card types (2 layers, d 128), 8 sequential picks, masked logits, value head that reads the other hands |
| play training | `train/loop.py`, `ppo.py`, `rollout.py`, `parallel.py`, `shared.py` | PPO, GAE γ 1 λ 0.95, points-only reward; `RolloutPool` workers + batched forward in the parent |
| pre-play training | `train/starts.py`, `declare_loop.py`, `bury_loop.py` | `make_starts` builds play starts from the declaration and burial policies; `DeclareLearner` trains declaration (and burial, when given a `BuryLearner`) with play frozen |
| population | `train/population.py` | eight agents, one per seat at two tables, reseated every N steps; routes rows by `agents[hand, seat]` through `collect`/`RolloutPool`/`make_starts`/`critic_values`; `read_checkpoint("…pt#k")` hands out one agent. README § *A population of eight* |
| evaluation | `eval/bots.py`, `eval/search.py`, `src/evalhost.ts`, `src/env.ts` | games against the TS bots through a Node host; inference-time search |
| Modal | `modal_train.py`, `modal_eval.py`, `modal_perf.py`, `modal_bench.py` | training, benchmark, and profiling entrypoints |

**Reward.** Trick points only: each seat gets ±points/400 per trick for
its team, and the kitty bonus is credited with the last trick
(`rollout._credit_trick`). So the critic's value × 400 is the seat's
expected team point margin from that position.

**Checkpoint format** (`train/loop.py`): `model`, `optimizer`, `step`,
`config`, `model_config`, plus for full-hand runs `declare: {model,
optimizer, phases_done, config}` and, when the burial policy is on,
`bury: {model, optimizer, config}` (the two `config`s from 2026-09-28
on; absent means v5). A population checkpoint holds `agents`, a list of
those per-agent parts; `read_checkpoint("….pt#k")` returns one.

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

1. ~~**Benchmark `v5-bury/final.pt`**~~ Done 2026-09-28 (§0): the learned
   burial matches the heuristic and beats the climb, so burial is done
   for now. If burial is revisited, its gain over the heuristic (+0.005
   ± 0.011) is not yet significant. The options, if a later run shows
   it exploiting critic error, remain:
   - return = the hand's actual outcome, or a mix of outcome and critic;
   - a KL penalty toward the heuristic;
   - more random burials.

   The critic reward was the user's explicit choice, so **propose the
   change; don't switch it silently**.

   Cheap and not yet measured: v5 final with the learned burial and
   search (`--pre-play agent-policy --search-worlds 8`, ~30 min a bot).
2. **Population of 8: built and smoke-tested 2026-09-28; the real run is
   not launched.**
   The user's spec:
   - K = 8 randomly initialised agents;
   - one agent per seat, so partners can be different agents;
   - two tables;
   - seats reshuffled every **10** steps;
   - **4096 hands a step**, so each agent gets 2048 seat-trajectories a
     step, half the single agent's;
   - landlord-set hands only.

   Design and the reasoning: README § *A population of eight*. Code:
   `train/population.py`, tests `tests/test_population.py`.

   **The smoke run** (`pop8-smoke`, 2026-09-28, 4 steps, ~10 min on an
   L40S) ran clean: ~102 s a play step from random initialisation, 12–15 s
   a pre-play step, 4.1 GB peak GPU memory, 28.7 GB host, 1.1 GB a
   checkpoint, every agent on 44–45k rows a step. The step is long
   because untrained hands run ~88 decisions against v5's ~60; per
   decision it costs what v5's did, so expect ~72 s once hands shorten.
   Numbers in README § *A population of eight*. The command, for a rerun:

   ```bash
   # from research/; the app name keeps it apart from any live training run
   TRACTOR_TRAIN_APP=tractor-train-pop-smoke modal run --detach modal_train.py::launch \
     --gpu l40s --name pop8-smoke --population 8 --reshuffle-every 10 --hands 4096 \
     --steps 4 --warmup-steps 2 --play-phase-steps 2 --declare-phase-steps 2 \
     --declare-hands 8192 --bury-policy --random-burial-frac 0.2 --length-buckets 4 \
     --checkpoint-every 2 --keep-last 2
   ```

   **pop8 uses the v6 architecture** (`--arch v6`, the user's choice on
   2026-09-28): SwiGLU everywhere at the same hidden widths, the card
   critic, and the widened declaration and burial networks. The user
   asked for a bake-off first; the card critic won it clearly (held-out
   explained variance 0.368 against 0.161–0.213 for the flat critic,
   paired +0.21 overall and +0.23 at play start; a widened flat critic
   gained nothing). README § *The v6 architecture, and the critic
   bake-off*. The critic's width then came down to 128 at the user's
   request (as good as 256 in a width bake-off). The v6 smoke runs
   (`pop8v6-smoke` at width 256, `pop8v6b-smoke` at 128 with 10 workers)
   ran clean at ~125–140 s a play step and ~26 s a pre-play step, 8.7 GB
   GPU; where the time goes and what was bought back is in README §
   *The critic's width, and where a v6 step goes*. Add `--arch v6
   --rollout-workers 10` to the smoke command above to rerun it.

   **Then the critic became shared and the step was optimised** (the
   user's calls, 2026-09-28): one critic for all eight agents, the
   rollout overlapped, the update compiled. A step is ~80 s where it was
   ~135 s (README § *A shared critic, and a step that costs 80 s*;
   `pop8v6c-smoke`). **Then ~63 s** (same day, README § *Keeping the
   round on the GPU, and a 63 s step*; `pop8v6d-smoke`): the round kept
   on the GPU for the updates with no read-backs a minibatch, the
   rollout's forwards compiled (both under `--compile`), the burial
   update in bf16; a pre-play step ~18 s where it was ~26 s.

   **Then the shared critic got its own history encoder** (the user's
   call, 2026-09-29; README § *What reading raw history costs the shared
   critic*): ~80 s a play step on the L40S, ~44 s on the H100 with CUDA
   graphs, where they were ~63 and ~36.

   **The proposed run** (the warm-up length and step count are proposals,
   not the user's decisions yet). A Modal function stops at 24 hours. At
   ~80 s a step, 300 warm-up steps are ~6.5 h and a 200-step block ~4.4 h
   plus ~30 min of pre-play, so a first launch fits about 900 steps
   (~21.7 h with the compiling; hands shorten as the agents learn, so
   likely less). Resume with `--resume /vol/pop8/checkpoints/final.pt`,
   the same flags and a larger absolute `--steps` (confirm the first line
   reads `[901/...]`):

   ```bash
   modal run --detach modal_train.py::launch --gpu l40s --name pop8 --arch v6 \
     --rollout-workers 10 --rollout-overlap --pool 4096 --compile \
     --population 8 --reshuffle-every 10 --hands 4096 --steps 900 \
     --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 \
     --declare-hands 8192 --bury-policy --random-burial-frac 0.2 \
     --length-buckets 4 --checkpoint-every 50 --keep-last 6
   ```

   **Or on an H100** (the user's call -- about 1.65x faster a step for
   about 1.2x the GPU cost an hour-for-hour; README § *The update as a
   CUDA graph*; `pop8v6f-h100-smoke`): ~44 s a play step and ~15 s a
   pre-play step, so a first launch fits about 1500 steps (~21.4 h).
   `--cuda-graphs` is what makes the H100 pay; 16 workers need the
   H100 function's 20 CPUs. Resume as above (`[1501/...]`):

   ```bash
   modal run --detach modal_train.py::launch --gpu h100 --name pop8 --arch v6 \
     --rollout-workers 16 --rollout-overlap --pool 4096 --compile --cuda-graphs \
     --population 8 --reshuffle-every 10 --hands 4096 --steps 1500 \
     --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 \
     --declare-hands 8192 --bury-policy --random-burial-frac 0.2 \
     --length-buckets 4 --checkpoint-every 50 --keep-last 6
   ```

   **`pop8` failed; do not resume it.** Launched 2026-09-29 with
   `--compile --cuda-graphs`. Every agent diverged in turn from step 36
   (approx_kl 0.006 -> 1e8, logits ~30 -> ~1e4), and the run died at step
   300 when its idle rollout workers timed out during the first pre-play
   phase (fixed, `96ad51a`). The cause is `torch.compile` of the update:
   README § *Why pop8 diverged*.

   **`pop8b`** (relaunched 2026-09-29, the user's call): the same recipe
   without compiling the update --

   ```bash
   TRACTOR_TRAIN_APP=tractor-train-pop8b modal run --detach modal_train.py::launch \
     --gpu h100 --name pop8b --arch v6 --rollout-workers 16 --rollout-overlap --pool 4096 \
     --fixed-shapes --cuda-graphs --rollout-forwards compiled \
     --population 8 --reshuffle-every 10 --hands 4096 --steps 1300 \
     --warmup-steps 300 --play-phase-steps 200 --declare-phase-steps 100 \
     --declare-hands 8192 --bury-policy --random-burial-frac 0.2 \
     --length-buckets 4 --checkpoint-every 50 --keep-last 6
   ```

   ~51 s a play step, so 1300 steps end inside the 24-hour limit. Do
   not add `--compile` back without KL guarding; see the README.

   It ran to step 934, KL-spiking in one or two agents at a time from
   step 150 on (README § *pop8b's KL spikes, and the guard*). At 04:16
   on 2026-09-30 Modal lost its worker and re-ran the call from step 0
   into the same directory, overwriting checkpoints by step number;
   stopped at the rerun's step 752. The first run's steps 150, 250,
   350, 500, 750, 800, 850 and 900 are in `/vol/pop8b/milestones/`;
   `metrics.jsonl` holds its 900 rows, then the rerun's 750.
   `tractor.train.relaunch` now makes a rescheduled call resume from
   its run's newest checkpoint, and refuses a fresh start into a run
   that has checkpoints.

   **`pop8c` is the run** (2026-09-30, the user's call): `pop8b`'s step
   900 onward, with the KL guard --

   ```bash
   TRACTOR_TRAIN_APP=tractor-train-pop8c modal run --detach modal_train.py::launch \
     --name pop8c --resume /vol/pop8b/milestones/step_0000900.pt --target-kl 0.05 \
     <the same flags as pop8b>
   ```

   400 steps, ~5.5 h with two declare phases; milestones in
   `/vol/pop8c/milestones/` (`::milestones --name pop8c`). It reached
   step 1300 on 2026-09-30 (README § *pop8b's KL spikes, and the
   guard*; ladder evals in `evals/2026-09-30/`: best agent +2.9 points
   with random pre-play, +0.9 with its own).

   **Continued** 2026-10-01 00:51 EDT (the user's call: +1500 steps, the
   guard loosened to 0.15 since 0.05 bound nearly every update), app
   `tractor-train-pop8c-2`, ~21 h:

   ```bash
   TRACTOR_TRAIN_APP=tractor-train-pop8c-2 modal run --detach modal_train.py::launch \n     --name pop8c --resume latest --steps 2800 --target-kl 0.15 <the same flags as pop8b>
   TRACTOR_TRAIN_APP=tractor-milestones-pop8c-2 modal run --detach \n     modal_train.py::milestones --name pop8c
   ```

   **Training longer.** Resume with a larger absolute `--steps` and the
   same flags; the step, the schedule's place and every optimiser come
   back from the checkpoint, and the deals continue on a stream of their
   own (seeded by the step). `--resume latest` takes the run's newest
   checkpoint -- `final.pt`, or the last `step_*.pt` if the 24-hour limit
   cut the run off. Use a new app name for each launch:

   ```bash
   TRACTOR_TRAIN_APP=tractor-train-pop8c-2 modal run --detach modal_train.py::launch \
     --gpu h100 --name pop8c --resume latest --steps 3000 --target-kl 0.05 <the same flags>
   ```

   **Milestones.** `--keep-last 6` keeps only the newest 300 steps of
   checkpoints, too few for a strength-over-steps curve, so a small job
   beside the run copies every 250th (and step 150) to
   `/vol/pop8b/milestones/`, each once the next checkpoint exists, and
   ends at `final.pt` (running since 2026-09-29; relaunch it with a
   resumed run):

   ```bash
   TRACTOR_TRAIN_APP=tractor-milestones-pop8b modal run --detach \
     modal_train.py::milestones --name pop8b --extra 150
   ```

   **Evaluating it.** Every agent against the whole ladder -- the four
   fair bots, 770 deals each, the full hand as trained -- with win rate,
   levels and points per bot and pooled, and the agents ranked (about
   1.5 h on up to eight A10Gs). It runs on Modal end to end
   (`population_job`, since 2026-09-30): the command returns at once, the
   record and ranking land in `/vol/evals/runs/<out>` and
   `<out>.summary.json` (fetch with `modal volume get`), failed pairs are
   retried, and a rerun with the same `--out` resumes. `--no-remote` is
   the old client-side loop, which a sleeping laptop or a memory kill
   cancels:

   ```bash
   TRACTOR_EVAL_APP=tractor-eval-pop8c modal run --detach modal_eval.py::population \
     --checkpoint /vol/pop8c/checkpoints/final.pt --out pop8c-final-policy.jsonl
   modal volume get tractor-rl /evals/runs/pop8c-final-policy.jsonl evals/2026-09-30/
   # a milestone the same way: --checkpoint /vol/pop8b/milestones/step_0001000.pt
   # (step 150: evals/2026-09-29/pop8b-150-random.jsonl, --pre-play random
   # --gpu, since declaring and burial train only from step 300; best agent
   # -3.8 points over the ladder, worst -18.0, where v5 final is +10.5 on
   # the same deals. --gpu because 32-core CPU containers sat unscheduled
   # for 20 minutes on 2026-09-29. Step 500, random: best -4.7, mean -9.0
   # against 150's -10.0 -- nearly flat; full hand 1-3 points worse than
   # random, the learned declaring and burial weaker than the bots'.)
   # --pre-play random for play alone; pair the best agents against
   # evals/2026-09-24/v5-final-{policy,random}.jsonl with tab_evals.py --paired
   ```
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
