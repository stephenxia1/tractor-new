# Tractor RL research stack

Standalone self-play RL for the 4-player, 2-deck Tractor variant defined in
[`rules.md`](../rules.md). Independent of the web app: it consumes
`@tractor/engine` as a library and nothing in `apps/` or `packages/` depends
on it.

The agent currently learns **in-trick play only**. Trump declaration and
kitty burial are delegated to the `smart-v34` heuristic.

---

## Why it is split TypeScript / Python

The rules already exist, tested, in TypeScript, and the ML ecosystem is in
Python. Rather than port either one, the boundary sits between them:

```
Node worker (N games in flight)  --binary frames-->  Python learner
   engine + candidate generation                     features + Q-net + DMC
                                 <----action idx----
```

* The **network never leaves Python**, so there is no weight export, no
  ONNX, no second forward-pass implementation to keep in sync.
* The **rules never leave TypeScript**, so there is no engine port and no
  parity-drift risk (`plan.md` calls that the project's biggest risk).
* Frames carry *raw game facts* — 54-slot card-count planes and small
  integers — not features. Every modelling decision lives in
  `py/tractor_rl/features.py`, so changing the representation is a
  Python-only edit.

Measured on the dev laptop (i7-1270P, 16 threads, no GPU):

| Component | Throughput |
| --- | --- |
| TS actor (engine + candidates + encoding) | ~9k decisions/s per worker |
| PyTorch inference, 0.74M params, K≈8 candidates | 8–10k decisions/s |
| PyTorch training step | ~3.5k samples/s |
| numpy featurisation | ~12k decisions/s in-loop |
| Wire volume | ~0.7 KB/decision |
| **End-to-end training loop** | **~4k decisions/s, ~56 hands/s** (tuned; see RESULTS.md) |

The network, not TypeScript, is the bottleneck; on a GPU box the actors
become the constraint and scale by adding workers.

---

## Layout

```
src/                     TypeScript: environment only
  cards.ts               54-slot card index
  record.ts              wire format + ByteWriter; source of truth for spec.json
  candidates.ts          fair-play candidate generation
  encode.ts              decision + episode serialisation
  env.ts                 one game slot: hands end to end, pausing at learner decisions
  worker.ts              N games per process, batched stdio protocol
  selftest.ts            random-policy run with wire-format invariants
  spec.ts                regenerates spec.json
spec.json                layout mirror, asserted by py/tractor_rl/layout.py
py/tractor_rl/
  layout.py              spec.json loader + version check
  features.py            records -> tensors (trump anchoring, unseen pool, scalars)
  protocol.py            worker pool, frame protocol
  model.py               Q-net (shared trunk, per-candidate scoring)
  dmc.py                 replay buffer, Monte-Carlo returns, learner
py/train_dmc.py          training loop
py/smoke_protocol.py     random-policy throughput/validity check
checkpoints/  runs/      artifacts (gitignored)
```

## Running

```sh
# from research/
npm run selftest                       # environment + wire format
npm run spec                           # regenerate spec.json after editing record.ts

# Python side needs numpy + torch (CPU is fine). On this machine:
#   /c/Users/xiast/anaconda3/envs/tractor/python.exe
PYTHONPATH=py python py/smoke_protocol.py --workers 4 --games 96 --seconds 20
PYTHONPATH=py python py/train_dmc.py --workers 4 --games 96 --minutes 60
```

Checkpoints land in `checkpoints/<run>_latest.pt`; per-round metrics stream
to `runs/<run>.jsonl`. The v2 trainers — `py/train2.py` and
`py/train_pop.py` — also take `--wandb`, which mirrors those same rows to
Weights & Biases (`--wandb-mode offline` needs no key); the JSONL stays the
source of truth.

---

## Running in the cloud

The laptop runs this loop at **base clock** under sustained load: 70-second
benchmarks report ~4.1k decisions/s, but an hour-long run settles at
~1.9k. A server CPU (or any box that holds its clocks) is worth roughly 2x
before any extra cores are counted.

Two ways to stand a box up, both from the `rl-research` branch:

```sh
# 1. Bare VM (Ubuntu/Debian), leaves training running after you disconnect
bash research/scripts/bootstrap_vm.sh --workers 24 --games 64 --threads 12 --minutes 1440

# 2. Container (build from the repo root, not from research/)
docker build -f research/Dockerfile -t tractor-train .
docker run --rm -v $PWD/out:/app/research/checkpoints tractor-train     --workers 12 --games 64 --threads 8 --groups 2 --minutes 600
```

Both verify the stack (`dupcheck`, `selftest`, return tests) before
starting, so a broken image fails in seconds rather than after an hour.
`bootstrap_vm.sh` switches to `--device cuda` automatically when torch sees
a GPU. `research/scripts/publish_progress.sh <run>` pushes metrics and the
latest checkpoint to a `training-progress` branch so another machine can
pull them.

A third way, and the one to use for anything that is a *set* of arms rather
than one long run:

```sh
# one container per arm, launched together, artifacts on a shared volume
modal run --detach research/modal_app.py::sweep \
    --run-prefix spt --minutes 300 --size a10g \
    --arms "one:--steps-per-train 1;two:--steps-per-train 2"
```

See [MODAL.md](MODAL.md). It answers A/B questions in one arm's wall clock
instead of the sum, puts every arm on the same declared shape, and resumes
itself after a preemption. The AWS path stays for long single-agent runs,
where a spot box is cheaper per core-hour.

**Sizing.** The tuned loop splits roughly 50% learner / 40% acting forward
pass / 6% featurisation / 2% environment, so it is torch-bound, and a GPU
is worth having. But the Node workers need about a core each to feed it:
prefer ~16 vCPU alongside a modest GPU over 8 vCPU alongside a large one.
Do not port the thread settings — on any new box run
`python3 py/sweep_throughput.py`, because the actor/torch thread split was
worth 3x here and is machine-specific.

**What to expect from a GPU.** Two things currently sized for a CPU will
need attention before one is saturated: the per-round batch (~384
decisions, ~2,700 candidate rows) is small, so raise games-in-flight; and
featurisation is numpy on the host, which is 6% of the loop now but becomes
the ceiling once the forward pass gets 10x faster. The wire format ships
raw bytes precisely so featurisation can move onto the device.

---

## Design notes

**Episodes are single hands.** Landlord seat and both team levels are
randomised per hand, so training sees all 13 trump ranks and all four
landlord positions instead of the narrow slice a single game sequence
visits.

**Fair play is enforced structurally.** The engine's `allLegalPlayActions`
runs the §7.6 throw-safety check against the other seats' actual hands,
which would leak hidden information into the *action set*. The research
layer passes `includeThrows: false` and generates throws from public
information only: a component is offered only when no unseen card can beat
it. That is a guarantee rather than an estimate, and `GameSlot.choose`
asserts the engine never fires the forced-lead penalty on a learner move —
if the derivation were wrong, training would silently learn from an action
that was never played.

**One network, not three.** The earlier attempt in
`packages/engine/src/rl/dmc` used a separate Q-net per role; here role is an
input feature, so all three roles share a representation and every sample
trains every parameter.

**Trump anchoring.** Card slots are permuted so the trump suit always
occupies the first 13 positions. One "trump" representation transfers
across all 13 trump ranks. No-trump hands (joker declarations — ~20% of
hands under smart declaration, and *not* rare enough to ignore) use the
identity permutation and are flagged.

**Reward** (from the acting seat's point of view): per-trick attacker point
captures, positive for attacking seats and negative for defending seats;
the kitty bonus attributed to the last trick; plus a terminal term for the
level swing, because the 80-point cliff in §8.3 makes raw points a
non-linear objective.

---

## Status

Working end to end and measured — see [`RESULTS.md`](./RESULTS.md) for every
number, including how the evaluation harness was validated before any agent
result was believed.

- **Duplicate-deal arena** (`py/arena.py`): each deal played twice with the
  teams swapped, scored on point and level margins with 95% CIs. Verified
  by a null test (same bot both sides ⇒ exactly 0.00 on 96/96 pairs) and by
  reproducing known gaps (`smart` vs `random` +84 pts).
- **Deterministic deals** (`src/dupcheck.ts`): 3,888 cards compared
  card-for-card across paired runs, plus negative controls.
- **Trained agent**: a 20-minute pilot moved the net from −17.4 (untrained,
  worse than random) to **+62.4 pts vs `random`** over 252 pairs.
- **Throughput**: tuned from 1.4k to ~4.1k decisions/s. The dominant term
  was torch thread count starving the Node actors, not the loop structure.

## Measurements on the PPO track (`py/tractor/`)

All reproducible: `modal run modal_train.py::critic`,
`::critic_hands`, `::rollout`.

### The reward is trick points, and the terminal term is off

The learner's `explained_variance` is computed on the rows the update
is about to train on, so it says how well the behaviour critic fit
returns it had already seen. `py/tractor/train/critic_diag.py` asks the
other question: keep a quarter of the *hands* out of the fit, run the
real update on the rest, score both halves. Hands, not rows — the
critic reads a privileged view of all four, so a seat held out of a
fitted hand is not held out at all.

256 hands, 4 epochs, `gamma = 1`, against the step-2500 checkpoint:

| `level_weight` | fitted | held out | gap | slope |
| --- | --- | --- | --- | --- |
| 0.00 | +0.842 | +0.405 | 0.437 | 0.71 |
| 0.25 | +0.903 | +0.167 | 0.736 | 0.59 |
| 0.50 | +0.919 | −0.105 | 1.025 | 0.45 |
| 0.75 | +0.939 | −0.196 | 1.135 | 0.40 |
| 1.00 | +0.941 | −0.324 | 1.265 | 0.34 |

Negative is worse than predicting the mean, so past the middle of the
anneal the value function was adding variance to the policy gradient
instead of removing it. The mechanism is effective sample size: with
`gamma = 1` and a terminal level reward every row of a hand carries the
same terminal term, so as weight moves onto it the returns within a
hand converge and ~11,000 rows collapse towards ~512 targets.

Two candidate remedies do nothing, measured at `level_weight = 1.0` on
the same rollouts so the arms are paired: `gamma = 0.99` gives −0.328
and `detach_critic = True` gives −0.320, against a −0.324 baseline. A
hand is about 22 decisions per seat, so that discount barely separates
rows sharing a terminal term, and the memorisation is in the critic
head reading the privileged view rather than in the shared encoder.

So the reward is trick points only: `anneal_steps` and
`final_level_weight` both default to 0, which is the regime the table
above shows the critic generalising in. Set them to restore the
annealed hand-outcome reward.

Two things the points-only reward cannot see. The kitty bonus is folded
into the attackers' total by `PlayState._finish` and was never in a
trick record, so no dense credit exists for it — and it is up to eight
times the kitty's points. And the game is scored on levels: 119 points
and 0 points are the same result, and the dense reward rates them 119
apart. The ladder win rate is where that proxy would show itself
failing.

### One epoch, and 1024 hands a round

Epoch count is the only lever that moved the anneal number (−0.324 →
−0.226 → +0.030 for 4, 2, 1), and the penalty shrinks but does not
vanish on the points-only reward. Paired on identical hands:

| hands | epochs | fitted | held out | gap | slope |
| --- | --- | --- | --- | --- | --- |
| 256 | 1 | +0.618 | +0.404 | 0.214 | 0.73 |
| 256 | 4 | +0.846 | +0.407 | 0.438 | 0.71 |
| 1024 | 1 | +0.615 | **+0.473** | **0.142** | **0.95** |
| 1024 | 4 | +0.826 | +0.424 | 0.402 | 0.79 |

What the critic actually wants is independent targets, and a hand is
about one target per seat however many rows it writes. Widening the
round at one epoch, points-only:

| hands | targets | fitted | held out | 95% CI | gap | slope |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 384 | +0.481 | +0.023 | [−0.435, +0.312] | 0.459 | 0.51 |
| 256 | 768 | +0.618 | +0.404 | [+0.213, +0.539] | 0.214 | 0.73 |
| 512 | 1536 | +0.640 | +0.378 | [+0.267, +0.465] | 0.262 | 0.82 |
| 1024 | 3072 | +0.615 | +0.473 | [+0.406, +0.525] | 0.142 | 0.95 |
| 2048 | 6144 | +0.636 | +0.479 | [+0.438, +0.516] | 0.157 | 0.90 |

The rise stops at 1024, where the calibration slope also peaks at 0.95;
2048 buys +0.006 for twice the rollout. `launch` therefore defaults to
`--hands 1024 --epochs 1`. Note that the rollout cost is exactly linear
in hands (7.0, 13.3, 25.8, 52.3, 102.6 seconds) and that the forward
batch is set by `pool`, not by round width, so past this point a wider
round buys neither throughput nor statistics.

### The rollout was using one of eight reserved cores

Span breakdown of a serial round on an A10G (128 hands, bf16):
enumerate 2.20 s, encode 1.54 s, simulator step 1.10 s, network forward
1.91 s. About two thirds is host work in pure Python, where threads
cannot help because the enumerator holds the GIL.

`py/tractor/train/parallel.py` moves the simulator into worker
processes and keeps one batched forward in the parent. Two measurements
shaped it.

*The observations never travel through a queue.* `py/bench_ipc.py` puts
a `multiprocessing.Queue` carrying the 28,466-byte observation per
decision at ~1,150 decisions/s — slower than the loop it replaces —
against ~23,600 for a shared-memory ring with the queue carrying a row
index, at a batch of 32. So the buffer *is* the transport
(`py/tractor/train/shared.py`) and the queues carry tags and integers.

*The workers are persistent.* A spawned worker takes ~9.5 s to reach
its first instruction, nearly all of it `import torch`. The first
version spawned per round and measured 1.08x, with the entire gain
inside the start-up. `RolloutPool` starts its workers once and a round
is a message.

| hands | serial | x4 | x7 | x11 | x15 |
| --- | --- | --- | --- | --- | --- |
| 128 | 6.16 s | 2.16x | 2.28x | — | — |
| 512 | 19.59 s | 2.08x | 2.40x | — | — |
| 1024 | 38.91 s | — | 2.47x | 2.69x | 2.81x |

Past seven workers it keeps improving and stops mattering. Fitting
`T(w) = A/w + B` to the 1024-hand row gives `A = 24.3`, `B = 12.25`,
and that model reproduces the eleven-worker point to the second: there
is a 12.25 s floor of parent work — the forward, the sampler, the
transport — that no number of workers touches, so the ceiling is 3.18x
and fifteen workers already reach 88% of it. The floor exists because
the design is lockstep: the parent's forward does not overlap the
workers' packing. Breaking it needs two batches in flight per worker,
not more cores.

Sixteen reserved cores are not worth buying for this. They lift collect
from 15.72 s to 13.87 s at 1024 hands, which against a ~35 s update is
under 4% of the step, for about 20% more per hour. `cpu=8.0` and seven
workers is the configuration the training functions use.

### The full hand: declaration and burial, trained in alternation with play

Play used to start from `gen/episode.py`'s draw — a declaration uniform
over the legal ones, a burial uniform over the landlord's cards. The
agent now makes both decisions itself (landlord-set hands only):

- **Declaration** is its own environment, `sim/deal.py`: 100 cards dealt
  one at a time from the landlord, every card a decision for the seat
  that received it (pass, or one of 10 declarations), then a closing
  round until four seats pass. Legality matches the engine on 600
  scenarios dumped by `src/dumpDeclare.ts`. A small network
  (`nn/declare.py`) is trained by PPO (`train/declare_loop.py`) with
  play frozen; each seat's reward is its privileged critic's value at
  play start, i.e. its team's expected point margin from there.
- **Burial** (`sim/bury.py`) is a hill climb over complete burials —
  swap one buried card for one kept card, keep the best improvement —
  scored by the actor's own value head from the landlord's seat. Every
  position it scores is a legal 25-card play start, which the value head
  was trained on; removing cards one at a time would pass through
  32-card hands it has never seen.
- **The schedule** (`train/loop.py`): `warmup_steps` of play on the
  uniform draw, then blocks of `play_phase_steps`, each opened by
  `declare_phase_steps` of declaration training. A block's play starts
  come from the frozen declaration policy and the burial climb, except
  `random_burial_frac` (20%) buried at random so the value head stays
  accurate on burials the climb would not propose.

Two changes the design forced:

- **The kitty bonus is credited** with the last trick (`_credit_trick`).
  A reward blind to it rates burying every point card as free.
- **The critic reads the context vector.** It knew the trump context
  only through the card features inside history tokens, so at play
  start, with no history, it could not tell one trump suit from another
  or the landlord from an attacker — exactly the position declaration
  is scored at. The new input starts at zero, so old checkpoints load
  and compute what they did (`load_model`). On resuming v3 step 2000 its
  explained variance went from ~0.37 to ~0.55 within 16 steps.

### Where a full-hand step goes, and what was bought back

L40S, 1024 hands, v3 step 2000 weights:

| change | measured |
| --- | --- |
| history encoder runs to the batch's longest history (exact) | rollout forward 3.4 s → 2.4 s a step |
| burial rows built on the device from per-hand bases | burial 22.5 s → 17.1 s (1024 hands, every swap) |
| burial considers the 12 lowest-scoring kept types, not all | 17.1 s → ~7.4 s, keeping ~83% of the value gain (+4.03 of +4.85 pts) |
| burial forward in TF32 | ~9.7 s → ~7 s at training scale |
| `ppo.length_buckets = 4` | update 9.7 s → 7.9 s, same losses and KL |
| prefetching starts on a thread | **no gain** (GIL): step 28.2 → 26.4 s, all of it the buckets |
| critic's pooled vector computed once when no row has history (exact) | play-start critic values cost only the privileged MLP |
| burial scored by the critic over 8 deals, every swap, instead of the actor | full-hand play step 22 s → 17.6 s; declaration step ~9 s → 2.4 s |

A warm-up step went from ~19 s to ~16 s with these. The declaration
phase costs ~9–14 s a step at 1024 hands, nearly all of it burial.
`modal run modal_perf.py::update` and `modal_train.py::pipeline`
reproduce the numbers.

### What the full hand bought, at v4-full step 2400

770 deals a bot (1,540 hands, ±0.025 on a win rate), seed 20260923, so
every row shares its shuffles. Raw per-hand rows: `evals/2026-09-24/`.

**Play only** (`--pre-play random`: both sides start from the uniform
draw, so the positions are identical and the change is paired):

| bot | v3 step 2000 | v4 step 2400 | paired win change | paired points change |
| --- | --- | --- | --- | --- |
| smart | 0.518 | 0.614 | +0.096 ± 0.020 | +12.0 ± 1.4 |
| smart-v18 | 0.499 | 0.569 | +0.071 ± 0.022 | +7.8 ± 2.0 |
| smart-v28 | 0.499 | 0.570 | +0.071 ± 0.022 | +7.8 ± 2.0 |
| smart-v34 | 0.498 | 0.574 | +0.076 ± 0.022 | +8.0 ± 1.9 |

The play policy, flat from step 500 to 2000, moved +0.07 against the
fair search bots in 400 steps. What changed together: the critic's
context input, the kitty bonus in the reward, and play from the agent's
own starts. Nothing here separates them.

**The full hand** (`--pre-play agent` and its ablations; positions differ
between rows, so these are unpaired). Win rate against smart-v18 / v28 /
v34, averaged:

| pre-play | win | levels | points | win as landlord |
| --- | --- | --- | --- | --- |
| v3: the bots declare and bury for everyone | 0.510 | −0.01 | −0.5 | 0.47 |
| v3: buries by its value head, never declares | 0.480 | −0.05 | −1.8 | 0.43 |
| v4: declares, buries by the actor's value climb | 0.521 | +0.05 | +2.2 | 0.45 |
| v4: never declares, actor climb | 0.513 | +0.03 | +1.7 | 0.46 |
| v4: declares, **critic** climb over 8 deals | 0.536 | +0.08 | +3.1 | 0.48 |
| v4: declares, the heuristic the climb starts from | 0.543 | +0.10 | +4.2 | 0.50 |

- Declaring is worth about +0.01, nearly all of it as attacker (0.566 →
  0.592 when attacking).
- **The actor-scored burial climb lost to its own starting point**, by
  0.045 in the hands the network was landlord: the best of ~100 noisy
  estimates is the one the estimate overrates. Scoring with the
  privileged critic averaged over sampled deals of the unseen cards
  recovers most of it and is within noise of the heuristic; the run
  switched to it at step 2550. Value-guided burial has not yet beaten
  a plain rule.

### Where v4-full ended (step 3150)

Same protocol. **Play only**, paired against earlier checkpoints on the
same positions:

| bot | v3 step 2000 | v4 step 2400 | v4 final | final − v3 | final − 2400 |
| --- | --- | --- | --- | --- | --- |
| smart | 0.518 | 0.614 | 0.619 | +0.101 ± 0.021 | +0.005 ± 0.019 |
| smart-v18 | 0.499 | 0.569 | 0.574 | +0.075 ± 0.022 | +0.005 ± 0.021 |
| smart-v28 | 0.499 | 0.570 | 0.577 | +0.077 ± 0.022 | +0.006 ± 0.021 |
| smart-v34 | 0.498 | 0.574 | 0.571 | +0.073 ± 0.022 | −0.003 ± 0.021 |

The whole gain over v3 (+8 points a deal against the fair bots) arrived
by step 2400; the last 750 steps added nothing measurable. The policy
has plateaued again, one level higher.

**The full hand** at the final checkpoint, averaged over smart-v18 /
v28 / v34:

| configuration | win | levels | points | win as landlord | win attacking | vs smart |
| --- | --- | --- | --- | --- | --- | --- |
| declares, critic-scored burial (as trained) | 0.525 | +0.08 | +3.7 | 0.454 | 0.596 | 0.706 |
| declares, heuristic burial | 0.553 | +0.11 | +4.8 | 0.510 | 0.596 | 0.718 |
| declares, critic-scored burial, + search (8 worlds, top 3) | **0.566** | **+0.17** | **+6.7** | 0.507 | 0.625 | **0.729** |

- **Value-guided burial still loses to the heuristic**, and by more at
  the end than at step 2400: 0.056 in the hands the network is
  landlord. Six hundred steps of training on its own burials did not
  close it. What a new session should try: a trust region around the
  heuristic (swap only when the estimated gain exceeds the estimate's
  own noise), more deals per candidate, or a burial *policy* trained
  from play outcomes rather than a climb on a value estimate.
- **Search adds about +0.04** on top of the full hand, as it did on the
  play-only policy.
- The network declares in 61% of hands; the bots in the rest.

**Deep search** (v3 step 2000, play only, 16 worlds, top 4, three
tricks past the current one before the critic): 0.575 / 0.556 / 0.555
against smart / v18 / v28, paired +0.056 over no search and +0.02 over
the shallow search (±0.022), at about 7× the cost — five hours a bot
at 770 deals. The v34 pair hit the 8-hour container limit.

---

## Next steps

1. **Longer training**, then evaluate against the real opponents. Budget
   matters: `smart-v18`/`v34` cost ~70 s/hand to play against (~50
   hands/hour/worker), so a 200-pair match is an ~8 worker-hour job.
   `smart-v13` at 12 s/hand is the affordable middle rung.
2. **Faster featurisation / GPU path**. `--device cuda` is wired but
   untested; featurisation would need to move onto the device with it.
3. **A population of 8** independently initialised agents, one per seat
   at two tables, reseated at intervals, in place of pure self-play.
   Learned pre-play (above) was the prerequisite.
4. **Bot adapter** so a trained checkpoint can fill a seat in the web app.
