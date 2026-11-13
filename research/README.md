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

### Where v5-bury ended (step 4150)

The burial is a policy now, `nn/bury_net.py`: the landlord picks eight
cards one at a time from the 33 it holds, scored by a set encoder over
the 54 card types with every type not held masked out. Every pick shares one
return, the privileged critic's value at the play start the burial
produced, from the landlord's seat, ×10 (`train/bury_loop.py`). It
trains in the pre-play phase on the declaration's hands with play
frozen, sampling in training and taking the argmax in evaluation. It
costs about 0.1 s per 1024 hands, where the critic climb cost about 7 s.

`v5-bury` resumed v4-full final and ran five blocks of 100 pre-play
steps and 200 play steps, with a fifth of play starts still buried at
random. The burial return, averaged over each pre-play phase:

| phase | from step | burial return | entropy | kitty points buried |
| --- | --- | --- | --- | --- |
| 0 | 3150 | +0.625 | 1.99 | 2.2 |
| 1 | 3350 | +0.917 | 1.41 | 2.8 |
| 2 | 3550 | +0.970 | 1.27 | 1.9 |
| 3 | 3750 | +0.921 | 1.22 | 2.1 |
| 4 | 3950 | +0.925 | 1.18 | 2.0 |

×40 turns the return into points of expected margin, so the landlord's
expected margin rose from +25 to +37 and was flat from the third phase.
That is the critic grading the policy trained to please it, so it
cannot show the burial is better. The benchmarks below can.

Same protocol, 770 deals a bot, seed 20260923. Averaged over smart-v18 /
v28 / v34 (±0.014 on the win rate):

| configuration | win | levels | points | win as landlord | win attacking | vs smart |
| --- | --- | --- | --- | --- | --- | --- |
| declares, **learned burial** (as trained) | **0.566** | **+0.17** | **+7.1** | **0.523** | 0.608 | 0.719 |
| declares, heuristic burial | 0.561 | +0.15 | +6.3 | 0.513 | 0.608 | 0.713 |
| declares, critic-scored climb (v4's method) | 0.552 | +0.16 | +6.5 | 0.495 | 0.608 | 0.725 |
| play only (`--pre-play random`) | 0.580 | +0.23 | +9.2 | 0.322 | 0.838 | 0.632 |

Paired on the same deals, pooled over the three search bots:

| comparison | win change | points change |
| --- | --- | --- |
| v5: learned burial − heuristic burial | +0.005 ± 0.011 | +0.80 ± 0.57 |
| v5: learned burial − critic climb | **+0.014 ± 0.010** | +0.60 ± 0.53 |
| v5 critic climb − v4 critic climb | +0.027 ± 0.014 | +2.85 ± 0.98 |
| v5 heuristic burial − v4 heuristic burial | +0.008 ± 0.014 | +1.56 ± 0.90 |
| play only: v5 final − v4 final | +0.006 ± 0.012 | +0.51 ± 1.02 |
| v5 learned burial − v4 critic climb + search | −0.000 ± 0.015 | +0.39 ± 0.94 |

- **Value-guided burial has now caught the heuristic.** The learned
  policy matches the heuristic and edges it (0.523 against 0.513 as
  landlord, not paired: the result files do not record each hand's
  side), and it beats the climb it replaced. Its points gain over the
  heuristic, +0.8 a deal, is at the edge of its interval.
- **It buries points.** The policy puts 2.1 points in the kitty on
  average against the heuristic's 0.04 (the bots bury 9.6), and still
  concedes fewer (76.1 a deal against 77.8, versus smart-v18). It
  trades a few kitty points for the shape of the hand it keeps.
- **Play did not move.** Play only, v5 is +0.006 over v4 final, inside
  the noise: the policy is still on the plateau v4 reached at step 2400.
  Most of the v5 − v4 gain under the critic climb is then likely the
  critic scoring burials better: under the heuristic burial, where only
  declaration and play changed, v5 gained +0.008.
- **The learned burial without search equals v4 with search.** Search
  on top of v5 is not measured yet; it added about +0.04 every time it
  was measured before.
- The network declares in 60% of hands, as v4 did. In training, a third
  of its declarations are no-trump, against about a fifth under the
  bots' declarations.

### A population of eight (built and smoke-tested 2026-09-28)

The single agent's partner is always itself. `train/population.py`
trains eight independently initialised agents instead, each a whole
single agent: its own play network and privileged critic, declaration
network and burial policy, each with its own optimiser. They sit one to
a seat at two tables, so a partner and both opponents are other agents,
and are reseated every 10 steps. Landlord-set hands only. These are the
user's choices: eight agents, random initialisation, two tables, the
10-step reshuffle, and 4096 hands a step.

- **Routing, not a new rollout.** Hand `i` is played at table `i % 2`;
  the table and the seat name the agent. That table, int64[hands, 4],
  goes to `rollout.collect` / `RolloutPool.collect` and to
  `make_starts`, which route every decision to its seat's network,
  every value to its seat's critic, every burial to the landlord's
  agent and every declaration return to the declaring seat's critic.
  The workers are unchanged; the parent groups a lockstep batch by
  agent and runs one forward per agent. GAE already groups by hand and
  seat, and each agent's learner then updates on its own rows.
  With one agent the routed path is the old one exactly: same batches,
  same random draws (`tests/test_population.py`).
- **Sample size.** An agent sits in one seat of half the hands, so it
  learns from an eighth of the seat-trajectories the single agent took
  from the same round. At 4096 hands a step each agent gets 2048, half
  what the single agent's 1024-hand round gave it. The pre-play phase
  needs 8192 hands a step to give each agent the declarations and
  burials a 1024-hand phase gave the single agent (each agent is
  landlord in about 1024 of them).
- **Seatings** are a function of `(seed, epoch)`, not of a live
  generator, so a resumed run seats every agent where the original
  would have. Pre-play phases draw their own seatings on the same
  10-step schedule.
- **Checkpoints** hold all eight agents (about 1 GB with optimiser
  state). `read_checkpoint("…/final.pt#3")` returns agent 3 in the
  single agent's layout, and `modal_train._load_net`,
  `modal_eval.py` and `eval/bots.py` all read through it. So an agent
  is benchmarked with the usual command and `--checkpoints
  /vol/<run>/checkpoints/final.pt#3`.
- In-loop evaluation is off for a population: it measures one network.

**Measured on the smoke run** (`pop8-smoke`, L40S, 4096 hands a step,
four steps from random initialisation, 2026-09-28):

| | step | collect (forward) | update | decisions | per hand |
| --- | --- | --- | --- | --- | --- |
| warm-up step 2 | 101.7 s | 46.6 s (15.2 s) | 54.5 s | 362,756 | 88.6 |
| full-hand step 4 | 102.9 s | 46.9 s (15.0 s) | 52.5 s | 354,368 | 86.5 |
| v5, for scale (1024 hands) | 18.2 s | 8.5 s (2.2 s) | 8.9 s | 61,614 | 60.2 |

- Per decision, a population step costs what a v5 step did: the update
  0.148 ms against v5's 0.144, the collect 0.132 ms against 0.138. The
  step is long because an untrained agent plays ~88 decisions a hand
  against v5's ~60. At v5's hand length the same step would be about
  72 s.
- Splitting the forward eight ways costs about 17% (0.042 ms a decision
  against 0.036); `pool` 2048 gives a mean lockstep batch of ~1,500, so
  ~190 rows an agent.
- A pre-play step at 8192 hands takes 12–15 s; building the play starts
  for 4096 hands about 3 s.
- Peak GPU memory 4.1 GB of 48; host RSS 28.7 GB of 64. A checkpoint is
  1.1 GB.
- Every agent trained on 44–45k rows a step.

### The v6 architecture, and the critic bake-off (2026-09-28)

The user asked for SwiGLU in place of every GELU and for bigger critic,
declaration and burial networks. `nn/arch.py` names the set `v6`; every
config still defaults to `v5`, and fixtures pin the v5 networks'
layouts and outputs to the bit (`tests/test_arch.py`).

| | v5 | v6 |
| --- | --- | --- |
| feed-forwards and heads | GELU | SwiGLU at the same hidden width |
| actor | 10.68M | 13.27M |
| critic | flat: history transformer (128 × 2) + MLP over raw counts, 0.48M | `CardCritic`: one transformer (256 × 4, ff 1024) over CLS, context, 54 card tokens and the history, 4.51M |
| declaration | MLP 2 × 256, 0.27M | residual SwiGLU MLP 3 × 512, 5.0M |
| burial | transformer 128 × 2 (ReLU), 0.42M | SwiGLU blocks 256 × 4, value head 512, 3.6M |

The card critic's token for each card type carries the card's static
features in this trump context -- the block the actor reads -- and its
count in each hand, the kitty and each seat's play to the current
trick. At play start, where the flat critic is an MLP over a
270-vector of counts, that value is the reward for every declaration
and burial.

**Bake-off** (`train/critic_bakeoff.py`, `modal run
modal_train.py::critic_bakeoff`, L40S): 8 rounds of 1024 hands played by
v5-bury final from its own declarations and burials; the last round's
1024 hands held out; targets the Monte-Carlo return. Each arm trains
from scratch on the same 426,296 rows in the same order (AdamW 3e-4,
batch 256) reading v5's frozen history encodings; `v5` is the
checkpoint's own trained critic, for reference. Explained variance on
the held-out hands, every row and at play start (95% intervals
resample hands):

| arm | params | pass | fit | held out | at play start | s/pass |
| --- | --- | --- | --- | --- | --- | --- |
| v5, trained over the whole run | 0.48M | – | 0.425 | 0.404 [0.360, 0.443] | 0.400 [0.353, 0.440] | – |
| flat, fresh | 0.48M | 1 | 0.313 | 0.213 [0.174, 0.249] | 0.227 [0.188, 0.262] | 12.9 |
| flat, fresh | 0.48M | 4 | 0.643 | 0.161 [0.109, 0.212] | 0.135 [0.070, 0.197] | 12.9 |
| flat widened to 256 × 4 | 3.71M | 1 | 0.302 | 0.212 [0.174, 0.245] | 0.223 [0.185, 0.259] | 29.3 |
| flat widened to 256 × 4 | 3.71M | 4 | 0.697 | 0.180 [0.124, 0.230] | 0.145 [0.081, 0.205] | 29.4 |
| **cards** | 4.51M | 1 | 0.374 | 0.338 [0.303, 0.370] | 0.340 [0.296, 0.384] | 39.5 |
| **cards** | 4.51M | 4 | 0.501 | **0.368** [0.330, 0.403] | **0.364** [0.317, 0.411] | 39.6 |

Paired on the same held-out hands, cards minus flat at pass 4: **+0.206
[+0.168, +0.250]** on every row and **+0.229 [+0.178, +0.283]** at play
start; at pass 1, +0.125 and +0.112. Widened flat minus flat is within
±0.025 at every pass.

- **The card critic generalises; the flat one memorises.** From the same
  data the flat critic peaks after one pass and then declines on unseen
  hands while its fit keeps climbing (0.64 fitted, 0.16 held out, slope
  0.66). The card critic holds or improves on unseen hands through four
  passes (0.338, 0.352, 0.369, 0.368), its gap is 0.13 against 0.48,
  and it stays calibrated (slope 0.94–0.98).
- **Size is not what does it.** The flat design at the card critic's
  width and depth is indistinguishable from the original. The gain is
  the structure (card tokens in the trump context); SwiGLU is inside the
  card arm and is not separated here.
- **Fresh, on 28 steps' worth of data, it nearly reaches v5's critic**,
  which trained over the whole v2-v5 lineage (0.368 against 0.404 held
  out, 0.364 against 0.400 at play start).
- It costs about 3× the flat critic's training time per row here (the
  frozen history encoder is in both numbers).

Raw: `evals/2026-09-28/critic-bakeoff-20260928.json`.

**The v6 population smoke run** (`pop8v6-smoke`, same four steps as
`pop8-smoke`, `--arch v6`):

| | v5 shapes | v6 |
| --- | --- | --- |
| warm-up play step (step 2) | 101.7 s | 141.0 s |
| full-hand play step (step 4) | 102.9 s | 134.2 s |
| pre-play step, 8192 hands | 12–15 s | 26 s |
| peak GPU memory | 4.1 GB | 8.7 GB |
| host RSS | 28.7 GB | 28.7 GB |
| checkpoint (8 agents, with optimisers) | 1.1 GB | 2.4 GB |
| critic explained variance, steps 1–4 | −0.009, +0.214, +0.252, +0.265 | −0.007, +0.008, +0.038, +0.189 |

A v6 step costs about 1.35× a v5-shaped one; at v5's hand length (~60
decisions against the untrained ~87) it would be about 94 s. The card
critic starts slower from random weights -- the deeper, residual-scaled
network takes a few hundred minibatches to move where the flat MLP moves
in one step -- and had nearly caught up by step 4; the bake-off has it
ahead after about 1,700 minibatches, some ten population steps.

**The critic's width, and where a v6 step goes** (2026-09-28). The user
asked for a narrower critic. The same bake-off data, paired, at three
widths (`evals/2026-09-28/critic-bakeoff-widths.json`):

| card critic | params | held out, pass 1 / 4 | at play start, pass 4 | s/pass |
| --- | --- | --- | --- | --- |
| 128 × 4, ff 512 | 1.17M | 0.270 / **0.383** | **0.379** | 25.2 |
| 192 × 4, ff 768 | 2.57M | 0.350 / 0.368 | 0.364 | 31.5 |
| 256 × 4, ff 1024 | 4.51M | 0.338 / 0.363 | 0.368 | 40.2 |

Width 128 is level with 256 from the second pass on and slower only in
the first; v6's critic is now 128.

`modal run modal_perf.py::pop_profile` breaks a population step apart
(`evals/2026-09-28/pop-profile-v6-128.json`). What it and two v6 smoke
runs showed:

- **Two full copies of the round are gone.** The rollout compacted the
  shared block and then copied it out: two single-threaded passes over
  ~10 GB, 3.3–4.8 s a step in training. `shared.copy_ranges` puts each
  worker's rows straight into the learner's buffer on threads: 0.3 s.
- **10 rollout workers beat 7** on the 12-core container: the parent's
  wait on them fell from ~24 s to ~18 s (`--rollout-workers 10`).
- **Splitting a forward by history length does not pay.** Each forward
  carries a fixed ~3 ms; two splits added ~12 s of forward a step in
  training (22.6 → 34.8 s), three were worse still in the profile.
  `forward_chunks` stays at 1.
- **The update looked dispatch-bound, and is not.** Shrinking the critic
  3.8× moved it only from 75.7 s to 73.8 s, but that compared two
  containers; measured in one (below), the cost per row is flat from 256
  to 1024 rows a minibatch. It is arithmetic and memory traffic.
- **Host CPUs differ between containers** enough to swamp these savings
  when two runs are compared; compare inside one container.

### A shared critic, and a step that costs 80 s (2026-09-28)

**The critic is shared** (the user's call): one `CardCritic` for all
eight agents, trained on every row of every step (`train/critic_learner.py`),
reading the raw history -- eight agents have eight history encoders, so
a shared critic can read none of them -- and told which agent sits in
each seat. The agents are actors alone. It values a whole round in one
forward, and its explained variance in the smoke run was +0.40 at step 2,
where the per-agent critics were at +0.008: it learns from eight times
the rows. On the bake-off data, reading raw history rather than a
trained actor's encoding costs it 0.08 of held-out explained variance
(0.295 against 0.376, `evals/2026-09-28/critic-bakeoff-raw.json`) -- an
unequal comparison, since the encoding came from v5's fully trained
actor and pop8's actors start from nothing.

**Optimisations**, measured in one container each (`modal run
modal_perf.py::opt_bench`, `::opt_probe`; `evals/2026-09-28/opt-bench-v6.json`,
`opt-probe-v6.json`):

| | eager | optimised |
| --- | --- | --- |
| rollout round, 4096 hands | 42.8 s lockstep | **28.4 s** overlapped, pool 4096 |
| actor update, per row | 135 µs | **95 µs** compiled |
| shared critic training step, 256 rows | 12.8 ms | **8.6 ms** compiled |
| shared critic forward, 1536 rows | 32.7 ms | 21.2 ms compiled |
| actor forward, 192 rows | 7.4 ms | 5.4 ms compiled |

- **Rollout overlap** (`--rollout-overlap --pool 4096`): each worker's
  hands are two halves taking turns with the parent's forward, so a
  round costs about the larger of the parent's and the workers' time,
  not their sum. Doubling the pool without the overlap bought nothing
  (41.0 s); the doubled pool keeps each half's forward as wide.
- **`torch.compile`** (`--compile`) on fixed shapes: every minibatch
  exactly 256 rows (the ragged tails skipped, 0.014% of rows), history
  trimmed to a multiple of 16 (`model.history_bucket`), the model in
  `static_shapes` mode so it reads no length back to the host. At most
  eight widths, each compiled once and shared by all eight agents (a
  second agent recompiled nothing); loss within bf16 rounding of eager.
  The first compile attempt's two failures were exactly those: new
  shapes every minibatch, and inductor failing to lower the mask's
  `torch.arange` -- now a cumulative sum of ones.
- **Not worth it**: larger minibatches (flat per-row cost: 141 / 122 /
  128 µs at 256 / 512 / 1024); CUDA graphs (`reduce-overhead`, 5-13×
  slower in training); a history-length split of the rollout forward;
  stacking the eight agents' updates with `vmap` (the backward fails in
  the memory-efficient attention kernel under `vmap`).

**All of it together** (`pop8v6c-smoke`, v6 with the shared critic,
10 workers, overlap, pool 4096, compiled; from random weights, so hands
run ~88 decisions):

| per step | v6, before | now |
| --- | --- | --- |
| collect | ~55–60 s | 24.2 s |
| actor updates | ~74 s (with their critics) | 37.4 s |
| shared critic update | – | 13.8 s |
| **step** | **~135 s** | **~80 s** |

Compiling costs ~5–6 minutes spread over the first steps of a launch
(and once more when a history width first appears). The pre-play step
is still ~26 s, most of it the burial network's PPO update, uncompiled.

### Keeping the round on the GPU, and a 63 s step (2026-09-28)

A second round of measurements, each the code the run used against a
candidate, in one L40S container (`modal run modal_perf.py::opt_bench2`;
`evals/2026-09-28/opt2-v6.json`; v6, eight agents, 4096 hands):

| | today | candidate |
| --- | --- | --- |
| collection, 10 → 20 rollout workers | 30.8 s | 30.8 s: the parent is the bound, busy 28.2 s of it |
| the parent's round, ~1140 rows | 101.8 ms | 100.1 ms, one gather and one read-back a round |
| | | 164.6 ms, the eight agents stacked into one `vmap` forward |
| | | **62.7 ms, compiled with dynamic shapes** |
| actor update, a 256-row minibatch | 30.5 ms | 27.4 ms, the rows already on the device |
| | | **20.9 ms, and nothing read back a minibatch** |
| | | 447 ms, gathered on a thread into pinned memory |
| shared critic update, a minibatch | 16.3 ms | **11.6 ms, rows on the device, nothing read back** |
| burial update, one agent's rows | 1.79 s | **0.92 s in bf16**; 0.64 s bf16 and compiled |
| declaration update, one agent's rows | 0.33 s | 0.23 s compiled -- 2.6 s a step in all, left alone |

- **The rollout's forward was the parent's cost**, not its gathers or
  its syncs: batching those saved 2%. Compiling the forwards saved 37%.
  Under `vmap` the attention kernel falls back and the stack is slower.
- **The update was waiting on the host.** Each minibatch was gathered
  from pageable memory -- every such upload a stream sync -- and read
  about fifteen numbers back. With the round on the device and the
  reads gone, the profile shows 20.2 ms of device time in a 20.9 ms
  minibatch: 38% matmuls, 12% attention, 7% the fused AdamW, 12% the
  largest fused elementwise and norm kernels, and ~31% smaller kernels
  below the profile's cut. The device is busy the whole time, and most
  of that time is memory traffic rather than matmul arithmetic.
- **The padded move slots are free.** 128 slots, a median of 4 filled
  (mean 11), but every round and 89% of minibatches hold a list over
  32, so trimming to a batch's longest saves nothing -- and at 16 slots
  an eager minibatch costs what it does at 128 (35.2 against 35.4 ms).

**What went in:**

- `buffer.DeviceRows`: the round copied to the device once a step
  (~10 GiB at 4096 hands, 1.2 s), every learner's minibatches indexed
  there. `PPOLearner` and `CriticLearner` then run `_update_on_device`:
  the legal-action check once per update; a non-finite loss or gradient
  refused on the device by the fused AdamW's `found_inf` -- weights,
  moments and step count untouched, as the host path's not stepping;
  the metrics summed on the device and read once. Taken with fixed
  shapes (`--compile`), a fused optimiser and no `target_kl`
  (`PopulationConfig.device_rows`, on by default). The tests hold it to
  the host path's weights and metrics, and to refusing a NaN step.
- `rollout.routed_step(forwards=)`: `"static"` cuts each forward's
  history on the host to its width bucket; `"compiled"` runs those
  through `torch.compile(dynamic=True)`, one graph for every batch size,
  width and agent. `--compile` turns it on (`TrainConfig.rollout_forwards`).
  Tested: the static path collects the same actions, log-probabilities
  and values as eager, serially and through the pool.
- `BuryTrainConfig.bf16`, on by default and on CUDA only: the burial
  update's forward under bf16 autocast, as the play network's and the
  critic's already were. Its picks are still sampled in float32.

**The smoke run** (`pop8v6d-smoke`, the flags of `pop8v6c-smoke`):

| per step | `pop8v6c` | `pop8v6d` |
| --- | --- | --- |
| collect | 24.2 s | 20.7 s (the parent's forward 9.3 s) |
| actor updates | 37.4 s | 27.6 s |
| shared critic update | 13.8 s | 9.3 s |
| the round's upload | -- | 1.2 s |
| **step** | **~80 s** | **~63 s** |
| pre-play step | ~26 s | ~18 s |

The first step compiles (493 s), the second is still compiling the
rollout (105 s), and the first play step after a pre-play block compiled
newly seen widths (update 57.8 s, critic 24.7 s). The GPU peaked at
15.5 GB. What remains is the parent's own work -- busy 18.5 s of the
20.7 s collection: the compiled forward 9.3 s, the critic 4.2 s, the
gather 3.5 s -- and arithmetic in the updates, which only changing what
is computed (model size, minibatches a step) would move.

**A bigger GPU, more CPUs** (`modal run modal_perf.py::hw_bench`; an H100
SXM and, as the control, an L40S, 32 CPUs each; the code the run uses;
`evals/2026-09-28/hw-h100.json`, `hw-l40s.json`):

| | L40S | H100 |
| --- | --- | --- |
| collect, 10 / 16 / 24 / 30 workers | 21.1 / **18.2** / 20.5 / 20.6 s | 19.7 / **15.2** / 17.8 / 18.3 s |
| actor updates, eight agents | 29.4 s (20.9 ms a minibatch) | 23.9 s (17.0 ms) |
| shared critic update | 9.4 s | 11.8 s |
| pre-play step | 17.1 s | 13.9 s |
| a play step, 16 workers (sum of the parts) | ~62 s | ~56 s |

- **More workers help up to 16**, ~3 s a step, then the parent is the
  bound again (busy 17.0 s of 18.2 on the L40S) and more cost time.
  At 10 workers 32 CPUs are no faster than 12 (21.1 s against 20.7).
- **The H100 buys ~10% of a step.** The rollout's forwards are small
  batches and barely speed up (8.1 → 6.6 s), and the updates stop being
  the device's: the critic's is *slower* on the H100, and the H100
  host's copies were ~15% slower too (the parent's gather 3.8 against
  3.3 s), so the updates are bound by the host launching their kernels
  -- ~17 ms an actor minibatch, ~8 ms a critic one there. (On the L40S
  both updates are the device's: the capture below, which removes the
  host's time, moved them 2% and 6%.) Removing that host time is what
  would let the faster GPU pay; the H100 at ~2x the price an hour does
  not, as things stand.

### The update as a CUDA graph, and a 35 s step on an H100 (2026-09-28)

`train/graphs.py`: each update step -- the minibatch's gather from the
device rows, the forward and loss, the backward, the clip, the fused
AdamW's step (made `capturable`) with its on-device refusal of a
non-finite step, the metric sums -- captured as one CUDA graph per
history width after two uncaptured steps of that width, and replayed.
The graphs are dropped when anything they read by address may have
moved (the device rows, the optimiser state, the learning rate), and a
learner's graphs share one memory pool, which is safe because nothing
in it outlives a replay. `--cuda-graphs` (`PPOConfig.cuda_graphs`, and
the critic's through it); `DeviceRows` now persists across steps,
refilled in place, so the graphs keep their addresses.

Per minibatch (`modal run modal_perf.py::graph_bench`;
`evals/2026-09-28/graph-{l40s,h100}.json`):

| | L40S | L40S, captured | H100 | H100, captured |
| --- | --- | --- | --- | --- |
| actor | 20.7 ms | 20.3 ms | 16.4 ms | **9.4 ms** |
| shared critic | 6.45 ms | 6.05 ms | 8.46 ms | **3.27 ms** |

The capture changes nothing it should not (`::graph_lockstep`,
`graph-lockstep-l40s.json`: three copies stepped together on the same
minibatches, the weight gap after every step). The actor's captured
weights equal the uncaptured ones exactly at every one of 352 steps.
The critic's stay as close to an uncaptured run as a second uncaptured
run does -- 9.9e-5 against 1.6e-4 after the first replay, 0.070 against
0.080 after 704 steps -- because its attention backward accumulates in
a different order each run, captured or not. `tests/test_cuda_graphs.py`
(GPU only; `modal run modal_perf.py::gpu_tests`) holds the captured
update to the uncaptured one, eager and compiled, through a refused
NaN step, an optimiser reload and new device rows.

**The smoke run** (`pop8v6e-h100-smoke`: H100, 20 CPUs, 16 workers,
`--compile --cuda-graphs`, otherwise the flags of `pop8v6d-smoke`):

| per step | L40S, 10 workers (`pop8v6d`) | H100, 16 workers, captured (`pop8v6e`) |
| --- | --- | --- |
| collect | 20.7 s | 13.6–14.1 s |
| actor updates | 27.6 s | 12.5–13.2 s |
| shared critic update | 9.3 s | 4.6–4.9 s |
| **play step** | **~63 s** | **~35–37 s** |
| pre-play step | ~18 s | ~13–15 s |

The first step compiles and captures (608 s), and the first play step
after a pre-play block compiles and captures newly seen widths (134 s).
16.6 GiB of GPU, 27.4 GiB of host memory.

**What a run costs in time.** At the v5-bury lineage's 4150 play steps,
on pop8's schedule (300 warm-up, then 100 pre-play steps before each
200 play), from these untrained-agent step times -- hands shorten as
the agents learn, so these are high:

| | play step | pre-play step | 4150 steps | GPU hours at list price |
| --- | --- | --- | --- | --- |
| L40S, 10 workers | ~63.7 s | ~17.5 s | ~83 h | ~83 × $1.95 |
| H100, 16 workers, captured | ~37 s | ~14 s | ~50 h | ~50 × $3.95 |

About 1.65x faster for about 1.2x the GPU cost, before CPUs and memory
(the H100 container asks for 20 CPUs to the L40S's 12).

### What reading raw history costs the shared critic (2026-09-29)

The shared critic reads the raw history tokens, and on the bake-off
data that cost it ~0.08 of held-out explained variance against a critic
reading v5's trained actor's encoding -- a comparison unfair to a
population whose agents start from nothing. Two more arms, on the same
data and seed as before (`train/critic_bakeoff.py`; `modal run
modal_train.py::critic_bakeoff --arms flat,cards-128,cards-raw-128,cards-rand-128,cards-pre-128`;
`evals/2026-09-28/critic-bakeoff-shared-arms.json`), paired against the
raw-history arm on the same held-out hands:

| arm, width 128 | held-out EV, pass 4 | against raw history, pass 4 |
| --- | --- | --- |
| flat | 0.161 | −0.145 [−0.186, −0.102] |
| raw history (`model.shared_critic`) | 0.306 | -- |
| **an untrained actor's encoding, frozen** | **0.358** (0.365 at pass 3) | **+0.052 [+0.024, +0.081]** |
| v5's trained actor's encoding | 0.383 | +0.077 [+0.052, +0.103] |
| its own two-layer encoder, trained with it | 0.300 (0.004 for two passes) | −0.006 [−0.028, +0.015] |

(The first three repeat the earlier runs to within 0.01.)

- **Most of the gap is not what v5 learned.** A randomly initialised
  history encoder of the actor's design (384 wide, 3 causal layers),
  frozen, recovers about two thirds of it. Raw history tokens, each
  projected on its own, are harder for the critic to use than tokens
  that have each been mixed with the plays before them -- even by
  random weights.
- **The critic's own encoder, trained end to end, did not help here.**
  It sat at the mean for two passes and was still climbing at the
  fourth. (Not the gradient clip, as first guessed -- see below.)
- This is the small-data regime (about 28 PPO steps of critic data). A
  population's shared critic fits eight times the rows every step, which
  may narrow the raw arm's gap on its own.

**The critic's own encoder, again** (`--arms cards-raw-128,cards-rand-128,
cards-pre-128,cards-pre128x2-sep-128,cards-pre384x3-sep-128 --epochs 6`;
`evals/2026-09-28/critic-bakeoff-pre-sep.json`). `cards-pre<W>x<L>-sep`
is an encoder `W` wide and `L` deep whose gradients are clipped apart
from the critic's; every pre-encoder arm logs both gradient norms.

| arm | parameters | best held-out EV | pass 6 | against raw, pass 4 |
| --- | --- | --- | --- | --- |
| raw history | 1.13M | 0.304 | 0.304 | -- |
| untrained encoder, frozen | 1.17M | 0.352 (pass 4) | 0.277 | +0.059 [+0.031, +0.090] |
| own encoder 128x2, one clip | 1.69M | 0.057 (pass 6) | 0.057 | −0.289 |
| own encoder 128x2, clipped apart | 1.69M | 0.353 (pass 6) | 0.353 | +0.054 [+0.022, +0.084] |
| **own encoder 384x3, clipped apart** | 8.34M | **0.367 (pass 3)** | 0.293 | **+0.070 [+0.042, +0.097]** |

- **The clip was not the problem.** The narrow encoder's gradient norms
  before clipping were 4e-5 falling to 2e-7, the critic's ~0.014 --
  both far under the 0.5 bound, so neither clip ever acted. The 128x2
  arm starts collapsed -- the critic predicting the mean, EV 0.004 --
  and escapes at a pass that differs from run to run (the third before,
  the sixth and the second here; clipped together or apart made no
  difference, since the clip never bit). Why it collapses is not known.
- **At the actor's width it does not collapse,** and it learns fastest:
  0.328 after one pass, 0.367 after three, level with the frozen random
  encoder and 0.07 above raw history, then over-fits the seven rounds
  (0.293 by pass 6, as the frozen encoder does). It is 7M parameters
  of encoder in front of a 1.2M critic.
- Over-fitting a fixed set of seven rounds is this measurement's limit,
  not a run's: a run's critic sees new rows every step. The best-pass
  numbers are the comparable ones.

**Adopted** (the user's call, 2026-09-29): v6's shared critic runs the
raw history through its own causal encoder, 384 wide and 3 layers deep,
trained with it (`ModelConfig.critic_history_d` / `critic_history_layers`,
`CardCritic`; 0 layers reads it raw as before). 8.34M parameters, 7.16M
of them the encoder. What it costs a step (`pop8v6f-smoke`,
`pop8v6f-h100-smoke`, the flags of `pop8v6d` / `pop8v6e`):

| per play step | L40S, raw | L40S, encoder | H100 captured, raw | H100 captured, encoder |
| --- | --- | --- | --- | --- |
| collect (its critic forward) | 21.3 s (4.2) | 26.0 s (7.7) | 13.6 s (1.3) | 16.1 s (2.4) |
| actor updates | 29.1 s | 28.3 s | 13.2 s | 12.9 s |
| shared critic update | 9.5 s | 22.7 s | 4.8 s | 10.5 s |
| **step** | **~63 s** | **~80 s** | **~36 s** | **~44 s** |
| pre-play step | ~18 s | ~19 s | ~14 s | ~15 s |

About a quarter more a step: the critic's update more than doubles, its
rollout forward nearly doubles, and nothing else moves. Compiling the
encoder adds ~3-6 minutes to a launch. Peak GPU memory is unchanged
(16.1 / 16.7 GiB). At the v5-bury lineage's 4150 play steps: ~102 h on
the L40S, ~60 h on the H100.

**The history, encoded a trick at a time -- tried and removed** (the
user's call on these numbers; built in `d7eea4b`, reverted in `47c7385`,
where `model.HistoryStore` and `rollout.RolloutCaches` can be restored
from). Within a hand a seat's
history only grows and the encoder is causal, so each decision needs
only the plays since the seat's last -- about four, where the history
averages ~44 -- plus the open trick's last three cached tokens again,
whose trick-won flag is written when the trick closes (resuming without
them measurably goes stale; the tests hold both). One store per agent's
encoder over the (hand, seat) slots it plays, and one for the shared
critic's; about 20 GB of GPU at 4096 hands. Measured against the same
flags without it (`pop8v6g-smoke`, `pop8v6g-h100-smoke`), at equal
decision counts:

| parent, per play step | L40S | L40S, cached | H100 | H100, cached |
| --- | --- | --- | --- | --- |
| actors' forward | 10.6 s | 9.9 s | 6.4 s | 7.1 s |
| critic's forward | 7.9 s | 4.9 s | 2.4 s | 2.0 s |
| collect | 26.3 s | 22.1 s | 15.8 s | 15.3 s |
| peak GPU memory | 16 GiB | ~36 GiB | 16 GiB | 36 GiB |

- **About 4 s a step on the L40S** (~80 -> ~76 s), nearly all of it the
  critic's forward. Its updates were a further ~2 s faster in that run,
  but the cache does not touch them: that is container variation.
- **Nothing on the H100.** The store's incremental encoding runs
  uncompiled -- some sixty small launches per agent per round -- and on
  the faster card that host time costs what the arithmetic saves; the
  actors' forward got slower.
- Compiling the store's encoding (dynamic shapes, as the rollout's
  forwards are) is what would turn the actors' half into a gain; not
  done. Not worth 20 GB and a code path for ~5% on one card.

---

### Why pop8 diverged (2026-09-29)

`pop8` (H100; compiled rollout and update, CUDA graphs) was healthy for
35 steps -- approx_kl ~0.006 -- and then each agent in turn ran away:
agent 0 at step 36, the last at 83; approx_kl to ~1e8, half an agent's
rows at the log-ratio clamp, gradients 1e17 before clipping, logits from
~30 to ~1e4 in a few steps (`evals/2026-09-29/`: the run's metrics,
`diag-pop8.json`, `logit-scan-diag-r1.json`). Nothing preceded it in the
auxiliary losses; each agent's entropy had been falling, and it broke
somewhere between 1.0 and 1.4.

Runs from scratch, same seed, 70-90 warm-up steps each, isolated it:

| run | rollout | update | first agent broken | play step |
| --- | --- | --- | --- | --- |
| R1 | compiled | compiled, CUDA graphs | step 35 (six by 44) | ~42 s |
| R3 | eager | compiled, CUDA graphs | step 30 (seven by 51) | -- |
| R2 | eager | eager, host | none by 90 (one spike, recovered) | ~73 s |
| R4 | compiled | eager, host | none by 60 | ~67 s |
| R5 | compiled | fixed shapes, CUDA graphs, **not compiled** | none by 40+ | ~51 s |

- **`torch.compile` of the update is the cause.** The compiled rollout
  is not (R4), nor the fixed shapes, the device rows or the CUDA graphs
  (R5; graphs matched compiled-without-graphs exactly in the diagnostic).
  On healthy networks every path agrees with eager to bf16 noise; what
  the compiled training graph does differently shows only as logits grow.
- **The recipe has a milder version of it.** The all-eager run's agent 7
  spiked to approx_kl 0.27 at step 64 (entropy 1.07) and recovered. A KL
  guard on the update (stop an agent's update past a threshold) would
  keep such a spike from compounding; not added, the user's call.
- Once an agent's logits are ~1e4, bf16 rounding makes every path's
  log-probabilities disagree by tens of nats (up to 64), so a broken
  agent never recovers: the precision is the amplifier, not the cause.
- **Not an extreme reward.** Kitty multipliers can make a row's reward
  ~4.8 (a normal trick's is at most ~0.25), but advantages are normalised
  per minibatch and clipped, the returns' own value loss is flat through
  the onset, agents broke one at a time rather than a table at once, and
  the eager runs see the same rewards without breaking.

`pop8b` runs R5's configuration.

### pop8b's KL spikes, and the guard (2026-09-29)

`pop8b` did not diverge, but its milder instability grew. Agent-steps
with approx_kl over 0.05, per 50 steps: 0, 1, 6, 13, 24, 6 to step 300;
then 39, 23, 114, 100 once learned declaring and burial changed the
hands. By steps 401-500 agents 1 and 6 were over 0.05 on nearly every
step (peaks 0.4-0.7, up to 3% of their rows at the log-ratio clamp),
recovering each time but never settling; the other six stayed near
0.01. Explained variance fell from ~0.19 to ~0.04 at step 300 with the
critic's loss unchanged: the kitty's large, predictable rewards left
with random burial (15 points buried -> 0.3), not a critic fault.

`modal_perf.py::diag_kl` on the step-400 checkpoint
(`evals/2026-09-29/diag-kl-pop8b-400.json`) settled where the KL comes
from. Before the update, every agent's eager log-probabilities match the
compiled rollout's (KL 0.0003-0.0014 in bf16, no clamped rows), so it is
not a rollout/learner mismatch. Within one update agent 6's
per-minibatch KL climbs: 0.002-0.005 for the first minibatches, 0.05 by
the 16th of 136, 0.2-0.34 through the second half, 1.05 at worst. A
healthy agent's never passes 0.045. The run's path (fixed shapes, CUDA
graphs, device rows) gives the eager host update's KL to four digits.

So the policy walks too far within a pass, which is what PPO's
`target_kl` early stop is for. It now runs on the device path as well
(`_update_on_device`: a flag set by the first minibatch over the
threshold refuses it and every later one through the fused optimiser's
`found_inf`, as for a non-finite step; the weights are the host path's),
reports `kl_at_stop` and, per agent, `early_stop` and `n_minibatches`,
and is `--target-kl` at launch. At 0.05 it would have stopped agent 6 at
minibatch 16 of that update and left a healthy agent untouched.

**`pop8c` -- `pop8b`'s step 900 to 1300 with `--target-kl 0.05`
(2026-09-30, `evals/2026-09-30/pop8c-metrics.jsonl`).** The spikes went:
no update's approx_kl passed 0.025 (`pop8b`'s reached 0.7), no agent was
ever left without a step, and the guard refused single minibatches that
had jumped to KL 1.25 and 1.8. But by step 900 the threshold bound almost
every update, not only the spikes: 88-100% of agent-updates stopped
early, the median one after 13 of ~125 minibatches, and the agents took
24% of their minibatches overall (agent 7 5%, agent 4 8%, agent 5 48%).
The KL at a stop sat just past the line (median 0.079, 99th percentile
0.27). At step 400 a healthy agent's minibatches never passed 0.045; by
step 900, with entropy 0.40 against ~0.55, the same learning rate
crosses 0.05 within a few dozen steps. So 0.05 cut the learning roughly
fourfold. Entropy kept falling (0.40 -> 0.37), the critic's loss rose
(0.0016 -> 0.0017).

Ladder at step 1300 (`evals/2026-09-30/pop8c-final-*`): the population's
mean went from -9.0 points at step 500 to -3.2 with random pre-play
(-10.1 to -3.5 with its own), the best agent to +2.9 (0.537 win) and
+0.9 (0.500). Whether the guard helped is not separable from that: the
gain spans 400 unguarded and 400 guarded steps (`pop8b` step 900 is kept
to split them).

**Continued to step 2800 with the guard at 0.15** (2026-10-01, the
user's call). Updates stopped early 56% of the time and took 58% of
their minibatches (agents from 20%, agent 3, to 84%); no update's KL
passed 0.086, none went without a step, and 37 single minibatches that
had jumped past KL 1 (up to 3.7) were refused. Entropy held (0.38 ->
0.37); the critic's loss kept rising (0.0017 -> 0.0023).

### Which follows the network is shown, in training and in evaluation (2026-11-13)

**Training and evaluation offer the network different follows.** Training
shows it the simulator's MoveSet (`legal_moves(cap=max_candidates)`, 128).
The evaluation host shows it the engine's `enumerateCandidates` at 128,
and the agent can only pick from what it is shown. Both were compared with
the exact legal set: brute force, judged by the respective rules, which
agree exactly (below). Random play over 2,000 hands (simulator) and
600 hands (host), policy weighted toward large plays:

| multi-card follows | simulator (training) | host (evaluation) |
| --- | --- | --- |
| offered set incomplete | 0.8% | 13.8% |
| a trick-winning follow is legal, none offered | **1.3%** (52 of 4,060) | **11.8%** (149 of 1,264) |
| the same, pair leads only | 0.2% | 7.2% |
| the same, 4-card leads | 28% | 49% |

With a uniform random policy the losses are 0.7% and 9.8%. So the network
trains with nearly every follow on offer and is benchmarked with a
thinner list: about 0.25 lost winning follows a hand in evaluation, against
~0.03 in training. Every number against the bots so far was measured
through the host's list, so they are likely to understate the agents. The
remaining simulator gap sits at tractor and throw leads, where it falls
back to the single canonical follow once 1,024 fills are exceeded.

```bash
python scripts/follow_coverage.py 2000 128 big      # the simulator
npx tsx src/followCoverage.ts 600 big                # the evaluation host
```

**The two follow rules agree.** `src/dumpFollowAll.ts` judges *every*
follow of 600 structure-rich positions (trump led half the time, throws
carrying tractors), not a random hundred as `tests/test_differential.py`
does: 212,585 follows, 1,968 of them legal, and the Python rule matches
the engine on all of them (`scripts/compare_follow_all.py`).

**`smart` was the bot refusing its own follows.** pop8c's agent 3 against
`smart` with bot pre-play failed on every attempt with "must include
tractors of length 2 in trump". The host's candidates all pass the
reducer (412,084 checked over 700 hands); `smart`'s own follows did not.
Its follow paths predate the order-independent obligations, and on 20 of
the 600 positions above it played loose pairs where the rule demands a
tractor. `pickBotAction` now repairs a refused follow on its way out and
counts it (`botFollowRepairs()`), leaving legal follows, and the bots'
internal simulations, exactly as they were. `smart-v18`/`v28`/`v34` needed
no repairs over 48 real positions with a live tractor obligation, too few
to clear them; the guard covers them either way. Re-running the failing
pair would confirm the fix end to end.

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
