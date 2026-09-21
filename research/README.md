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

## Next steps

1. **Longer training**, then evaluate against the real opponents. Budget
   matters: `smart-v18`/`v34` cost ~70 s/hand to play against (~50
   hands/hour/worker), so a 200-pair match is an ~8 worker-hour job.
   `smart-v13` at 12 s/hand is the affordable middle rung.
2. **Faster featurisation / GPU path**. `--device cuda` is wired but
   untested; featurisation would need to move onto the device with it.
3. **Learned pre-play**: kitty burial as 8 sequential card choices, then
   declaration. Both are currently `smart-v34`'s job.
4. **Bot adapter** so a trained checkpoint can fill a seat in the web app.
