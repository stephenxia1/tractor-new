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
to `runs/<run>.jsonl`.

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
