# Results

Measured numbers only. Every entry says what hardware it ran on, since the
laptop (i7-1270P, 16 threads, no GPU) and any cloud box differ a lot.

## Harness validation

Before any agent number is worth reading, the measuring instrument has to
be shown not to lie. Two checks, both on the laptop.

**Deal pairing** (`npx tsx src/dupcheck.ts`). The two halves of a duplicate
pair must be the same cards. Compared card for card between slots that
differ in learner seats, opponent bot, policy stream and slot RNG seed:

```
paired deals verified   36 (3 game slots x 12 hands)
cards compared          3888
different base differs   yes (0/12 identical)
different game differs   yes
digests unique           12/12
```

**Null test** (`py/arena.py --agent bot:lowball --opponent lowball`). The
same deterministic bot on both sides must score exactly zero:

```
96 duplicate pairs (192 hands)
point margin: +0.00 pts [95% CI +0.00, +0.00]  (0 better / 0 worse / 96 level)
level margin: +0.00 levels [95% CI +0.00, +0.00]
attacker points: agent 95.6 / opponent 95.6 on the same deals
```

Exactly zero on all 96 pairs, so the seat swap, the pairing and the
scoring introduce no bias of their own.

**Directional checks.** The instrument also has to *detect* differences it
should, and rank them the way the historical lineage did:

| Match | pairs | point margin (95% CI) | level margin | pairs better |
| --- | --- | --- | --- | --- |
| `lowball` vs `random` | 192 | +0.21 [-4.56, +4.98] | -0.03 | 86 / 98 worse |
| `smart` vs `random` | 240 | **+83.96 [+79.70, +88.22]** | +2.14 | 238 / 2 worse |
| `smart-v2` vs `smart` | 240 | **+20.40 [+17.17, +23.62]** | +0.50 | 187 / 37 worse |

A tie reads as a tie, and both known gaps come out large, tight and in the
direction the lineage recorded (`smart` beats random ~100%; `smart-v2`
beat `smart` 116-4). Each of those took ~0.1 min. Note the CI width:
±4.8 points on 192 pairs. The 32-game win-rate benchmarks this project
used before have a CI of roughly ±18% and could not resolve any of it.

## Cost of evaluation

Per-hand cost when a bot fills two of the four seats (laptop, one worker):

| Bot | s/hand | hands/hour/worker |
| --- | --- | --- |
| `random`, `lowball`, `point-hoarder`, `smart` | 0.01-0.02 | 150k-350k |
| `smart-v2` | 0.03 | 135k |
| `smart-v13` | 12.2 | 296 |
| `smart-v34` | 69.2 | 52 |
| `smart-v18` | 75.2 | 48 |

The determinization-search bots are ~3 orders of magnitude more expensive
than the heuristics, so evaluation against them has to be budgeted rather
than run casually: 200 duplicate pairs (400 hands) against `smart-v34`
costs ~8 worker-hours. An initial attempt at exactly that was abandoned
after 19 minutes without completing its first round.

## Throughput (laptop)

Component costs, measured separately:

| Component | Rate |
| --- | --- |
| TS actor: engine + candidates + record encoding | ~9k decisions/s per worker |
| — of which lead enumeration, before pruning throws | 725 µs/lead (96% of lead cost) |
| — after public-information throws | ~26 µs/lead |
| torch CPU inference, 0.74M params, K≈8 candidates | 8–10k decisions/s |
| torch CPU inference, 1.7M params | 1.9k decisions/s |
| torch CPU training step, batch 512 | ~8.6 steps/s = 4.4k samples/s |
| numpy featurisation, in-loop | ~12k decisions/s |
| Wire volume | ~0.7 KB/decision |

**Configuration sweep** (`py/sweep_throughput.py`, 70 s per setting, real
training loop, exclusive use of the machine):

| workers × games | torch threads | groups | dec/s | hands/s |
| --- | --- | --- | --- | --- |
| 8 × 48 | 6 | 2 | **4,108** | 56.2 |
| 6 × 64 | 6 | 2 | **4,043** | 55.3 |
| 4 × 192 | 8 | 2 | 1,814 | 20.3 |
| 4 × 96 | 8 | 1 | 1,419 | 14.6 |
| 4 × 96 | 8 | 2 | 1,336 | 12.7 |
| 3 × 128 | 10 | 1 | 1,324 | 19.0 |
| 6 × 64 | 8 | 2 | 1,194 | 12.8 |

Nearly 3x, and the lever is not the one I expected. Overlapping workers
with Python (groups=2) does help, but the dominant term is **torch thread
count**: at 8-10 threads torch and the Node actors fight for the same
cores and both lose. Every configuration at 6 threads beats every
configuration at 8+, regardless of worker count. Worth re-measuring on any
new machine rather than porting these numbers.

End-to-end loop, random policy, `py/smoke_protocol.py`:

| Workers × games | Decisions/s | Hands/s |
| --- | --- | --- |
| 2 × 32 | 5,379 | 62 |
| 4 × 32 | 4,087 | 42 |
| 4 × 96 | 4,815 | 39 |

Those were measured before the group-rotation and thread findings above,
with a random policy and torch idle; the sweep table supersedes them for
anything with the network in the loop.

## Agent results

**Untrained network** (0.67M params, random init), 251 pairs vs `random`:

```
point margin: -17.43 pts [95% CI -22.82, -12.04]  (77 better / 166 worse / 8 level)
level margin: -0.43 levels [95% CI -0.58, -0.28]
attacker points: agent 80.3 / opponent 97.8 on the same deals
```

Worse than random, which is the right floor and worth understanding: an
untrained scoring net is not a uniform random policy, it is a *systematically
biased deterministic* one — it always prefers whatever candidate its random
weights happen to favour. Training has ~17 points to climb before it has
learned anything at all.

**Pilot run** — 20 minutes of DMC self-play on the laptop (4 workers × 96
games, 0.67M params, ε=0.02, RMSprop 1e-4, batch 512): **18,100 hands,
1.16M decisions, 1,460 learner steps**, loss 1.62 → ~1.25. Evaluated
greedily over 252 duplicate pairs vs `random`:

```
point margin: +62.42 pts [95% CI +57.04, +67.80]  (230 better / 20 worse / 2 level)
level margin: +1.58 levels [95% CI +1.44, +1.73]
attacker points: agent 124.9 / opponent 62.5 on the same deals
agent reached 80+ as attacker on 93.7% of deals, opponent 35.3%
```

So the agent moved from **−17.4 to +62.4** points against random in 20
minutes of training, covering about three quarters of the gap to the
hand-written `smart` heuristic (+83.96 on the same measure). The learning
signal works; what it has learned at this scale is presumably the coarse
stuff (take the trick when points are on it, do not throw away 10s and
Ks), and 1,460 gradient steps is nowhere near converged.

Caveat on that run's wall clock: it was given 20 minutes but took 37,
because duplicate-deal evaluations were running on the same box and
starved the workers. Training and evaluation should not share this
machine — the throughput numbers above assume exclusive use.

### What a serious run costs here

At the measured ~1,800 decisions/s and ~2.3 learner steps/s, the laptop
delivers roughly **150k hands and 200k gradient steps per day**. DouZero
needed on the order of 10^7-10^8 games for Dou Dizhu, so this hardware can
reach a decent heuristic-beating agent overnight at best, and the case for
moving training to a GPU box is mostly a case about gradient steps per
second, not about the environment.
