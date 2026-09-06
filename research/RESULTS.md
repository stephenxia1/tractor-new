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
| `smart-v13` | 12.2 (3.8 uncontended) | 296 |
| `smart-v34` | 69.2 | 52 |
| `smart-v18` | 75.2 | 48 |

The determinization-search bots are ~3 orders of magnitude more expensive
than the heuristics, so evaluation against them has to be budgeted rather
than run casually: 200 duplicate pairs (400 hands) against `smart-v34`
costs ~8 worker-hours. Caveat on this table: it was measured while other
work shared the laptop, and it overstates the cost by roughly 3x — a later
uncontended `smart-v13` evaluation ran 245 hands in 2.6 minutes across 6
workers (~3.8 s/hand), against the 12.2 s/hand predicted here. An initial attempt at exactly that was abandoned
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

**These are burst numbers.** Each setting ran 70 s, which fits inside the
laptop's turbo window. A sustained run at the winning setting (6 x 64, 6
threads, 2 groups) settles at **~1,900 decisions/s, ~26 hands/s** — 2.1x
lower — with the CPU pegged at 88% of its 2.2 GHz nominal clock, i.e. no
turbo at all. Short benchmarks systematically overstate this machine; the
sustained figure is the one to plan with, and the gap is itself an argument
for moving training somewhere that holds its clocks.

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

## Moving to AWS (2026-09-05)

A `c7i.8xlarge` spot instance (32 vCPU Sapphire Rapids, $0.616/hr) runs the
same loop **17x faster than the laptop**, far beyond the 3-5x I estimated:

| | laptop (sustained) | c7i.8xlarge |
| --- | --- | --- |
| decisions/s | ~1,900 | **24,300** |
| hands/s | ~26 | **381** |
| learner steps/s | ~5 | **36** |
| loop split env/feat/fwd/learn | 2/6/39/51 | 9/24/19/36 |

The split is the explanation. On the laptop torch was 90% of the loop; here
it is 55%. Sapphire Rapids has AVX-512, which the i7-1270P has fused off,
and it holds its clocks instead of dropping to base under sustained load.
The laptop was not merely smaller, it was running the arithmetic-heavy half
of this workload without the instructions for it.

**Featurisation is the next ceiling, as predicted.** It was 6% of the loop
on the laptop and is 24% here, because it is numpy on the host and does not
speed up when torch does. On a GPU box it would dominate, so moving it onto
the device is the prerequisite for GPU work rather than an optimisation
after it.

**Tuning flipped objective.** With 24k decisions/s the environment stopped
being the constraint and gradient steps became scarce, so `py/tune_learner.py`
optimises steps/s at a sane replay ratio instead of decisions/s:

| config | dec/s | steps/s | replay |
| --- | --- | --- | --- |
| 10 workers / 22 threads, every2 x1 | 24,866 | 38.3 | 0.79 |
| 8 / 24, every1 x2 | 10,516 | 79.7 | 3.88 |
| 8 / 24, every1 x4 | 6,351 | 92.4 | 7.45 |
| 5 / 27, every1 x4, batch 1024 | 4,170 | 62.6 | 15.36 |

Chose the first: fresh data at replay 0.79 rather than 2.4x the steps
grinding each sample seven times.

### First cloud run (in progress)

`c7i.8xlarge`, 10 workers / 22 threads, replay 0.79. After **80 minutes**:
1.9M hands, 122M decisions, **190k gradient steps** — 15x the hands and
8.6x the steps of the laptop's entire 8-hour run. Training loss hovers
between 0.67 and 0.92 rather than falling, which is what DMC looks like
when the policy keeps shifting the target distribution; the arena is the
progress signal, not the loss.

Evaluated greedily on duplicate deals at the 190k-step checkpoint:

| opponent | point margin (95% CI) | level margin | pairs better |
| --- | --- | --- | --- |
| `random` | **+89.13** [+84.31, +93.94] | +2.20 | 250 / 1 worse |
| `smart` (v1) | **+14.88** [+10.27, +19.49] | +0.40 | 157 / 86 worse |
| `smart-v2` | -0.96 [-6.02, +4.10] | +0.04 | 116 / 128 worse |
| `smart-v13` | **+15.46** [+9.02, +21.89] | +0.37 | 74 / 36 worse |
| `smart-v18` | **-10.20** [-17.60, -2.80] | -0.22 | 31 / 61 worse |

For reference, `smart` itself scores +83.96 against `random`, so the agent
has passed the multi-heuristic bot outright and drawn level with
`smart-v2`, the 1-ply search bot that beat `smart` 116-4 in the original
lineage. The progression on this measure:

    untrained            -17.4  vs random
    20 min / 1.5k steps  +62.4  vs random
    80 min / 190k steps  +89.1  vs random   (> smart's +84.0)

Beating `smart-v13` is the more meaningful of these: it is the first
*fair-play* bot in the lineage, the one that established masking opponent
hands via determinization, and it costs ~3 orders of magnitude more
compute per decision than the agent does.

**It still loses to `smart-v18`**, the deployed fair-play peak, by 10.2
points. That is the honest standing: an 80-minute checkpoint sits between
`smart-v13` and `smart-v18`. Worth keeping in proportion in both
directions — v18 runs a 5-ply determinized search costing ~1 s per lead
decision, the network answers in about a millisecond, and the gap is 10
points rather than 50. Whether training closes it is the question the rest
of the run answers; this checkpoint is 190k steps of a ~2M-step run.

Note both sides use `smart-v34` for declaration and kitty burial in every
one of these matches, so what is being compared is in-trick play alone.

An incidental gap in the old lineage this harness could close cheaply: v2
and v13 were never measured against each other directly, and the agent
ties one while beating the other.

## Architecture v2 (experiment running)

Changes, in the order they matter:

1. **Points-primary targets.** The level bracket is a step function of
   points and fired once per hand; the point return is smooth and arrives
   every trick. Level becomes an auxiliary head so the 80-point cliff
   stays represented.
2. **Privileged critic.** A critic seeing all four hands is distilled into
   the policy network, which is unbiased in expectation and far quieter.
3. **Chain-position encoding**, built from the engine's own `chainPosition`
   via `chains.json` so it cannot drift from the rules.
4. **Per-role heads**, causal-attention move history, a state-value head
   for search, and a belief head over opponents' hands.
5. **Opponent mixture**: 65% self-play, 20% vs frozen snapshots, 10% with a
   snapshot as *partner*, 5% vs a cheap bot.

**Cost, measured on the same box:** 3.13M params (v1: 0.67M) and **7.3
learner steps/s against v1's 38** -- roughly 4x fewer gradient steps per
hour, split env 4 / features 26 / forward 22 / learn 46. So v2 has to be
worth about 4 steps of v1 per step of its own. That is the experiment.

**Early diagnostics** (420 steps): the privileged critic already explains
~60% of the target's spread (`target_std` 0.68 vs `critic_residual_std`
0.27), and the belief head predicts opponents' card positions at 84%.
Those are the two mechanisms the design rests on, both confirmed live
rather than assumed.

### What a serious run costs here

At the measured ~1,900 decisions/s and ~5 learner steps/s, the laptop
delivers roughly **150k hands and 200k gradient steps per day**; the
8-hour laptop run reached 1.9h, 184908 hands, 31811 steps, loss 0.62555 before being superseded. The AWS box
delivers ~33M hands and ~3M steps per day for about $15. DouZero
needed on the order of 10^7-10^8 games for Dou Dizhu, so this hardware can
reach a decent heuristic-beating agent overnight at best, and the case for
moving training to a GPU box is mostly a case about gradient steps per
second, not about the environment.
