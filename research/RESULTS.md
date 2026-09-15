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

**Diagnostics over 107k steps**: the privileged critic explains ~63% of
the target's spread throughout (`target_std` ~0.57 vs
`critic_residual_std` ~0.21), and belief accuracy climbed 0.847 -> 0.879.
Both mechanisms the design rests on held up for the whole run rather than
just at the start.

### v2 vs v1 on the ladder

v2 at **107k steps / 4.0h** against v1 at **190k steps / 1.3h** -- v2 had
*fewer* gradient steps, since it runs ~4x slower per step:

| opponent | v1 | v2 |
| --- | --- | --- |
| `random` | +89.13 | **+95.33** [+88.90, +101.76] |
| `smart-v2` | -0.96 (tie) | **+17.40** [+11.74, +23.06] |
| `smart-v13` | +15.46 | **+20.53** [+13.23, +27.83] |
| `smart-v18` | **-10.20** (loss) | **+1.90** [-4.17, +7.97] (tie) |

v2 is ahead on every rung, and the one that matters is the last: v1 lost
to the deployed fair-play peak by 10 points, v2 draws level with it. Since
v2 took *fewer* gradient steps to get there, the gain is sample
efficiency -- the architecture and the target -- rather than compute.

Reading the v18 row honestly: a tie, not a win. The CI spans zero and 120
pairs is a small sample against an expensive opponent. What can be said is
that the 10-point deficit is gone.

## Search (Phase 3)

Determinized value-guided search: sample worlds consistent with public
play, play each candidate in each, roll forward to the seat's *own* next
decision, and score the resulting position with the learned value head.
Leaves travel as ordinary decision records carrying zero candidates,
tagged with the candidate they inform, so one real decision still costs
one network round trip.

**Against `smart-v2`, the same checkpoint:**

| configuration | point margin |
| --- | --- |
| no search | +17.40 [+11.74, +23.06] |
| search, leaves at trick end | -6.14 [-20.58, +8.31] |
| search, leaves at trick end, all candidates | -13.89 [-24.81, -2.96] |
| **search, rolled to own turn, all candidates** | **+29.76** [+22.07, +37.44] |

Search made the agent *worse* twice before it helped, and both reasons are
worth writing down.

**The value head had never seen the positions it was being asked about.**
Training records exist only where the recorded seat is about to move, so
V(s) is trained exclusively on "my turn" states. Leaves taken at trick end
are usually someone else's turn -- out of distribution, and confidently
wrong there. Simulations now roll forward to the searching seat's next
decision.

**Search was picking from the wrong menu.** The worker searched
`candidates[0..N]` in enumeration order rather than the top N by Q -- the
policy prior the design called for was never built -- and then forced a
pick inside that arbitrary subset, discarding the network's preference
entirely. Searching every candidate turned a 14-point loss into a
30-point win. Python now overrides the network only where search covered
all candidates.

**Against `smart-v18`, the deployed fair-play peak:**

```
121 duplicate pairs (246 hands)
point margin: +16.90 pts [95% CI +10.49, +23.31]  (78 better / 33 worse / 10 level)
level margin: +0.48 levels [95% CI +0.31, +0.65]
attacker points: agent 83.8 / opponent 66.9 on the same deals
search: 5,353/8,088 decisions searched (212,103 leaves)
```

The full progression against v18, which is the bar this project has been
measured against for most of its life:

    v1, 190k steps          -10.20   loss
    v2, 107k steps           +1.90   tie
    v2 + search             +16.90   win

**Against `smart-v34`, the currently deployed peak:**

```
120 duplicate pairs (245 hands)
point margin: +17.75 pts [95% CI +11.09, +24.41]  (77 better / 33 worse / 10 level)
level margin: +0.49 levels [95% CI +0.32, +0.67]
attacker points: agent 84.1 / opponent 66.3 on the same deals
```

v18 and v34 come out within a point of each other (+16.90 and +17.75),
which is consistent with the lineage's own finding that v34 beats v18 only
53% of the time -- the two are close, and the agent is clear of both.

Worth being precise about what search is here: it is `smart-v18`'s own
algorithm -- sample opponent hands, simulate, average -- with the
hand-tuned rollout policy and cost heuristics swapped for a learned
rollout and a learned evaluator. Beating v18 by 17 points with the same
method is a statement about the evaluator, not about search.

The lesson generalises: a search is only as good as the agreement between
the distribution its evaluator was trained on and the positions it is
asked to judge, and adding search to a policy silently *replaces* that
policy's ranking with the search's -- so a bug in the search is not a
smaller effect than a bug in the policy, it is a larger one.

### The run was cut short

The spot instance was reclaimed at ~7.9h uptime, 4.0h into a 9h v2 run.
Nothing was lost: checkpoints and metrics mirror to S3 every 20 minutes,
so the 107k-step checkpoint and the full curve survived the instance.
Total spend ~$4.90. The lesson for next time is in the setup rather than
the outcome -- the box carried a 20h self-terminate while the run was 9h,
so an uninterrupted instance would have idled ~7h on the meter.

### What a serious run costs here

At the measured ~1,900 decisions/s and ~5 learner steps/s, the laptop
delivers roughly **150k hands and 200k gradient steps per day**; the
8-hour laptop run reached 1.9h, 184908 hands, 31811 steps, loss 0.62555 before being superseded. The AWS box
delivers ~33M hands and ~3M steps per day for about $15. DouZero
needed on the order of 10^7-10^8 games for Dou Dizhu, so this hardware can
reach a decent heuristic-beating agent overnight at best, and the case for
moving training to a GPU box is mostly a case about gradient steps per
second, not about the environment.

## Does training longer still help? (Phase 4)

The honest worry about this whole setup is that the losses look converged
almost immediately and stay that way. Over the 112k steps of the third
run they did not move at all:

| steps | `loss_q` | `critic_residual_std` | `target_std` | `loss_level` | `belief_acc` |
| --- | --- | --- | --- | --- | --- |
| 8k | 0.01462 | 0.20056 | 0.60331 | 0.3732 | 0.8743 |
| 40k | 0.02134 | 0.20848 | 0.61409 | 0.4824 | 0.8751 |
| 72k | 0.02636 | 0.22002 | 0.64570 | 0.4803 | 0.8744 |
| 112k | 0.02119 | 0.19905 | 0.58987 | 0.4414 | 0.8829 |

That flatness is not evidence of convergence, and it is worth being
explicit about why. The regression target is a Monte-Carlo return
generated by the *current* policy against a *moving* pool of snapshot
opponents. As the agent improves, the target distribution moves with it,
so `loss_q` measures the fit to a distribution that keeps sliding, not the
quality of the play. The only convergence signal that means anything here
is strength against a fixed opponent as a function of steps.

Measured, same deal base (90210) and same 120-pair budget as the earlier
reading, no search on either side:

| steps | vs `smart-v13` | gain |
| --- | --- | --- |
| 107k | +20.53 [+13.23, +27.83] | -- |
| 210k | +30.60 [+23.83, +37.38] | +10.07 |
| **417k** | **+38.17** [+31.87, +44.47] | +7.57 |

```
124 duplicate pairs (269 hands) in 17.3 min
  point margin: +30.60 pts [95% CI +23.83, +37.38]  (101 better / 19 worse / 4 level)
  level margin: +0.77 levels [95% CI +0.59, +0.96]
  attacker points: agent 102.3 / opponent 71.7 on the same deals
  agent reached 80+ as attacker on 75.8% of deals, opponent 41.1%
```

**+10.07 points over the second 100k steps, while the loss did not move.**
Treated as independent samples the gain is +10.07 +/- 9.96 and only just
clears zero; the two runs share a deal base, so a paired test would be
tighter than that, and 101 of 124 pairs improved. Call it a real gain
rather than a confirmed effect size.

### The network passes `smart-v18` without search

The rung that has been the bar for most of this project, measured on the
same 210k-step checkpoint with **no search on either side**:

```
125 duplicate pairs (277 hands) in 70.5 min
  point margin: +11.92 pts [95% CI +3.52, +20.32]  (72 better / 46 worse / 7 level)
  level margin: +0.33 levels [95% CI +0.11, +0.54]
  attacker points: agent 89.3 / opponent 77.4 on the same deals
  agent reached 80+ as attacker on 64.0% of deals, opponent 48.8%
```

The full progression against the deployed fair-play peak:

    v1, 190k steps         -10.20   loss
    v2, 107k steps          +1.90   tie
    v2, 107k + search      +16.90   win
    v2, 210k steps         +11.92   win, no search

The policy itself now beats `smart-v18`, where at 107k it needed search to
do it. The CI is wide and clears zero by 3.5 points, so this is a win
rather than a rout -- but it is the *network* winning, and search stacks
on top of it rather than substituting for it.

The third point extends the curve after a full uninterrupted 7h run
(207,733 steps on top of 209,715, so ~417k cumulative). Strength is still
climbing and the loss is still flat, but the gain per step is shrinking --
+10.07 over the second 100k, +7.57 over the next 207k, roughly logarithmic.
Not converged, but no longer cheap.

The practical conclusion: **stop reading the loss curve.** Budget for
periodic ladder evaluations instead, and treat step count as far from
saturated -- 210k steps is still on the improving part of the curve.

### Third reclaim, and the checkpoint fix paying off

`i-0d53f7b3922812dff` (c7i.8xlarge, us-east-1d) was reclaimed at 3.88h,
the third spot reclaim in a row across two AZs. The previous reclaim cost
22,636 steps because `--checkpoint-every 50000` worked out to ~100 minutes
at 8.3 steps/s, longer than the instance lived. With
`--checkpoint-minutes 10` and a SIGTERM handler added, the last S3 mirror
landed 6 minutes before termination and the loss was ~3k steps.

Cumulative on the weights: **~209,715 gradient steps, ~2.77M hands**
(100,000 from the first run's checkpoint plus 109,715 from the third; the
step counter restarts on resume, so the checkpoint's `steps` field is
run-local, not cumulative). Throughput held at 8.31 steps/s and 6,389
decisions/s, split 48% learn / 27% featurise / 18% forward / 3% env.

### The snapshot pool is wiped by every reclaim

`save()` persists `model`, `critic`, `net_config`, `config`, `steps` and
`hands` -- but not the snapshot deque, which `train2.py` initialises empty
on every start (`snapshots: deque[dict] = deque(maxlen=args.snapshot_pool)`).
So each resume begins with no frozen opponents and falls back to training
against copies of the current net until the pool refills.

At `--snapshot-every 10000` a 10-deep pool takes 100k steps to fill, and
reclaims are now cutting runs at ~110k steps. The agent therefore spends
much of its life against a shallow or empty opponent pool, which is
exactly the diversity the self-play mixture (65% self / 20% vs snapshot /
10% snapshot-as-partner / 5% vs bot) was designed to provide.

Fix is to serialise the deque into the checkpoint. Cost is size: ten
3.13M-parameter state dicts is ~125 MB on top of a 17 MB checkpoint, which
matters only because the file mirrors to S3 every 15 minutes. Storing
snapshots at a longer interval, or keeping fewer of them, would cap that.

**Fixed.** `save()` now serialises the deque and resume rebuilds it.
Snapshots are stored at half precision -- they are frozen opponents, never
the weights being trained, so a few thousandths of drift is imperceptible
and it halves a file that mirrors to S3 on a timer. A 10-deep pool costs
~18 MB on top of the 17 MB checkpoint, not the ~125 MB a naive fp32 copy
would have.

Three related resume bugs surfaced while testing it, all from the same
root -- state that lives only in the process:

* **The checkpoint write was not atomic.** `torch.save` overwrote
  `latest.pt` in place while the S3 mirror copies it on an independent
  15-minute timer, so the mirror could upload a half-written file. A
  truncated `latest.pt` is a *failed resume at the next boot*, which is
  the one failure the whole checkpointing scheme exists to prevent. Now
  written to `.tmp` and `os.replace`d.
* **`learner.steps` restarted at 0 on resume**, so `critic_warmup_steps`
  restarted too: for the first 2,000 steps after every reclaim the
  learner discarded its already-trained privileged critic and fell back
  to pure MC targets. Steps and hands are now cumulative, which also
  stops new snapshot labels colliding with restored ones and makes the
  checkpoint's `steps` field mean what it looks like it means.
* **A snapshot that did not match the current architecture crashed the
  run at startup.** On a spot box that is boot time, so the crash costs
  the entire run. Restored snapshots now load defensively and a bad one
  is dropped with a message.

Also stopped writing a numbered archive on the 10-minute checkpoint
trigger; it fires ~40x a run and nothing reads those files.

`py/test_snapshot_resume.py` covers the round trip: the pool survives a
save, comes back as float weights that load into a real net (max drift
< 1e-2), a pre-fix checkpoint still resumes with an empty pool, and no
temp file is left behind. Two end-to-end resumes confirm the rest --
a matching pool prints `opponent pool warm: [180000, 190000, 200000]`
and continues from step 209715, and a deliberately mismatched pool logs
`dropping snapshot at step 170000` and keeps running.

None of this warms the *current* box, which is mid-run on the old code
with poolless checkpoints. The next launch ships the fix and starts
banking snapshots; the reclaim after that is the first to resume warm.

## Shipping the agent (Phase 5)

The 417k checkpoint now plays in the app as the `rl-v2` bot, running as a
pure-TypeScript forward pass inside `@tractor/engine`.

**Why not a sidecar.** The ws server is a 256 MB `node:20-alpine` machine
on Fly. Torch does not fit, and `onnxruntime-node` ships glibc builds
only, so both the obvious answers were out. The whole net -- chain
encoding, the residual stem, the causal history attention, the per-role
points head -- is reimplemented in `packages/engine/src/rl/v2/`, with
weights exported to float16 (5.9 MB) by `py/export_v2_weights.py` and
widened to float32 at load. A decision costs ~18 ms, which no human
notices between plays.

**One encoder, not two.** `cards.ts`, `record.ts`, `candidates.ts` and
`encode.ts` moved out of `research/src/` into the engine; the research
files are now re-export shims. Two copies would drift, and a drift between
the encoding training saw and the encoding inference produces is *silent*
-- nothing crashes, the bot merely plays a worse game than the arena
measured. That failure mode is the reason for the move and for the test
below.

**Parity, 400 real decisions** (`py/test_ts_parity.py`):

```
ok  state features:   368,000 values bit-identical
ok  history tokens:   486,400 values bit-identical
ok  action features:  422,064 values bit-identical
ok  Q values: max abs diff 3.89e-04 (tolerance 2e-03)
    argmax: 399/400 decisions pick the same candidate
```

Features are required to be bit-identical -- both sides do the same
arithmetic on the same integers, so there is no tolerance to spend. The Q
values get a small one because the shipped weights are float16 and the two
forward passes accumulate in a different order. The single argmax
disagreement is a near-tie between two candidates.

**End to end, the shipped bot scores like the trained one.** Running the
TypeScript `rl-v2` through the arena as an ordinary `BotType`, same deal
base as the Python reading:

| runner | vs `smart-v13` |
| --- | --- |
| Python, `v2_latest.pt` | +38.17 [+31.87, +44.47] |
| **TypeScript, `rl-v2`** | **+34.64** [+29.31, +39.96] |

```
192 duplicate pairs (384 hands) in 7.9 min
  point margin: +34.64 pts [95% CI +29.31, +39.96]  (159 better / 29 worse / 4 level)
  level margin: +0.88 levels [95% CI +0.73, +1.03]
  attacker points: agent 100.1 / opponent 65.4 on the same deals
```

The intervals overlap across most of their width, which is what "the same
agent" looks like when the two runs cover different numbers of pairs. Note
the TypeScript path is also ~4x faster to evaluate here (7.9 min for 192
pairs against 19.3 for 123), because it skips the worker/learner round trip
entirely.

### The browser bundle forced the wiring

The first web build failed outright:

```
UnhandledSchemeError: Reading from "node:url" is not handled by plugins
  node:url -> rl/v2/weights.ts -> net.ts -> agent.ts -> bot.ts
           -> index.ts -> app/room/[code]/RoomClient.tsx
```

`RoomClient` is a *client* component and imports `VISIBLE_BOT_TYPES` and
`botTypeLabel` from the engine barrel, so anything `bot.ts` can reach ends
up in the browser bundle, where a `node:` import is a hard error. Tree
shaking does not help: webpack has to build the module before it can drop
it.

So `bot.ts` no longer imports the agent. It exposes `registerV2Agent`, and
Node entry points opt in with `import '@tractor/engine/rl/v2/register'`
(the ws server and `research/src/env.ts` both do). Selecting `rl-v2`
without that import throws a message naming the missing import rather than
quietly falling back to another bot -- a silent fallback here would look
exactly like "the neural bot is weak".

Room route after the change: 99.4 kB first-load JS, i.e. the weights are
nowhere near the browser.

### `rl-dmc` was never a real bot

The `rl-dmc` entry that has been in the lobby dropdown is a placeholder:
`rl/dmc/embeddedWeights.ts` is all zeros and was never overwritten by a
trained export, so every Q value tied and the bot played the first
enumerated candidate each turn. It stays in `BOT_TYPES` for the protocol
but `rl-v2` replaces it in `VISIBLE_BOT_TYPES`.

## Where the run ended (Phase 6)

Training stopped on its deadline at **1,103,832 steps / 14.5M hands**.
1,149,500 steps were actually computed across **14 instance lives** and 41.5
instance-hours at a sustained 7.7 steps/s; the 4% difference is what spot
reclaims cost, and it is small only because the resume path carries the
optimiser-free state, the cumulative counters and the snapshot pool.

### One opponent is not a measurement

The 966k checkpoint was first measured against `smart-v13` alone, scored
+34.11 against the 417k checkpoint's +38.17, and was written off as a
plateau. Run across nine opponents it is better on **eight of them**:

| opponent | 417k | 966k | 1104k |
| --- | --- | --- | --- |
| `random` | +108.93 | +112.12 | +107.19 |
| `lowball` | +107.26 | +108.48 | +112.91 |
| `point-hoarder` | +118.09 | +126.55 | +127.64 |
| `aggressive` | +133.25 | +137.52 | +137.73 |
| `smart` | +45.93 | +51.04 | +50.83 |
| `smart-v2` | +33.62 | +40.41 | +35.74 |
| `smart-v4` * | -7.52 | +1.67 | +3.46 |
| `smart-v9` * | -7.09 | -1.41 | -8.08 |
| `smart-v13` | +38.17 | +34.11 | +36.07 |

`* sees the real opponent hands.`

417k -> 966k: **+4.43 mean, better on 8 of 9**. `smart-v13` is the lone
dissenting rung and it happened to be the one measured first. A single
opponent cannot distinguish "the agent stopped improving" from "this
opponent cannot resolve the difference", and picking the wrong one produced
a confident, wrong conclusion. `py/ladder.py` exists so the comparison is
one command over the whole list on one deal base.

### The plateau is real, one checkpoint later

966k -> 1104k: **-0.78 mean, better on 5 of 9** -- scatter with no
direction, which is what re-measuring the same agent looks like. The last
137k steps bought nothing, so 966k stays deployed rather than being
replaced with newer weights of equal strength.

### `smart-v4` .. `smart-v12` are not fair-play opponents

`smart-v13` is defined in the engine as "`smart-v9` with opponent hands
masked", so every rung below it reads the real hands inside its search.
Those rows are a perfect-information reference, not a ladder rung. Read
that way, the agent went from losing to `smart-v4` by 7.5 points to
holding level with it -- level against an opponent that can see its cards
-- while still trailing `smart-v9` by 8. That residual gap is the value of
perfect information and no amount of self-play will close it.

### What is left

More gradient steps are no longer the lever. The two that remain are a
search that actually pays -- the shipped one measured +0.82 [-4.52, +6.16]
against the bare network, i.e. nothing, and was never re-tested after its
configuration was corrected to the validated `worlds=4` -- and expert
iteration, training on the search's choices rather than the policy's own.

## The plateau was a decline, and replay ratio is why

Measured 2026-09-11. All numbers below are 1000 duplicate pairs against
`smart-v2` from deal base 424200 -- a +/-2.5 point CI, where the 120-pair
runs used earlier give +/-7 and cannot resolve anything under ~10 points.

| checkpoint | steps | replay | margin vs `smart-v2` |
| --- | --- | --- | --- |
| `v2_final` | 1,103,832 | ~1.6 | **+32.88** [+30.41, +35.35] |
| `anneal_ctl` | 1,300,000 | ~2.5 | +24.00 [+21.47, +26.53] |
| `anneal_a65` | 1,300,000 | ~2.5 | +24.13 [+21.70, +26.55] |
| `anneal_a10` | 1,300,000 | ~2.5 | +17.66 [+15.00, +20.31] |
| `xpA` (cross-play) | 579,366 | ~9.0 | **-29.76** |

Three arms resumed from `v2_final` and every one finished ~9 points *worse*
196k steps later, confidence intervals disjoint. The 1104k checkpoint was
not a ceiling the run had reached; it was the last good point before a
decline. The one systematic difference between how the run arrived at
1104k and how the arms left it is `--steps-per-train`, 1 against 2, which
moves the replay ratio from ~1.6 to ~2.5.

The cross-play pair is the same effect at a larger dose. Two agents were
trained from scratch on one 8-vCPU box with three workers each, so the GPU
learner outran its data and reused every sample nine times; both ended up
*worse than `smart-v2`*, sixty points below the plateau, after 579k steps.

### Why the replay sweep got this wrong

`--replay-sweep` reported "nothing gained past `steps_per_train=2`" from
throughput and loss curves. Those metrics invert: `loss_q` **falls** as
replay rises, because a network reusing a narrow buffer memorises it.
`xpA` sat at `loss_q` 0.006-0.009 where the healthy lineage runs 0.027.
Read as convergence, it was overfitting. Replay has to be judged on the
ladder; loss curves actively mislead about it.

## Annealing the privileged teacher out: refuted

`lambda_mc` was made a schedule (`--lambda-mc-final`) after noticing that
the target blend is algebraically a regulariser: `MSE(q, l*G + (1-l)*P)`
equals `l*MSE(q,G) + (1-l)*MSE(q,P)` up to a constant, so the shipped
l=0.3 penalises disagreement with the teacher 7:3 *against the data*.

At a matched 1,300,000 steps, l=0.3 and l=0.65 are indistinguishable
(+24.00 vs +24.13, a 0.13 point difference) and l=1.0 -- no teacher -- is
significantly worse (+17.66, CI disjoint from the control). The teacher is
earning its weight and the FPDou-style worry about its bias binding late
does not hold here. The weight is not load-bearing at 0.7 though: half of
it can go for free.

A warning about sample size, since it nearly produced a wrong answer. At
120 pairs `a65` led the control by 6.7 points *and* led on all three rungs.
At 1000 pairs the lead was 0.13. Correlated rungs do not make a weak
signal strong.

### Confirmed: the replay ratio caused it

The controlled run. Same start, same box family, same everything the three
annealing arms used, with `--steps-per-train 1` instead of 2. Matched at
two step counts against `smart-v2`, 1000 duplicate pairs, deal base 424200:

| run | `steps_per_train` | replay | @1,250,000 | @1,300,000 |
| --- | --- | --- | --- | --- |
| `v2_final` (start, 1,103,832) | -- | -- | +32.88 [+30.41, +35.35] | -- |
| `replay1` | 1 | 0.82 | **+30.91** [+28.54, +33.29] | **+30.54** [+28.15, +32.93] |
| `anneal_ctl` | 2 | ~2.5 | +25.37 [+23.03, +27.70] | +24.00 [+21.47, +26.53] |

Confidence intervals are disjoint at both step counts, +5.54 and +6.54.
The only difference between the two runs is how many gradient steps are
taken per training round.

So the decline was not continued training, and not the privileged teacher.
At a healthy replay ratio the same 196k steps are **flat** -- +30.54
against the start's +32.88, intervals nearly touching -- which makes 1104k
a genuine plateau rather than a peak being fallen from. At replay ~2.5 the
same steps cost nine points, and at ~9.0 they cost sixty.

Dose-response, four runs:

| replay | 196k steps from `v2_final` |
| --- | --- |
| 0.82 | flat (-2.3, marginal) |
| ~2.5 | -8.9 |
| ~9.0 | -60 (from scratch, worse than `smart-v2`) |

**Practical consequence.** `--steps-per-train` is not a throughput knob, it
is the most destructive hyperparameter measured in this project so far. It
was raised to 2 on the strength of a sweep that scored it on loss curves,
and every run since has been quietly losing ground. Keep it at 1, and
score any future change to it on the ladder.

## The batched GEMM works, and buys nothing

Built `tractor_rl/stacked.py`: N agents as one module, every Linear a
`[N, in, out]` weight driven by `baddbmm`, including the causal history
attention (projections as per-agent bmms, the agent axis folded into the
batch for `scaled_dot_product_attention`). Parity with N separate
`TractorNet`s holds to 1e-16 in float64 and 1e-8 in float32, outputs and
gradients, on CPU and CUDA. It is correct. On the network we actually train
it is also pointless.

T4, batch 512 per agent, full forward and backward:

| N | separate ms | stacked ms | speedup | cost vs N=1 |
| --- | --- | --- | --- | --- |
| 1 | 13.77 | 15.26 | 0.90x | 1.00x |
| 4 | 55.55 | 57.08 | 0.97x | 3.74x |
| 8 | 108.32 | 112.25 | 0.97x | 7.35x |
| 16 | 215.50 | 217.22 | 0.99x | 14.23x |

Same measurement with `--history none`:

| N | separate ms | stacked ms | speedup | cost vs N=1 |
| --- | --- | --- | --- | --- |
| 1 | 6.64 | 5.90 | 1.13x | 1.00x |
| 8 | 66.61 | 23.17 | **2.87x** | **3.93x** |

So the batching does exactly what it was designed to do -- the MLP path
goes from 8x to 3.93x -- and then the causal history encoder swamps it.
Attention is **61% of a step at N=1 and 79% at N=8**, and it scales 9.52x
from N=1 to N=8, slightly *worse* than linear. It is compute-bound, not
launch-bound: it runs over 16 tokens per row, so 8192 tokens where the
trunk sees 512, and no arrangement of the agent axis changes how much
arithmetic that is.

**This corrects an earlier claim in this file.** The 5.69x figure came from
`bench_multiagent.py`, which covered the MLP path and said so, and was read
as "the batched GEMM is a requirement of the population design, not an
optimisation". With attention included it is neither: a plain Python loop
over 8 agents on a T4 runs at 73.9 agent-steps/s against the stacked net's
71.3. The population needs 3.53 steps/s per agent -- 28.2 aggregate -- so
**N=8 is already viable on one GPU with no batching at all**. What made the
population run slow was that it ran on a CPU, where it managed 0.99.

### The real lever is the history encoder

Dropping it takes N=8 from 112.25 ms to 23.17 ms a step, 71 agent-steps/s
to 345. Its value has never been measured: it was added on the argument
that planes say *what* was played but not by whom or when, and nothing has
ever tested a run without it against the ladder. At 61-79% of every
gradient step in the project, that is now the most expensive unmeasured
assumption in the stack.

## Pre-registration: does the move-history encoder earn its cost? (run 2)

Written before the run finished, so the analysis cannot be chosen to suit
the numbers. Launched 2026-09-15 03:08 UTC, two 12-hour fleets.

**What the first attempt could and could not show.** Arms level at equal
machine-hours: `attn` +17.05 [14.50, 19.59] at 195,514 steps, `none` +18.04
[15.62, 20.45] at 352,797, 1000 pairs against `smart-v2`. Belief accuracy
identical (0.8803 vs 0.8799), which is the most direct evidence available
that the encoder is not doing the hidden-hand inference it was added for --
`--history none` removes the *move-sequence encoder*, not memory; the card
planes still record everything played, only who played it and when is lost.

It could not separate sample efficiency from speed, because only `_latest`
was mirrored and the arms were therefore only ever observed at different
step counts. And neither converged: both scored +17-18 where the deployed
agent scores +31.81, so the comparison lives in early training, where
attribution plausibly matters less than it does among fine distinctions.

**Design of run 2.** Identical arms except `--history`, same seed, same
`--steps-per-train 1`, same instance family, 12 hours. Checkpoints mirrored
at 50k, 100k, 200k, 300k, 400k, 500k and 600k steps, so both arms can be
read at **matched steps** and at **matched wall clock** -- the latter from
S3's `LastModified`, which survives the resume that resets the in-process
elapsed counter. Every evaluation is 1000 duplicate pairs from base 424200
against `smart-v2` and `smart-v4`.

**Pre-registered readings:**

| outcome | conclusion |
| --- | --- |
| `none` matches `attn` at matched steps | the encoder adds nothing |
| `attn` > `none` at steps, not at hours | more sample-efficient, still not worth 61% of a step |
| `attn` > `none` at both | it earns its cost; keep it |
| `none` > `attn` at matched steps | something is wrong; investigate before believing it |

Harness is `py/hist2_eval.py`, written before the results existed.

### Result: keep the encoder

The rerun answered the question the first attempt could not. Against
`smart-v2`, 1000 duplicate pairs from base 424200, at matched step counts:

| steps | attn | none | attn's edge |
| --- | --- | --- | --- |
| 50,000 | +4.59 [+2.1, +7.1] | +3.58 [+1.2, +5.9] | +1.01 |
| 100,000 | +12.66 [+10.2, +15.1] | +8.73 [+6.3, +11.1] | +3.93 |
| 200,000 | +18.35 [+15.9, +20.8] | +15.59 [+13.1, +18.1] | +2.76 |
| 300,000 | +23.57 [+21.2, +25.9] | +14.19 [+11.7, +16.7] | +9.38 |

**The encoder is more sample-efficient**, ahead at every matched step on
`smart-v2` and at three of four on `smart-v4`. The first attempt's null was
an artifact of observing the arms at 195k and 353k steps and calling that a
comparison.

On machine-hours the two are level -- `attn` runs 8.4 steps/s against
`none`'s 15.9, so at ten training-hours it is +23.57 at 300k against
+24.94 at 600k, intervals overlapping. That is pre-registered row 2, and on
that basis alone the encoder does not pay. Three things argue against
acting on it, and the decision was to **keep the encoder**:

* `attn` is still climbing steeply at its last point (+18.35 -> +23.57 over
  100k steps) where `none`'s last two are flat (+24.72 -> +24.94). Neither
  is near the deployed +31.81, and the curves are not done.
* The 300k `none` checkpoint is contaminated: worse than its own 200k on
  *both* rungs, with a restart logged at 265,408 steps. It drives the
  largest matched-step gap; without it the edge is ~2.6 points.
* The wall-clock axis is dirty in the other direction -- `attn` was
  reclaimed seven times to `none`'s three, so its machine-hour efficiency
  is understated.

The cheap, general lesson is the one below.

## Optimiser state now survives a checkpoint

`train2.py` saved model and critic weights but not RMSprop's second
moments, so **every resume restarted the optimiser cold**. On interruptible
hardware that is not an edge case: the `attn` arm above was reclaimed seven
times in twelve hours, and the `none` arm's checkpoint taken 35k steps
after one such reset measured worse than its own checkpoint 100k steps
earlier, on two independent opponents.

Both `train2.py` and `train_pop.py` (per agent) now persist `opt` and
`critic_opt` and restore them after the optimisers exist. A checkpoint
predating the field still loads and starts cold, and a shape mismatch is
logged rather than fatal -- losing the moments is a transient, losing a
spot run at boot is not. Cost is **+8.1 MB on a 16.1 MB checkpoint**, and
`py/test_snapshot_resume.py` checks the moments round-trip element-wise
after real gradient steps.
