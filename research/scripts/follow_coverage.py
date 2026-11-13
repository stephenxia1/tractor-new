"""How much of the legal follow set the training simulator offers.

    python scripts/follow_coverage.py [hands] [cap] [uniform|big]    (from research/)

Plays random deals in `py/tractor/sim` and, at every follow, compares the
MoveSet the network is shown (`legal_moves(cap)`; training passes the
model's `max_candidates`, 128) with the exact legal set, built the same
way without the cap. Reports how often the offered set is incomplete,
how often it falls back to the single canonical follow, and how often a
follow that would take the trick as it stands is legal but not offered,
by lead size. `big` weights the random policy toward the largest plays,
so tractor and throw leads, where the caps bind, are better sampled.

The evaluation host offers the engine's candidates instead
(`src/followCoverage.ts` measures those), so the two numbers together say
whether training and evaluation show the network the same moves.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))
from tractor.rules.cards import Ctx, NUM_TYPES  # noqa: E402
from tractor.rules.combos import covers, decompose, play_rank, play_suit  # noqa: E402
from tractor.rules.follow import lead_obligations  # noqa: E402
from tractor.sim.actions import legal_follows  # noqa: E402
from tractor.sim.state import PlayState  # noqa: E402
HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 64
POLICY = sys.argv[3] if len(sys.argv) > 3 else 'uniform'
EXACT_CAP = 40000  # exact enumeration ceiling; above it the decision is skipped
rng = np.random.default_rng(7)
DECK = np.repeat(np.arange(NUM_TYPES), 2)

def key(m): return m.tobytes()

def winning(state, moves):
    lead = state.trick_plays[0]
    parts = decompose(lead, state.ctx, state.num_decks); suit = play_suit(lead, state.ctx)
    best = play_rank(parts, suit)
    for p in state.trick_plays[1:]:
        if covers(p, parts, suit, state.ctx, state.num_decks):
            r = play_rank(decompose(p, state.ctx, state.num_decks), play_suit(p, state.ctx))
            best = max(best, r)
    out = []
    for m in moves:
        if covers(m, parts, suit, state.ctx, state.num_decks):
            r = play_rank(decompose(m, state.ctx, state.num_decks), play_suit(m, state.ctx))
            out.append(r > best)
        else:
            out.append(False)
    return out

st = defaultdict(int); cov_sum = defaultdict(float); by_k = defaultdict(lambda: defaultdict(int))
t0 = time.time()
for h in range(HANDS):
    perm = rng.permutation(DECK)
    hands = np.zeros((4, NUM_TYPES), dtype=np.int8)
    for s in range(4):
        np.add.at(hands[s], perm[s * 25:(s + 1) * 25], 1)
    kitty = np.zeros(NUM_TYPES, dtype=np.int8); np.add.at(kitty, perm[100:], 1)
    ctx = Ctx(int(rng.integers(13)), int(rng.integers(5)))
    state = PlayState.new(ctx, hands, kitty, int(rng.integers(4)), 2)
    while not state.done:
        capped = state.legal_moves(cap=CAP)
        if not state.is_lead:
            k = int(state.trick_plays[0].sum())
            parts, suit = state.lead_components()
            ob = lead_obligations(parts, state.num_decks)
            exact = legal_follows(ob, k, suit, state.hands[state.to_act], ctx, 2, cap=EXACT_CAP)
            st["follows"] += 1
            by_k[k]["follows"] += 1
            if exact.unenumerated:
                st["exact_too_big"] += 1
            else:
                ex = {key(m): m for m in exact.moves}
                cp = {key(m) for m in capped.moves}
                n_ex = len(ex); n_in = sum(1 for kk in ex if kk in cp)
                if capped.unenumerated:
                    st["fallback"] += 1; by_k[k]["fallback"] += 1
                if n_in < n_ex:
                    st["incomplete"] += 1; by_k[k]["incomplete"] += 1
                cov_sum["frac"] += n_in / n_ex
                if k > 1:
                    st["multi"] += 1; cov_sum["frac_multi"] += n_in / n_ex
                    exm = list(ex.values()); w = winning(state, exm)
                    if any(w):
                        st["win_avail"] += 1; by_k[k]["win_avail"] += 1
                        if not any(wi and key(m) in cp for m, wi in zip(exm, w)):
                            st["win_lost"] += 1; by_k[k]["win_lost"] += 1
        if POLICY == 'big':
            w = np.array([float(m.sum()) ** 2 for m in capped.moves]); w /= w.sum()
            mv = capped.moves[int(rng.choice(len(capped.moves), p=w))]
        else:
            mv = capped.moves[int(rng.integers(len(capped.moves)))]
        state.step(mv)
el = time.time() - t0
f = lambda a, b: f"{100 * a / max(b, 1):.1f}%"
print(f"{HANDS} hands, cap {CAP}, policy {POLICY}, {el:.0f}s")
print(f"follows {st['follows']}; multi-card {st['multi']}; exact set too big to build {st['exact_too_big']}")
print(f"capped set incomplete: {st['incomplete']} ({f(st['incomplete'], st['follows'] - st['exact_too_big'])}); "
      f"fallback to one canonical follow: {st['fallback']}")
print(f"mean coverage: all follows {cov_sum['frac'] / max(1, st['follows'] - st['exact_too_big']):.3f}, multi-card {cov_sum['frac_multi'] / max(1, st['multi']):.3f}")
print(f"multi-card follows with a legal trick-winning follow: {st['win_avail']}; none offered: {st['win_lost']} ({f(st['win_lost'], st['win_avail'])})")
for k in sorted(by_k):
    d = by_k[k]
    if k > 1: print(f"  lead size {k}: follows {d['follows']}, incomplete {d['incomplete']}, fallback {d['fallback']}, win available {d['win_avail']}, win lost {d['win_lost']}")
