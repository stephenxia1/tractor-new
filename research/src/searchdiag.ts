// Why did belief weighting change nothing?
//
// A null A/B has two very different explanations: the prior is useless, or
// the experiment could not have detected it. This separates them by asking
// how much room search had to differ at all -- how often it even runs, how
// often it overrides the network's own ranking, and how often the two
// samplers end up on different cards.
import { GameSlot } from './env';
import { makeV2SearchAgent } from '@tractor/engine/rl/v2/searchAgent';
import { DEFAULT_BELIEF_WEIGHTS } from '@tractor/engine/rl/v2/beliefSampler';
import { pickBotAction } from '@tractor/engine';

const target = Number(process.argv[2] ?? 150);
const worlds = Number(process.argv[3] ?? 2);
const maxSearched = Number(process.argv[4] ?? 8);

const fallback = (s: any, seat: any) => pickBotAction(s, seat, 'smart-v34');
const weighted = makeV2SearchAgent({ worlds, maxSearched, belief: DEFAULT_BELIEF_WEIGHTS, fallback });
const uniform = makeV2SearchAgent({ worlds, maxSearched, belief: null, fallback });

const slot = new GameSlot({
  gameId: 0, seed: 20260907,
  learnerSeats: [true, true, true, true],
  baselineBot: 'smart-v34', prePlayBot: 'smart-v34',
  maxCandidates: 48, randomize: true,
});

const key = (a: any) => (a.kind === 'play_cards' ? a.cards.map((c: any) => JSON.stringify(c)).sort().join('|') : a.kind);
let n = 0, same = 0, guard = 0;
const t0 = Date.now();
while (n < target && guard++ < 100_000) {
  const { pending } = slot.advance();
  if (pending === null) continue;
  const a = weighted(slot.gameState, pending.seat);
  const b = uniform(slot.gameState, pending.seat);
  if (key(a) === key(b)) same += 1;
  n += 1;
  slot.choose(n % pending.candidates.length);
}
const secs = (Date.now() - t0) / 1000;
const w = weighted.stats, u = uniform.stats;
process.stdout.write(
`decisions compared        ${n}   (${(secs / n * 1000).toFixed(0)} ms per pair of decisions)
search actually ran on    ${w.searched}/${w.decisions}  (${(w.searched / Math.max(1, w.decisions) * 100).toFixed(1)}%)
  skipped: candidates > maxSearched(${maxSearched}), or a candidate reached no leaf
search overrode the net   weighted ${w.overrides}/${Math.max(1, w.searched)} (${(w.overrides / Math.max(1, w.searched) * 100).toFixed(1)}%)
                          uniform  ${u.overrides}/${Math.max(1, u.searched)} (${(u.overrides / Math.max(1, u.searched) * 100).toFixed(1)}%)
leaves evaluated          weighted ${w.leaves}, uniform ${u.leaves}
relaxed worlds            weighted ${w.relaxedWorlds}, uniform ${u.relaxedWorlds}
SAME final card played    ${same}/${n}  (${(same / Math.max(1, n) * 100).toFixed(1)}%)
`);
