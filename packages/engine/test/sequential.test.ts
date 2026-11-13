import { describe, expect, it } from 'vitest';

import { applyEvent, createGame, defaultV1Config, nextToPlay, pickBotAction } from '../src';
import type { BotType } from '../src/bot';
import type { GameState, Seat } from '../src/state';
import type { Card } from '../src/types';
import { fairLegalPlays, playSignature } from '../src/rl/v2/oracle';
import { orderKey, SequentialPlay } from '../src/rl/v2/sequential';

// Every play decision from a set of simulated hands, with the landlord's
// buried kitty (which the landlord legitimately knows).
function* playPositions(
  hands: number,
  bots: ReadonlyArray<BotType>,
  seed0: number,
): Generator<{ state: GameState; seat: Seat; myKitty?: ReadonlyArray<Card> }> {
  let seed = seed0;
  for (let h = 0; h < hands; h++) {
    let s = createGame(defaultV1Config());
    s = applyEvent(s, { kind: 'start_hand', seed: seed++ }).state;
    let guard = 0;
    while (s.phase !== 'between_hands' && s.phase !== 'game_over' && guard++ < 3000) {
      if (s.phase === 'trick_review') {
        s = applyEvent(s, { kind: 'advance_trick' }).state;
        continue;
      }
      if (s.phase === 'declare_lockout') {
        s = applyEvent(s, { kind: 'release_lockout', seed: seed++ }).state;
        continue;
      }
      let acted = false;
      for (let seat = 0; seat < 4 && !acted; seat++) {
        if (s.phase === 'play') {
          if (nextToPlay(s) !== seat) continue;
          const hand = s.currentHand!;
          const myKitty = seat === hand.landlord ? hand.kitty : undefined;
          yield { state: s, seat, myKitty };
        }
        const bot = bots[(seat + h) % bots.length]!;
        const a = pickBotAction(s, seat, bot);
        if (a.kind === 'wait') continue;
        const ev = a.kind === 'pick_up_kitty' ? { kind: a.kind, seat } : { ...a, seat };
        const r = applyEvent(s, ev as never);
        if (r.ok) {
          s = r.state;
          acted = true;
        }
      }
      if (!acted) break;
    }
  }
}

type Walk = {
  submitted: Map<string, number>; // signature -> times reached
  deadEnds: number;
  nonCanonical: number;
  plays: Card[][];
};

function walk(seq: SequentialPlay, state: GameState, budget = 200_000): Walk | null {
  const hand = state.currentHand!;
  const ctx = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };
  const out: Walk = { submitted: new Map(), deadEnds: 0, nonCanonical: 0, plays: [] };
  const stack: number[][] = [[]];
  let visited = 0;
  while (stack.length > 0) {
    if (visited++ > budget) return null;
    const prefix = stack.pop()!;
    const opts = seq.options(prefix);
    const complete = seq.mode === 'follow' ? prefix.length === seq.size : opts.canSubmit;
    if (complete) {
      const cards = seq.cards(prefix);
      const sig = playSignature(cards);
      out.submitted.set(sig, (out.submitted.get(sig) ?? 0) + 1);
      out.plays.push(cards);
    }
    if (!complete && opts.next.length === 0) out.deadEnds += 1;
    const last = prefix.length > 0 ? orderKey(prefix[prefix.length - 1]!, ctx) : Infinity;
    for (const c of opts.next) {
      if (orderKey(c, ctx) > last) out.nonCanonical += 1;
      stack.push([...prefix, c]);
    }
  }
  return out;
}

function check(hands: number, bots: ReadonlyArray<BotType>, seed: number) {
  const stats = {
    decisions: 0,
    leads: 0,
    follows: 0,
    multiFollows: 0,
    skipped: 0,
    plays: 0,
    reducerChecked: 0,
  };
  for (const { state, seat, myKitty } of playPositions(hands, bots, seed)) {
    stats.decisions += 1;
    const oracle = fairLegalPlays(state, seat, { myKitty, limit: 60_000 });
    if (!oracle.complete) {
      stats.skipped += 1;
      continue;
    }
    const seq = new SequentialPlay(state, seat, { myKitty });
    if (seq.mode === 'lead') stats.leads += 1;
    else {
      stats.follows += 1;
      if ((seq.size ?? 0) > 1) stats.multiFollows += 1;
    }
    const w = walk(seq, state);
    if (w === null) {
      stats.skipped += 1;
      continue;
    }
    const where = `seed ${seed}, decision ${stats.decisions} (${seq.mode})`;
    expect(w.deadEnds, `dead ends at ${where}`).toBe(0);
    expect(w.nonCanonical, `non-canonical options at ${where}`).toBe(0);
    for (const [sig, times] of w.submitted) {
      expect(times, `${sig} reached ${times} times at ${where}`).toBe(1);
    }
    const reached = [...w.submitted.keys()].sort();
    const legal = [...oracle.plays.keys()].sort();
    const missing = legal.filter((s) => !w.submitted.has(s));
    const extra = reached.filter((s) => !oracle.plays.has(s));
    expect(missing, `unreachable legal plays at ${where}`).toEqual([]);
    expect(extra, `reachable illegal plays at ${where}`).toEqual([]);
    stats.plays += reached.length;

    // The reducer accepts what the walk produces, and a throw never draws
    // the forced-lead penalty: offered throws are safe by construction.
    const sample = w.plays.length <= 12 ? w.plays : w.plays.filter((_, i) => i % Math.ceil(w.plays.length / 12) === 0);
    for (const cards of sample) {
      const r = applyEvent(state, { kind: 'play_cards', seat, cards });
      expect(r.ok, `reducer rejected ${playSignature(cards)} at ${where}`).toBe(true);
      expect(r.ok && r.notice?.kind === 'forced_lead', `forced lead at ${where}`).toBe(false);
      stats.reducerChecked += 1;
    }
  }
  return stats;
}

describe('sequential action space vs the brute-force oracle', () => {
  it('heuristic play: complete, sound, unique, no dead ends', () => {
    const stats = check(24, ['smart', 'smart-v2', 'aggressive', 'point-hoarder'], 101);
    console.log('[sequential/heuristic]', JSON.stringify(stats));
    expect(stats.skipped).toBeLessThan(stats.decisions * 0.02);
    expect(stats.multiFollows).toBeGreaterThan(0);
  }, 300_000);

  it('random play reaches odd positions too', () => {
    const stats = check(16, ['random', 'smart', 'random', 'lowball'], 707);
    console.log('[sequential/random]', JSON.stringify(stats));
    expect(stats.skipped).toBeLessThan(stats.decisions * 0.02);
  }, 300_000);
});
