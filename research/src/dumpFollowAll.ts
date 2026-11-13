// Like dumpFollow.ts, but every distinct follow of the right size is
// judged, not a random sample of them. Random probes rarely land on the
// few legal follows of a structured position, which is exactly where an
// obligation rule can differ, so the exhaustive form is what finds a
// disagreement. Leads also include throws that carry a tractor, and
// trump is led more often.
//
//   npx tsx src/dumpFollowAll.ts 600 > follow_all.json
//
// Hands are generated structure-rich on purpose: tractors alongside
// isolated pairs are what separates the obligation rules, and random
// deals produce them too rarely to cover.
//
//   npx tsx src/dumpFollow.ts 1200 > follow.json

import {
  classifyCombo,
  decomposePlay,
  effectiveSuit,
  mulberry32,
  NATURAL_RANKS,
  SUITS,
  validateFollow,
  validateFollowThrow,
  type Card,
  type Suit,
  type TrumpContext,
} from '@tractor/engine';

const NUM_TYPES = 54;
const DECKS = 2;

function cardOfSlot(slot: number, deckId: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId };
  return {
    kind: 'standard',
    suit: SUITS[Math.floor(slot / 13)]!,
    rank: NATURAL_RANKS[slot % 13]!,
    deckId,
  };
}

// Expand a slot->count map into concrete cards with distinct deck ids.
function cardsOf(counts: Map<number, number>): Card[] {
  const out: Card[] = [];
  for (const [slot, k] of counts) {
    for (let d = 0; d < k; d++) out.push(cardOfSlot(slot, d));
  }
  return out;
}

function slotsOf(counts: Map<number, number>): number[] {
  const out: number[] = [];
  for (const [slot, k] of counts) for (let d = 0; d < k; d++) out.push(slot);
  return out;
}

function main(): void {
  const want = Number(process.argv[2] ?? '1200');
  const rand = mulberry32(0x5eed1234);
  const cases: Array<{
    levelRank: number;
    trumpSuit: number;
    ledSuit: number;
    lead: number[];
    hand: number[];
    probes: Array<{ cand: number[]; ok: boolean }>;
  }> = [];

  let guard = 0;
  while (cases.length < want && guard++ < want * 60) {
    const lr = Math.floor(rand() * NATURAL_RANKS.length);
    const ts = Math.floor(rand() * 5);
    const trumpSuit: Suit | null = ts === 4 ? null : SUITS[ts]!;
    const ctx: TrumpContext = { trumpRank: NATURAL_RANKS[lr]!, trumpSuit };

    const ledSuit = rand() < 0.5 ? 4 : Math.floor(rand() * 4);
    const pool: number[] = [];
    for (let slot = 0; slot < NUM_TYPES; slot++) {
      const es = effectiveSuit(cardOfSlot(slot, 0), ctx);
      const idx = es === 'trump' ? 4 : SUITS.indexOf(es as Suit);
      if (idx === ledSuit) pool.push(slot);
    }
    if (pool.length < 5) continue;

    // Lead: a pair, a tractor, or a throw of two structures.
    const roll = rand();
    const leadCounts = new Map<number, number>();
    const sorted = [...pool].sort((a, b) => a - b);
    const at = (i: number): number => sorted[i % sorted.length]!;
    const base = Math.floor(rand() * sorted.length);
    if (roll < 0.25) {
      leadCounts.set(at(base), 2);
    } else if (roll < 0.55) {
      leadCounts.set(at(base), 2);
      leadCounts.set(at(base + 1), 2);
    } else if (roll < 0.8) {
      leadCounts.set(at(base), 2);
      leadCounts.set(at(base + 1), 2);
      leadCounts.set(at(base + 2), 2);
    } else if (roll < 0.9) {
      leadCounts.set(at(base), 2);
      leadCounts.set(at(base + 1), 2);
      leadCounts.set(at(base + 4), 2);
      leadCounts.set(at(base + 5), 2);
    } else {
      // A tractor plus a separate pair or single: a throw with structure.
      leadCounts.set(at(base), 2);
      leadCounts.set(at(base + 1), 2);
      leadCounts.set(at(base + 3), rand() < 0.5 ? 2 : 1);
    }
    const leadCards = cardsOf(leadCounts);
    const leadDecomp = decomposePlay(leadCards, ctx, DECKS);
    if (leadDecomp === null) continue;
    const leadCls = classifyCombo(leadCards, ctx, DECKS);

    // Hand: a structure-rich holding in the led suit, plus off-suit
    // filler so short holdings are exercised too.
    const handCounts = new Map<number, number>();
    const shuffled = [...pool];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const jj = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[jj]] = [shuffled[jj]!, shuffled[i]!];
    }
    const distinct = 3 + Math.floor(rand() * 5);
    for (const slot of shuffled.slice(0, distinct)) {
      handCounts.set(slot, rand() < 0.6 ? 2 : 1);
    }
    for (let slot = 0; slot < NUM_TYPES && handCounts.size < distinct + 4; slot++) {
      const es = effectiveSuit(cardOfSlot(slot, 0), ctx);
      const idx = es === 'trump' ? 4 : SUITS.indexOf(es as Suit);
      if (idx === ledSuit || handCounts.has(slot)) continue;
      if (rand() < 0.3) handCounts.set(slot, 1);
    }
    const handCards = cardsOf(handCounts);
    const leadCount = leadCards.length;
    if (handCards.length < leadCount) continue;

    // Probes: every distinct multiset of the hand of the right size.
    const probes: Array<{ cand: number[]; ok: boolean }> = [];
    const slotsHeld = [...handCounts.entries()];
    const cur: number[] = [];
    let tooMany = false;
    const rec = (i: number, left: number): void => {
      if (tooMany) return;
      if (left === 0) {
        if (probes.length >= 20000) { tooMany = true; return; }
        const counts = new Map<number, number>();
        for (const s of cur) counts.set(s, (counts.get(s) ?? 0) + 1);
        const pick = cardsOf(counts);
        const v = leadCls.ok
          ? validateFollow(leadCls.combo, pick, handCards, ctx, DECKS)
          : validateFollowThrow(leadDecomp, pick, handCards, ctx, DECKS);
        probes.push({ cand: [...cur], ok: v.ok });
        return;
      }
      if (i >= slotsHeld.length) return;
      const [slot, have] = slotsHeld[i]!;
      for (let take = Math.min(have, left); take >= 0; take--) {
        for (let t = 0; t < take; t++) cur.push(slot);
        rec(i + 1, left - take);
        for (let t = 0; t < take; t++) cur.pop();
      }
    };
    rec(0, leadCount);
    if (tooMany) continue;
    if (probes.length === 0) continue;

    cases.push({
      levelRank: lr,
      trumpSuit: ts,
      ledSuit,
      lead: slotsOf(leadCounts),
      hand: slotsOf(handCounts),
      probes,
    });
  }

  process.stdout.write(JSON.stringify({ decks: DECKS, cases }));
}

main();
