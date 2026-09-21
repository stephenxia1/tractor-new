// Dumps the engine's classification and decomposition for randomly
// generated single-suit plays, so the Python rules package can be
// checked against the engine rather than against a second reading of
// rules.md.
//
//   npx tsx src/dumpCombos.ts 20000 > combos.json

import {
  classifyCombo,
  decomposePlay,
  effectiveSuit,
  mulberry32,
  NATURAL_RANKS,
  SUITS,
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

type Comp = { kind: string; top: number; length: number; n: number };

function encode(c: ReturnType<typeof classifyCombo>): Comp | null {
  if (!c.ok) return null;
  const k = c.combo;
  if (k.kind === 'single') {
    return { kind: 'single', top: k.topPosition, length: 1, n: 1 };
  }
  if (k.kind === 'nkind') {
    return { kind: 'nkind', top: k.topPosition, length: 1, n: k.n };
  }
  return { kind: 'tractor', top: k.topPosition, length: k.length, n: k.n };
}

function main(): void {
  const count = Number(process.argv[2] ?? '20000');
  const rand = mulberry32(0xc0ffee);
  const out: Array<{
    levelRank: number;
    trumpSuit: number;
    slots: number[];
    classified: Comp | null;
    decomposed: Comp[] | null;
  }> = [];

  while (out.length < count) {
    const lr = Math.floor(rand() * NATURAL_RANKS.length);
    const ts = Math.floor(rand() * 5);
    const trumpSuit: Suit | null = ts === 4 ? null : SUITS[ts]!;
    const ctx: TrumpContext = { trumpRank: NATURAL_RANKS[lr]!, trumpSuit };

    // All types sharing one effective suit, so the play is well formed.
    const target = Math.floor(rand() * 5);
    const pool: number[] = [];
    for (let slot = 0; slot < NUM_TYPES; slot++) {
      const es = effectiveSuit(cardOfSlot(slot, 0), ctx);
      const idx = es === 'trump' ? 4 : SUITS.indexOf(es as Suit);
      if (idx === target) pool.push(slot);
    }
    if (pool.length === 0) continue;

    // Draw a handful of types, each once or twice.
    const want = 1 + Math.floor(rand() * 8);
    const cards: Card[] = [];
    const slots: number[] = [];
    const shuffled = [...pool];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
    }
    for (const slot of shuffled) {
      if (cards.length >= want) break;
      const copies = rand() < 0.55 ? 2 : 1;
      for (let d = 0; d < copies && cards.length < want; d++) {
        cards.push(cardOfSlot(slot, d));
        slots.push(slot);
      }
    }
    if (cards.length === 0) continue;

    const cls = classifyCombo(cards, ctx, DECKS);
    const dec = decomposePlay(cards, ctx, DECKS);
    out.push({
      levelRank: lr,
      trumpSuit: ts,
      slots,
      classified: encode(cls),
      decomposed:
        dec === null
          ? null
          : dec.components.map((c) => {
              if (c.kind === 'single') {
                return { kind: 'single', top: c.topPosition, length: 1, n: 1 };
              }
              if (c.kind === 'nkind') {
                return { kind: 'nkind', top: c.topPosition, length: 1, n: c.n };
              }
              return {
                kind: 'tractor',
                top: c.topPosition,
                length: c.length,
                n: c.n,
              };
            }),
    });
  }

  process.stdout.write(JSON.stringify({ decks: DECKS, cases: out }));
}

main();
