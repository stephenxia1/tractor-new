// Dumps the engine's effective suit and chain position for every card
// slot in every trump context, so the Python rules package can be checked
// against it rather than against a second reading of rules.md.
//
//   npx tsx src/dumpOrdering.ts > ordering.json

import {
  chainPosition,
  effectiveSuit,
  NATURAL_RANKS,
  SUITS,
  type Card,
  type Suit,
  type TrumpContext,
} from '@tractor/engine';

const NUM_TYPES = 54;

function cardOfSlot(slot: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId: 0 };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId: 0 };
  return {
    kind: 'standard',
    suit: SUITS[Math.floor(slot / 13)]!,
    rank: NATURAL_RANKS[slot % 13]!,
    deckId: 0,
  };
}

const SUIT_INDEX = new Map<string, number>(SUITS.map((s, i) => [s, i]));

function main(): void {
  const contexts: Array<{
    levelRank: number;
    trumpSuit: number;
    eff: number[];
    strength: number[];
  }> = [];

  for (let lr = 0; lr < NATURAL_RANKS.length; lr++) {
    // Trump suit index 4 stands for a no-trump (joker) declaration.
    for (let ts = 0; ts < 5; ts++) {
      const trumpSuit: Suit | null = ts === 4 ? null : SUITS[ts]!;
      const ctx: TrumpContext = { trumpRank: NATURAL_RANKS[lr]!, trumpSuit };
      const eff: number[] = [];
      const strength: number[] = [];
      for (let slot = 0; slot < NUM_TYPES; slot++) {
        const card = cardOfSlot(slot);
        const es = effectiveSuit(card, ctx);
        eff.push(es === 'trump' ? 4 : SUIT_INDEX.get(es)!);
        strength.push(chainPosition(card, ctx));
      }
      contexts.push({ levelRank: lr, trumpSuit: ts, eff, strength });
    }
  }

  process.stdout.write(JSON.stringify({ numTypes: NUM_TYPES, contexts }));
}

main();
