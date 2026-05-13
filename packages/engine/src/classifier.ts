import { chainPosition } from './combos';
import { effectiveSuit } from './ordering';
import {
  cardKey,
  type Card,
  type EffectiveSuit,
  type TrumpContext,
} from './types';

// A classified combo — the structural meaning of a set of cards.
// Used for both lead validation and trick resolution.
//
// `topPosition` is the chain position of the highest card/group in the
// combo (used to compare combos of the same structure for trick winning).
// For singles and n-of-a-kinds this is just the position of the (sole)
// group. For tractors it's the position of the highest group.
export type ClassifiedCombo =
  | {
      readonly kind: 'single';
      readonly card: Card;
      readonly effectiveSuit: EffectiveSuit;
      readonly topPosition: number;
    }
  | {
      readonly kind: 'nkind';
      readonly n: number;
      readonly cards: ReadonlyArray<Card>;
      readonly effectiveSuit: EffectiveSuit;
      readonly topPosition: number;
    }
  | {
      readonly kind: 'tractor';
      readonly n: number;
      readonly length: number;
      readonly cards: ReadonlyArray<Card>;
      readonly effectiveSuit: EffectiveSuit;
      readonly topPosition: number;
      readonly bottomPosition: number;
    };

export type ClassifyResult =
  | { readonly ok: true; readonly combo: ClassifiedCombo }
  | { readonly ok: false; readonly reason: string };

// Classify cards as a "simple" combo: single, n-of-a-kind, or tractor.
// Returns failure for: empty input, mixed effective suits, group-size
// mismatches, or non-consecutive groups (would-be tractors with gaps).
// Throws (multi-component leads) are handled separately by `classifyThrow`.
export function classifyCombo(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): ClassifyResult {
  if (cards.length === 0) return { ok: false, reason: 'empty' };

  const es = effectiveSuit(cards[0]!, ctx);
  for (const c of cards) {
    if (effectiveSuit(c, ctx) !== es) {
      return { ok: false, reason: 'cards span multiple effective suits' };
    }
  }

  if (cards.length === 1) {
    return {
      ok: true,
      combo: {
        kind: 'single',
        card: cards[0]!,
        effectiveSuit: es,
        topPosition: chainPosition(cards[0]!, ctx),
      },
    };
  }

  // Group by cardKey.
  const groups = new Map<string, Card[]>();
  for (const c of cards) {
    const k = cardKey(c);
    const g = groups.get(k) ?? [];
    g.push(c);
    groups.set(k, g);
  }

  // Every group must be exactly n cards.
  for (const g of groups.values()) {
    if (g.length !== n) {
      return {
        ok: false,
        reason: `group size ${g.length} does not equal n=${n}`,
      };
    }
  }

  if (groups.size === 1) {
    return {
      ok: true,
      combo: {
        kind: 'nkind',
        n,
        cards,
        effectiveSuit: es,
        topPosition: chainPosition(cards[0]!, ctx),
      },
    };
  }

  // Tractor candidate: groups must be at consecutive chain positions.
  const groupInfo = [...groups.values()].map((g) => ({
    pos: chainPosition(g[0]!, ctx),
  }));
  groupInfo.sort((a, b) => a.pos - b.pos);
  for (let i = 1; i < groupInfo.length; i++) {
    if (groupInfo[i]!.pos - groupInfo[i - 1]!.pos !== 1) {
      return { ok: false, reason: 'groups not at consecutive chain positions' };
    }
  }
  return {
    ok: true,
    combo: {
      kind: 'tractor',
      n,
      length: groupInfo.length,
      cards,
      effectiveSuit: es,
      topPosition: groupInfo[groupInfo.length - 1]!.pos,
      bottomPosition: groupInfo[0]!.pos,
    },
  };
}

// Comparison for trick resolution. Returns +1 if `a` beats `b`, -1 if
// `b` beats `a`, 0 if they don't compete (different structure, or
// different non-trump suits).
//
// Trump beats non-trump. Within the same effective suit, higher
// `topPosition` wins. Equal `topPosition` returns 0 (caller breaks ties
// by play order).
export function comboBeats(
  a: ClassifiedCombo,
  b: ClassifiedCombo,
): -1 | 0 | 1 {
  // Must share structure to compete.
  if (a.kind !== b.kind) return 0;
  if (a.kind === 'nkind' && b.kind === 'nkind' && a.n !== b.n) return 0;
  if (a.kind === 'tractor' && b.kind === 'tractor') {
    if (a.n !== b.n || a.length !== b.length) return 0;
  }

  const aTrump = a.effectiveSuit === 'trump';
  const bTrump = b.effectiveSuit === 'trump';
  if (aTrump !== bTrump) return aTrump ? 1 : -1;

  // Same trump-status. For non-trumps, different suits → no competition.
  if (!aTrump && a.effectiveSuit !== b.effectiveSuit) return 0;

  if (a.topPosition > b.topPosition) return 1;
  if (a.topPosition < b.topPosition) return -1;
  return 0;
}
