import {
  classifyCombo,
  comboBeats,
  type ClassifiedCombo,
} from './classifier';
import { findNKinds, findTractors } from './combos';
import { effectiveSuit } from './ordering';
import { removeCards, type Card, type TrumpContext } from './types';

// A "throw" is a multi-component lead — two or more combos played
// simultaneously by the leader, all in the same effective suit. The
// throw succeeds only if no single opponent holds anything that can
// beat any component (rules.md §7.6).

export type ThrowDecomposition = {
  readonly components: ReadonlyArray<ClassifiedCombo>;
  readonly effectiveSuit: ClassifiedCombo['effectiveSuit'];
};

export type DecomposeResult =
  | { readonly ok: true; readonly throw: ThrowDecomposition }
  | { readonly ok: false; readonly reason: string };

// Decomposes cards into the canonical multi-component layout used for a
// throw: longest tractors first, then remaining n-of-a-kinds, then
// singletons. All cards must share an effective suit.
//
// Returns failure if cards span multiple effective suits or the input
// could be a single combo (callers should try `classifyCombo` first;
// throws are strictly compound).
export function decomposeThrow(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): DecomposeResult {
  if (cards.length < 2) return { ok: false, reason: 'too few cards' };

  const es = effectiveSuit(cards[0]!, ctx);
  for (const c of cards) {
    if (effectiveSuit(c, ctx) !== es) {
      return { ok: false, reason: 'cards span multiple effective suits' };
    }
  }

  // If the cards already form a single combo, it's not a throw.
  if (classifyCombo(cards, ctx, n).ok) {
    return { ok: false, reason: 'cards form a single combo, not a throw' };
  }

  const components: ClassifiedCombo[] = [];
  let remaining: Card[] = [...cards];

  // Greedy decomposition: longest tractors → n-of-a-kinds → singletons.
  while (remaining.length > 0) {
    const tractors = findTractors(remaining, ctx, n);
    if (tractors.length > 0) {
      tractors.sort((a, b) => b.length - a.length);
      const t = tractors[0]!;
      const cls = classifyCombo(t.cards, ctx, n);
      if (cls.ok) components.push(cls.combo);
      remaining = removeCards(remaining, t.cards);
      continue;
    }
    const nks = findNKinds(remaining, n);
    if (nks.length > 0) {
      const nk = nks[0]!;
      const cls = classifyCombo(nk.cards, ctx, n);
      if (cls.ok) components.push(cls.combo);
      remaining = removeCards(remaining, nk.cards);
      continue;
    }
    // Singletons — each card is its own component.
    for (const c of remaining) {
      const cls = classifyCombo([c], ctx, n);
      if (cls.ok) components.push(cls.combo);
    }
    remaining = [];
  }

  if (components.length < 2) {
    return { ok: false, reason: 'decomposition collapsed to a single combo' };
  }

  return { ok: true, throw: { components, effectiveSuit: es } };
}

// Safety check for a throw: per rules.md §7.6, every component must be
// unbeatable by any single opponent. Returns `safe: true` if no opponent
// can beat any component. Otherwise returns the smallest beatable
// component (used by the forced-lead penalty, deferred for v1).
export type SafetyResult =
  | { readonly safe: true }
  | {
      readonly safe: false;
      readonly smallestBeatable: ClassifiedCombo;
      readonly beaterIndex: number;
    };

// `opponentHands` is in caller-defined order; the returned `beaterIndex`
// refers to that same order so callers can map it back to a seat.
export function isThrowSafe(
  decomp: ThrowDecomposition,
  opponentHands: ReadonlyArray<ReadonlyArray<Card>>,
  ctx: TrumpContext,
  n: number,
): SafetyResult {
  const throwTotal = throwTotalCards(decomp);
  let smallestBeatable: ClassifiedCombo | null = null;
  let beaterIndex = -1;
  for (const comp of decomp.components) {
    for (let i = 0; i < opponentHands.length; i++) {
      const oppHand = opponentHands[i]!;
      if (componentCanBeBeaten(comp, oppHand, ctx, n, throwTotal)) {
        if (
          smallestBeatable === null ||
          componentCardCount(comp) < componentCardCount(smallestBeatable)
        ) {
          smallestBeatable = comp;
          beaterIndex = i;
        }
        break;
      }
    }
  }
  if (smallestBeatable === null) return { safe: true };
  return { safe: false, smallestBeatable, beaterIndex };
}

function componentCardCount(c: ClassifiedCombo): number {
  if (c.kind === 'single') return 1;
  return c.cards.length;
}

// True iff `hand` contains a same-structure combo IN THE LED EFFECTIVE
// SUIT that beats `target`. Trump combos can never block a non-trump
// throw — followers can't be forced to trump, so the leader is safe
// from any trump-only response.
function componentCanBeBeaten(
  target: ClassifiedCombo,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  _throwTotal: number,
): boolean {
  const targetSuit = target.effectiveSuit;
  const handInLed = hand.filter((c) => effectiveSuit(c, ctx) === targetSuit);
  return findSameStructureBeater(target, handInLed, ctx, n);
}

function findSameStructureBeater(
  target: ClassifiedCombo,
  candidates: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): boolean {
  if (target.kind === 'single') {
    for (const c of candidates) {
      const cls = classifyCombo([c], ctx, n);
      if (!cls.ok) continue;
      if (comboBeats(cls.combo, target) > 0) return true;
    }
    return false;
  }
  if (target.kind === 'nkind') {
    for (const nk of findNKinds(candidates, target.n)) {
      const cls = classifyCombo(nk.cards, ctx, n);
      if (!cls.ok) continue;
      if (comboBeats(cls.combo, target) > 0) return true;
    }
    return false;
  }
  // tractor
  for (const t of findTractors(candidates, ctx, n)) {
    if (t.length !== target.length) continue;
    const cls = classifyCombo(t.cards, ctx, n);
    if (!cls.ok) continue;
    if (comboBeats(cls.combo, target) > 0) return true;
  }
  return false;
}

// Total cards in a throw (sum across components). Used for the kitty
// multiplier when a throw wins the last trick.
export function throwTotalCards(decomp: ThrowDecomposition): number {
  let n = 0;
  for (const c of decomp.components) n += componentCardCount(c);
  return n;
}

// Decompose any play (single combo OR throw) into its component list.
// Returns null if the cards can be neither classified as a single combo
// nor decomposed as a throw (e.g. spans multiple effective suits).
export function decomposePlay(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): ThrowDecomposition | null {
  const cls = classifyCombo(cards, ctx, n);
  if (cls.ok) {
    return {
      components: [cls.combo],
      effectiveSuit: cls.combo.effectiveSuit,
    };
  }
  const d = decomposeThrow(cards, ctx, n);
  if (d.ok) return d.throw;
  return null;
}

function componentShapeKey(c: ClassifiedCombo): string {
  if (c.kind === 'single') return 's';
  if (c.kind === 'nkind') return `n${c.n}`;
  return `t${c.n}x${c.length}`;
}

// True iff `candidate` beats `incumbent` as competing trick plays:
// same component count, same shape multiset, and every candidate
// component out-ranks a distinct incumbent component (1:1 matching).
// For single-component plays this collapses to `comboBeats(...) > 0`.
export function decomposedBeats(
  candidate: ThrowDecomposition,
  incumbent: ThrowDecomposition,
): boolean {
  if (candidate.components.length !== incumbent.components.length) return false;
  const buckets = new Map<string, ClassifiedCombo[]>();
  for (const c of incumbent.components) {
    const k = componentShapeKey(c);
    const arr = buckets.get(k) ?? [];
    arr.push(c);
    buckets.set(k, arr);
  }
  for (const cand of candidate.components) {
    const k = componentShapeKey(cand);
    const bucket = buckets.get(k);
    if (!bucket || bucket.length === 0) return false;
    let matched = -1;
    for (let i = 0; i < bucket.length; i++) {
      if (comboBeats(cand, bucket[i]!) > 0) {
        matched = i;
        break;
      }
    }
    if (matched < 0) return false;
    bucket.splice(matched, 1);
  }
  return true;
}
