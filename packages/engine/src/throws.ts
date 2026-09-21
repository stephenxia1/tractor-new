import {
  classifyCombo,
  comboBeats,
  type ClassifiedCombo,
} from './classifier';
import { findNKinds, findTractors } from './combos';
import { effectiveSuit } from './ordering';
import {
  cardKey,
  removeCards,
  NATURAL_RANKS,
  SUITS,
  type Card,
  type TrumpContext,
} from './types';

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
        // rules.md §7.6 step 4 keeps "the smallest component the
        // opponent can beat (lowest single, or lowest pair, etc.)":
        // size first, then rank. Card identity breaks what remains.
        // That last key is arbitrary — two components can tie on both
        // size and rank, as two off-suit level cards do under a
        // no-trump declaration, where they share a rung and are
        // interchangeable. Arbitrary is fine; order-dependent is not,
        // and with no final key the winner was whichever component the
        // leader happened to list first.
        const smaller =
          smallestBeatable === null ||
          componentCardCount(comp) < componentCardCount(smallestBeatable) ||
          (componentCardCount(comp) === componentCardCount(smallestBeatable) &&
            (comp.topPosition < smallestBeatable.topPosition ||
              (comp.topPosition === smallestBeatable.topPosition &&
                lowestSlot(comp) < lowestSlot(smallestBeatable))));
        if (smaller) {
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

// Deck position of a card, and the lowest one a component holds. Used
// only as a deterministic last tie-break between components that are
// equal in size and rank.
function slotOf(c: Card): number {
  if (c.kind === 'joker') return c.rank === 'big' ? 53 : 52;
  return SUITS.indexOf(c.suit) * 13 + NATURAL_RANKS.indexOf(c.rank);
}

function lowestSlot(c: ClassifiedCombo): number {
  const cards = c.kind === 'single' ? [c.card] : c.cards;
  let best = Number.MAX_SAFE_INTEGER;
  for (const card of cards) {
    const slot = slotOf(card);
    if (slot < best) best = slot;
  }
  return best;
}

function componentCardCount(c: ClassifiedCombo): number {
  if (c.kind === 'single') return 1;
  return c.cards.length;
}

// True iff `hand` contains a same-structure combo IN THE LED EFFECTIVE
// SUIT that beats `target`. Per rules.md §7.6, the safety check (block)
// is restricted to the led suit — a follower can only be *forced* to
// follow led-suit cards, so a trump-only response can never be
// guaranteed at lead time. (Trumps can still BEAT a throw at trick
// resolution if the follower chooses to spend them; that is handled in
// `resolveTrick` via structural comparison.)
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

// True iff `candidate` beats `incumbent` as competing trick plays.
//
// Two paths to a win:
//   (a) **Same-shape match.** Candidate's natural components have the
//       same shape multiset as incumbent's; we 1:1-match shapes and
//       require every candidate component to out-rank its match.
//   (b) **Stronger structure covers weaker.** Candidate has a "tighter"
//       structure than incumbent (e.g., a pair vs. two singles, a
//       tractor vs. several pairs). We split candidate down to
//       incumbent's component shapes and check that the resulting
//       singletons / pieces beat each incumbent component, matched
//       1:1 via bipartite matching.
//
// Both paths use trump-vs-non-trump comparison from `comboBeats`. When
// `ctx` is not provided, only path (a) is attempted.
export function decomposedBeats(
  candidate: ThrowDecomposition,
  incumbent: ThrowDecomposition,
  ctx?: TrumpContext,
  n?: number,
): boolean {
  // Path (a): same shape multiset.
  if (
    candidate.components.length === incumbent.components.length &&
    sameShapeMatchingBeats(candidate, incumbent)
  ) {
    return true;
  }
  if (ctx === undefined || n === undefined) return false;

  // Path (b): re-decompose the candidate into the incumbent's shape. An
  // all-singleton incumbent keeps the exact bipartite matcher below,
  // which is faster and not subject to the search budget.
  if (incumbent.components.every((c) => c.kind === 'single')) {
    return candidateBeatsAllSingles(candidate, incumbent, ctx, n);
  }

  const cards: Card[] = [];
  for (const comp of candidate.components) {
    if (comp.kind === 'single') cards.push(comp.card);
    else cards.push(...comp.cards);
  }
  let need = 0;
  for (const c of incumbent.components) need += shapeCardCount(c);
  if (cards.length !== need) return false;

  // Tightest components first, so the most constrained placements happen
  // before the loose ones and dead branches die early.
  const ordered = [...incumbent.components].sort(
    (a, b) => tightnessRank(b) - tightnessRank(a),
  );
  return coverPartition(cards, ordered, ctx, n, { v: COVER_NODE_BUDGET });
}

// Guard against a pathological number of non-single incumbent
// components. Singles no longer contribute to the search at all, so
// reaching this needs a throw with more pairs and tractors than any
// real hand can hold.
const COVER_NODE_BUDGET = 200000;

// Every remaining card must cover one remaining single-component slot.
function singlesCover(
  cards: ReadonlyArray<Card>,
  comps: ReadonlyArray<ClassifiedCombo>,
  ctx: TrumpContext,
  n: number,
): boolean {
  if (cards.length !== comps.length) return false;
  const singles: ClassifiedCombo[] = [];
  for (const c of cards) {
    const cls = classifyCombo([c], ctx, n);
    if (!cls.ok) return false;
    singles.push(cls.combo);
  }
  return bipartiteAllBeat(singles, comps);
}

function shapeCardCount(c: ClassifiedCombo): number {
  if (c.kind === 'single') return 1;
  return c.cards.length;
}

function tightnessRank(c: ClassifiedCombo): number {
  if (c.kind === 'tractor') return 1000 + c.length;
  if (c.kind === 'nkind') return 100 + c.n;
  return 1;
}

// Every length-`L` tractor window present in `cards`. A tractor of length
// M contains M - L + 1 of them, and `units` holds its n-of-a-kinds in
// chain order, so the windows are slices of that.
function tractorWindows(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  L: number,
): Card[][] {
  const out: Card[][] = [];
  const seen = new Set<string>();
  for (const t of findTractors(cards, ctx, n, L)) {
    for (let i = 0; i + L <= t.length; i++) {
      const cs = t.units.slice(i, i + L).flatMap((u) => [...u.cards]);
      const key = cs
        .map(cardKey)
        .sort()
        .join('|');
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(cs);
    }
  }
  return out;
}

// Card groups in `cards` that match `inc`'s shape. Rank is checked by the
// caller; this is shape only.
function piecesOfShape(
  cards: ReadonlyArray<Card>,
  inc: ClassifiedCombo,
  ctx: TrumpContext,
  n: number,
): Card[][] {
  if (inc.kind === 'single') {
    const out: Card[][] = [];
    const seen = new Set<string>();
    for (const c of cards) {
      const k = cardKey(c);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push([c]);
    }
    return out;
  }
  if (inc.kind === 'nkind') {
    return findNKinds(cards, inc.n).map((k) => [...k.cards]);
  }
  return tractorWindows(cards, ctx, n, inc.length);
}

// Can `cards` be partitioned so every incumbent component is covered by a
// piece of its own shape that outranks it? rules.md §7.6 step 3: the
// incumbent's structure is fixed and only the candidate is re-decomposed,
// so a tractor may stand in for pairs but two singles never make a pair.
function coverPartition(
  cards: ReadonlyArray<Card>,
  comps: ReadonlyArray<ClassifiedCombo>,
  ctx: TrumpContext,
  n: number,
  budget: { v: number },
): boolean {
  if (comps.length === 0) return cards.length === 0;
  // Components arrive tightest-first, so once the head is a single the
  // whole tail is singles. Assigning cards to those slots is an exact
  // bipartite matching, not a search: backtracking over them was what
  // made a 14-card trick cost 65,084 nodes, overrun the budget, and
  // return an answer that depended on the order the cards happened to
  // sit in.
  if (comps[0]!.kind === 'single') return singlesCover(cards, comps, ctx, n);
  if (budget.v <= 0) return false;
  budget.v -= 1;
  const inc = comps[0]!;
  const rest = comps.slice(1);
  for (const piece of piecesOfShape(cards, inc, ctx, n)) {
    const cls = classifyCombo(piece, ctx, n);
    if (!cls.ok) continue;
    if (comboBeats(cls.combo, inc) <= 0) continue;
    if (coverPartition(removeCards(cards, piece), rest, ctx, n, budget)) {
      return true;
    }
  }
  return false;
}

// `comboBeats` already returns 0 across mismatched shapes, so one
// matching over all components is both the shape check and the rank
// check. Greedy first-fit was wrong here: against an incumbent of
// `3D 9D` a candidate of `KD 5D` covers (K beats 9, 5 beats 3), but a
// greedy pass spends K on 3 and then strands 5.
function sameShapeMatchingBeats(
  candidate: ThrowDecomposition,
  incumbent: ThrowDecomposition,
): boolean {
  return bipartiteAllBeat(candidate.components, incumbent.components);
}

// Re-decompose the candidate's cards into individual single combos and
// bipartite-match against the incumbent's single components.
function candidateBeatsAllSingles(
  candidate: ThrowDecomposition,
  incumbent: ThrowDecomposition,
  ctx: TrumpContext,
  n: number,
): boolean {
  const candCards: Card[] = [];
  for (const comp of candidate.components) {
    if (comp.kind === 'single') candCards.push(comp.card);
    else candCards.push(...comp.cards);
  }
  if (candCards.length !== incumbent.components.length) return false;
  const candSingles: ClassifiedCombo[] = [];
  for (const c of candCards) {
    const cls = classifyCombo([c], ctx, n);
    if (!cls.ok) return false;
    candSingles.push(cls.combo);
  }
  return bipartiteAllBeat(candSingles, incumbent.components);
}

// Bipartite perfect matching: every incumbent gets a distinct candidate
// that beats it (per `comboBeats > 0`). Uses augmenting paths.
function bipartiteAllBeat(
  candidates: ReadonlyArray<ClassifiedCombo>,
  incumbents: ReadonlyArray<ClassifiedCombo>,
): boolean {
  if (candidates.length !== incumbents.length) return false;
  const m = incumbents.length;
  const canBeat: boolean[][] = candidates.map((cand) =>
    incumbents.map((inc) => comboBeats(cand, inc) > 0),
  );
  const matchOfInc = new Array<number>(m).fill(-1);
  function augment(candIdx: number, visited: boolean[]): boolean {
    for (let j = 0; j < m; j++) {
      if (!canBeat[candIdx]![j]) continue;
      if (visited[j]) continue;
      visited[j] = true;
      if (matchOfInc[j] === -1 || augment(matchOfInc[j]!, visited)) {
        matchOfInc[j] = candIdx;
        return true;
      }
    }
    return false;
  }
  let matched = 0;
  for (let i = 0; i < candidates.length; i++) {
    const visited = new Array<boolean>(m).fill(false);
    if (augment(i, visited)) matched++;
  }
  return matched === m;
}
