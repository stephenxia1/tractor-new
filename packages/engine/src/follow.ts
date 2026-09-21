import type { ClassifiedCombo } from './classifier';
import { chainPosition, findNKinds, findTractors } from './combos';
import { effectiveSuit } from './ordering';
import type { ThrowDecomposition } from './throws';
import { throwTotalCards } from './throws';
import type { Card, TrumpContext } from './types';

export type FollowValidation =
  | { readonly ok: true }
  | { readonly ok: false; readonly error: string };

type LedSuit = ClassifiedCombo['effectiveSuit'];

// What a lead demands of a follower who holds more led-suit cards than
// the lead has slots (rules.md §7.3).
//
//   pairs     total pair slots; a tractor of length L supplies L of them
//   tractors  every tractor component's length, descending
//
// A throw may carry more than one tractor, and §7.3's closing principle
// is that a player may not withhold structure they hold, so the whole
// multiset is required. Both fields are summed over components, which is
// how a throw lead (rules.md §7.6 step 3) and a single-combo lead reach
// the same obligations.
export type Obligations = {
  readonly pairs: number;
  readonly tractors: ReadonlyArray<number>;
};

// Exported so that move enumeration is driven by the same value the
// validator checks. Deriving obligations twice is how the enumerator and
// the reducer drifted apart: the enumerator branched on the lead's combo
// kind and had no branch at all for a throw.
export function leadObligations(
  components: ReadonlyArray<ClassifiedCombo>,
  n: number,
): Obligations {
  let pairs = 0;
  const tractors: number[] = [];
  for (const c of components) {
    if (c.kind === 'nkind') {
      if (c.n >= n) pairs += 1;
    } else if (c.kind === 'tractor') {
      pairs += c.length;
      tractors.push(c.length);
    }
  }
  tractors.sort((a, b) => b - a);
  return { pairs, tractors };
}

// Lengths of the maximal runs of consecutive chain positions held as
// pairs, descending. Tractors within one effective suit are exactly
// these runs, so the runs are all the structure a tractor obligation can
// read. Two off-suit level pairs share a chain position and so occupy a
// single slot, which is why positions are collected into a set.
function pairedRuns(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): number[] {
  const positions = new Set<number>();
  for (const k of findNKinds(cards, n)) {
    positions.add(chainPosition(k.cards[0]!, ctx));
  }
  if (positions.size === 0) return [];
  const sorted = [...positions].sort((a, b) => a - b);
  const runs: number[] = [];
  let run = 1;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] === sorted[i - 1]! + 1) run += 1;
    else {
      runs.push(run);
      run = 1;
    }
  }
  runs.push(run);
  return runs.sort((a, b) => b - a);
}

// What `runs` can supply against tractor slots of the given lengths,
// largest slot first from the longest remaining run — the order
// rules.md §7.3 states. Cutting a length-t piece from one end of a run
// of length r leaves a contiguous run of r - t, so only lengths matter
// and no window search is needed.
// The most a holding's runs can supply against these tractor slots.
//
// Cutting a length-t piece off one end of a run of length r leaves a
// contiguous run of r - t, so only run lengths matter and no search
// over which window to take is needed. Which run feeds which slot does
// matter: runs [4, 3] against slots (3, 2, 2) yields (3, 2, 2) by
// sending the 3-run to the first slot and splitting the 4-run, while
// always drawing from the longest remaining run stops at (3, 2, 0).
//
// The assignment is therefore searched, maximising the profile in slot
// order. The state is the multiset of remaining run lengths, which is
// small: one suit yields a handful of runs and a lead a handful of
// slots.
function profileFromRuns(
  runs: ReadonlyArray<number>,
  lengths: ReadonlyArray<number>,
): number[] {
  const memo = new Map<string, number[]>();

  const better = (a: number[], b: number[]): boolean => {
    for (let i = 0; i < a.length; i++) {
      if (a[i]! !== b[i]!) return a[i]! > b[i]!;
    }
    return false;
  };

  const best = (pool: ReadonlyArray<number>, index: number): number[] => {
    if (index === lengths.length) return [];
    const key = `${pool.join(',')}|${index}`;
    const cached = memo.get(key);
    if (cached !== undefined) return cached;

    const want = lengths[index]!;
    let top: number[] | null = null;
    for (let i = 0; i < pool.length; i++) {
      const run = pool[i]!;
      if (run < 2) continue;
      const take = Math.min(want, run);
      const rest = [...pool];
      rest[i] = run - take;
      const next = rest.filter((r) => r > 0).sort((a, b) => b - a);
      const cand = [take, ...best(next, index + 1)];
      if (top === null || better(cand, top)) top = cand;
    }
    const answer = top ?? [0, ...best(pool, index + 1)];
    memo.set(key, answer);
    return answer;
  };

  const start = [...runs].filter((r) => r > 0).sort((a, b) => b - a);
  return best(start, 0);
}

// The tractor lengths `cards` must supply against this lead, slot by
// slot. Zero entries are slots the holding cannot fill.
export function handTractorProfile(
  obligations: Obligations,
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): number[] {
  if (obligations.tractors.length === 0) return [];
  return profileFromRuns(pairedRuns(cards, ctx, n), obligations.tractors);
}

// Shared core for both lead kinds.
function checkFollow(
  obligations: Obligations,
  leadCount: number,
  ledSuit: LedSuit,
  candidate: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): FollowValidation {
  if (candidate.length !== leadCount) {
    return { ok: false, error: `must play ${leadCount} cards` };
  }

  const handInLed = hand.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const candInLed = candidate.filter((c) => effectiveSuit(c, ctx) === ledSuit);

  // 1. Follow suit: spend as many led-suit cards as you hold, up to the
  // lead's count.
  const required = Math.min(handInLed.length, leadCount);
  if (candInLed.length < required) {
    return {
      ok: false,
      error: `must follow ${ledSuit}: ${required} required, ${candInLed.length} given`,
    };
  }

  // 2. With no more led-suit cards than slots, all of them are already
  // committed by step 1 and no structural choice remains.
  if (handInLed.length <= leadCount) return { ok: true };

  // 3. Tractor obligation, slot by slot: the longest tractor the hand can
  // supply for the longest slot, then the longest still available for the
  // next. Holding a 2-tractor against a 3-tractor lead, the 2-tractor
  // must be played rather than three unrelated pairs; holding two
  // 2-tractors against a throw of two, both must be played.
  if (obligations.tractors.length > 0) {
    const need = handTractorProfile(obligations, handInLed, ctx, n);
    const have = handTractorProfile(obligations, candInLed, ctx, n);
    for (let i = 0; i < need.length; i++) {
      if ((have[i] ?? 0) < (need[i] ?? 0)) {
        const wanted = need.filter((t) => t >= 2).join(' + ');
        return {
          ok: false,
          error: `must include tractors of length ${wanted} in ${ledSuit}`,
        };
      }
    }
  }

  // 4. Pair obligation: commit as many as the lead demands, bounded by
  // what the hand can supply.
  if (obligations.pairs > 0) {
    const handPairs = findNKinds(handInLed, n).length;
    const need = Math.min(handPairs, obligations.pairs);
    const candPairs = findNKinds(candInLed, n).length;
    if (candPairs < need) {
      return {
        ok: false,
        error: `must include ${need} pair(s) of ${ledSuit}: ${candPairs} given`,
      };
    }
  }

  return { ok: true };
}

// Validates a follower's play against a single-combo lead (rules.md §7.3).
export function validateFollow(
  lead: ClassifiedCombo,
  candidate: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): FollowValidation {
  return checkFollow(
    leadObligations([lead], n),
    leadCardCount(lead),
    lead.effectiveSuit,
    candidate,
    hand,
    ctx,
    n,
  );
}

// Validates a follower's play against a throw lead (rules.md §7.6 step 3).
export function validateFollowThrow(
  throwLead: ThrowDecomposition,
  candidate: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): FollowValidation {
  return checkFollow(
    leadObligations(throwLead.components, n),
    throwTotalCards(throwLead),
    throwLead.effectiveSuit,
    candidate,
    hand,
    ctx,
    n,
  );
}

function leadCardCount(c: ClassifiedCombo): number {
  if (c.kind === 'single') return 1;
  return c.cards.length;
}

// Physical identity: one string per card copy in the deck.
function copyId(c: Card): string {
  return c.kind === 'joker'
    ? `joker:${c.rank}#${c.deckId}`
    : `${c.suit}:${c.rank}#${c.deckId}`;
}

// The led-suit cards a follow must contain, built greedily in the order
// rules.md §7.3 states: every tractor the profile demands, then pairs,
// then whatever led-suit cards are left. `inLed` must already be
// filtered to the led suit.
//
// Both move enumerators call this rather than re-deriving the structure,
// which is what keeps them from proposing moves the validator rejects.
export function obligatoryCore(
  obligations: Obligations,
  leadCount: number,
  inLed: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): Card[] {
  const out: Card[] = [];
  const used = new Set<string>();
  const take = (c: Card): void => {
    out.push(c);
    used.add(copyId(c));
  };

  if (inLed.length <= leadCount) {
    for (const c of inLed) take(c);
    return out;
  }

  let havePairs = 0;
  for (const t of handTractorProfile(obligations, inLed, ctx, n)) {
    if (t < 2) continue;
    if (out.length + 2 * t > leadCount) continue;
    const free = inLed.filter((c) => !used.has(copyId(c)));
    const pick = findTractors(free, ctx, n, t)[0];
    if (pick === undefined) continue;
    for (const u of pick.units.slice(0, t)) {
      for (const c of u.cards) take(c);
    }
    havePairs += t;
  }

  const pairs = findNKinds(inLed, n);
  const needPairs = Math.min(pairs.length, obligations.pairs);
  for (const p of pairs) {
    if (havePairs >= needPairs) break;
    if (p.cards.some((c) => used.has(copyId(c)))) continue;
    if (out.length + p.cards.length > leadCount) continue;
    for (const c of p.cards) take(c);
    havePairs += 1;
  }

  for (const c of inLed) {
    if (out.length >= leadCount) break;
    if (used.has(copyId(c))) continue;
    take(c);
  }
  return out;
}

// One follow that satisfies every obligation. A legal follow always
// exists, and callers that must offer at least one move use this as
// their floor.
export function canonicalFollow(
  components: ReadonlyArray<ClassifiedCombo>,
  leadCount: number,
  ledSuit: LedSuit,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): Card[] {
  const obligations = leadObligations(components, n);
  const inLed = hand.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const out = obligatoryCore(obligations, leadCount, inLed, ctx, n);
  const used = new Set(out.map(copyId));
  for (const c of hand) {
    if (out.length >= leadCount) break;
    if (used.has(copyId(c))) continue;
    out.push(c);
    used.add(copyId(c));
  }
  return out;
}
