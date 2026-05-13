// Exhaustive(-ish) legal-play enumerator for the play phase.
//
// Differs from `legalActions` in that it returns a much wider set of
// candidate moves — closer to "all legal moves" — at the cost of
// enumerating combinatorially-large spaces in some pathological cases.
// Sized for use by the RL agent's candidate-scoring loop, where wider
// is generally better but we still need a manageable per-decision count.
//
// Structural-decomposition optimization (per the rules of tractor):
//   - A tractor is a sequence of consecutive n-of-a-kinds. Build tractors
//     from pairs.
//   - A throw is a multi-component play whose components are themselves
//     valid combos (singles / n-of-a-kinds / tractors). Build throws by
//     picking disjoint subsets of valid components.
//   - For follows, the led-suit obligation forces a "required core" of
//     led-suit cards. Enumerate variations of that core + filler choices,
//     not arbitrary k-subsets.

import { classifyCombo } from '../../classifier';
import { findNKinds, findTractors } from '../../combos';
import { validateFollow } from '../../follow';
import { effectiveSuit, isTrump, compareCards } from '../../ordering';
import { decomposeThrow, isThrowSafe } from '../../throws';
import { handFor, nextToPlay } from '../../reducer';
import { teamOf, type GameState, type Seat } from '../../state';
import {
  cardEquals,
  type Card,
  type EffectiveSuit,
  type TrumpContext,
} from '../../types';

export type EnumeratedAction = {
  readonly kind: 'play_cards';
  readonly seat: Seat;
  readonly cards: ReadonlyArray<Card>;
  readonly tags: ReadonlyArray<
    | 'lead-single'
    | 'lead-pair'
    | 'lead-tractor'
    | 'lead-throw-safe'
    | 'follow-led-suit'
    | 'follow-trump'
    | 'follow-discard'
  >;
};

export type EnumerationOptions = {
  // Cap throw enumeration to throws with at most this many components.
  // Default 4. Tractor components count as one. Bumping past 5 starts to
  // get expensive on cards-rich same-suit hands.
  readonly maxThrowComponents?: number;
  // Soft upper bound on the per-decision candidate count. The enumerator
  // emits in priority order (singles/pairs/tractors first, then throws,
  // then follow variants); when we exceed this, later candidates are
  // dropped. Default 1024.
  readonly maxCandidates?: number;
  // For tractor/throw follows, cap structural enumeration at this many
  // distinct "required core" choices. Default 64.
  readonly maxFollowCores?: number;
};

const DEFAULT_OPTIONS: Required<EnumerationOptions> = {
  // 2 keeps throw enumeration from blowing up on long same-suit holdings
  // while still emitting the most common throw shapes (pair+single,
  // pair+pair, tractor+single).
  maxThrowComponents: 2,
  maxCandidates: 256,
  maxFollowCores: 32,
};

// Top-level entry point. Returns all legal play_cards actions for `seat`
// at the current state. Returns [] for non-play phases or wrong-turn calls.
export function allLegalPlayActions(
  state: GameState,
  seat: Seat,
  options?: EnumerationOptions,
): EnumeratedAction[] {
  const opts = { ...DEFAULT_OPTIONS, ...(options ?? {}) };
  const out: EnumeratedAction[] = [];
  const hand = state.currentHand;
  if (hand === null) return out;
  if (state.phase !== 'play') return out;
  if (nextToPlay(state) !== seat) return out;
  if (hand.trumpRank === null) return out;

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const myCards = hand.hands[seat] ?? [];
  const trick = hand.currentTrick;

  if (trick === null) {
    enumerateLeads(state, seat, myCards, ctx, n, opts, out);
  } else {
    enumerateFollows(seat, myCards, trick, ctx, n, opts, out);
  }

  // Dedup by canonical card-set signature.
  return dedupBySignature(out, opts.maxCandidates);
}

// ----------- Lead enumeration --------------------------------------------

function enumerateLeads(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  opts: Required<EnumerationOptions>,
  out: EnumeratedAction[],
): void {
  // Singles: every card in hand.
  for (const c of myCards) {
    out.push({ kind: 'play_cards', seat, cards: [c], tags: ['lead-single'] });
  }
  // Pairs.
  for (const p of findNKinds(myCards, 2)) {
    out.push({ kind: 'play_cards', seat, cards: p.cards, tags: ['lead-pair'] });
  }
  // Tractors at every length (findTractors returns every sub-run, not
  // just maximal tractors).
  for (const t of findTractors(myCards, ctx, n)) {
    out.push({
      kind: 'play_cards',
      seat,
      cards: t.cards,
      tags: ['lead-tractor'],
    });
  }
  // Throws (multi-component, same effective suit, all components unbeatable).
  enumerateLeadThrows(state, seat, myCards, ctx, n, opts, out);
}

function enumerateLeadThrows(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  opts: Required<EnumerationOptions>,
  out: EnumeratedAction[],
): void {
  // Group cards by effective suit; throws live within one suit.
  const bySuit = new Map<EffectiveSuit, Card[]>();
  for (const c of myCards) {
    const es = effectiveSuit(c, ctx);
    const list = bySuit.get(es) ?? [];
    list.push(c);
    bySuit.set(es, list);
  }

  const opponentHands = collectOpponentHands(state, seat);

  for (const [, suitCards] of bySuit) {
    if (suitCards.length < 2) continue; // throws require ≥ 2 cards

    // Build the candidate-component pool for this suit.
    type Component = { cards: ReadonlyArray<Card>; size: number };
    const components: Component[] = [];
    // Tractors of every length.
    for (const t of findTractors(suitCards, ctx, n)) {
      components.push({ cards: t.cards, size: t.cards.length });
    }
    // Pairs.
    for (const p of findNKinds(suitCards, 2)) {
      components.push({ cards: p.cards, size: 2 });
    }
    // Singles.
    for (const c of suitCards) {
      components.push({ cards: [c], size: 1 });
    }

    // Subset enumeration over components. We want subsets where the
    // chosen components are pairwise card-disjoint, of size 2..maxComponents.
    // This is a small-enough space in practice (per-suit it's typically
    // < 30 components and we cap subset size at 4–5). To avoid duplicates
    // arising from "the same throw built two ways" (e.g., a length-2
    // tractor as one component vs as two pair components), we dedup
    // on the resulting card-set signature at the end.
    const visited = new Set<string>();
    const visit = (
      startIdx: number,
      chosen: Component[],
      usedKeys: Set<string>,
    ): void => {
      if (chosen.length >= 2) {
        const cards = chosen.flatMap((c) => [...c.cards]);
        const sig = signature(cards);
        if (!visited.has(sig)) {
          visited.add(sig);
          // Must NOT classify as a single combo (then it's not a throw).
          if (!classifyCombo(cards, ctx, n).ok) {
            const decomp = decomposeThrow(cards, ctx, n);
            if (decomp.ok) {
              const safety = isThrowSafe(decomp.throw, opponentHands, ctx, n);
              if (safety.safe) {
                out.push({
                  kind: 'play_cards',
                  seat,
                  cards,
                  tags: ['lead-throw-safe'],
                });
              }
            }
          }
        }
      }
      if (chosen.length >= opts.maxThrowComponents) return;

      for (let i = startIdx; i < components.length; i++) {
        const comp = components[i]!;
        // Disjointness check: every card in comp must not already be used.
        let disjoint = true;
        for (const c of comp.cards) {
          if (usedKeys.has(identityKey(c))) {
            disjoint = false;
            break;
          }
        }
        if (!disjoint) continue;
        for (const c of comp.cards) usedKeys.add(identityKey(c));
        chosen.push(comp);
        visit(i + 1, chosen, usedKeys);
        chosen.pop();
        for (const c of comp.cards) usedKeys.delete(identityKey(c));
      }
    };
    visit(0, [], new Set<string>());
  }
}

// ----------- Follow enumeration ------------------------------------------

function enumerateFollows(
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: { leader: Seat; plays: ReadonlyArray<{ cards: ReadonlyArray<Card> }> },
  ctx: TrumpContext,
  n: number,
  opts: Required<EnumerationOptions>,
  out: EnumeratedAction[],
): void {
  const lead = trick.plays[0]!;
  const leadCards = lead.cards;
  const leadCount = leadCards.length;
  const leadCls = classifyCombo(leadCards, ctx, n);
  const ledSuit: EffectiveSuit = leadCls.ok
    ? leadCls.combo.effectiveSuit
    : effectiveSuit(leadCards[0]!, ctx);

  const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);

  // ---- Single lead ------------------------------------------------------
  if (leadCount === 1) {
    if (myInLed.length > 0) {
      // Must play one led-suit card.
      for (const c of myInLed) {
        out.push({
          kind: 'play_cards',
          seat,
          cards: [c],
          tags: tagsFor([c], ledSuit, ctx, leadCls.ok ? leadCls.combo : null, n),
        });
      }
    } else {
      // Free to play any single card (including trumps).
      for (const c of myCards) {
        out.push({
          kind: 'play_cards',
          seat,
          cards: [c],
          tags: tagsFor([c], ledSuit, ctx, leadCls.ok ? leadCls.combo : null, n),
        });
      }
    }
    return;
  }

  // ---- Multi-card lead --------------------------------------------------
  // Strategy: enumerate the "required core" set of led-suit cards
  // (per the led-suit + pair + tractor obligations of rules.md §7.3),
  // then enumerate fillers from the remaining hand.
  enumerateMultiCardFollows(
    seat,
    myCards,
    myInLed,
    leadCount,
    ledSuit,
    leadCls.ok ? leadCls.combo : null,
    ctx,
    n,
    opts,
    out,
  );
}

function enumerateMultiCardFollows(
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  myInLed: ReadonlyArray<Card>,
  leadCount: number,
  ledSuit: EffectiveSuit,
  leadCombo: import('../../classifier').ClassifiedCombo | null,
  ctx: TrumpContext,
  n: number,
  opts: Required<EnumerationOptions>,
  out: EnumeratedAction[],
): void {
  // Compute the "required" minimum led-suit card set, then enumerate
  // candidate cores satisfying the obligations.
  const cores = enumerateRequiredCores(
    myInLed,
    leadCount,
    leadCombo,
    ctx,
    n,
    opts.maxFollowCores,
  );

  for (const core of cores) {
    // Fillers = leadCount − core.length cards from non-core hand.
    const fillerCount = leadCount - core.length;
    if (fillerCount === 0) {
      emitIfValid(seat, [...core], myCards, ledSuit, leadCombo, ctx, n, out);
      continue;
    }
    const remaining = myCards.filter((c) => !cardSetContains(core, c));
    enumerateFillers(
      seat,
      core,
      remaining,
      fillerCount,
      myCards,
      ledSuit,
      leadCombo,
      ctx,
      n,
      opts,
      out,
    );
  }
}

// Enumerate candidate "required cores" — the led-suit cards that must
// appear in the follow given the led-suit + pair + tractor obligations.
function enumerateRequiredCores(
  myInLed: ReadonlyArray<Card>,
  leadCount: number,
  leadCombo: import('../../classifier').ClassifiedCombo | null,
  ctx: TrumpContext,
  n: number,
  maxCores: number,
): Card[][] {
  const cores: Card[][] = [];
  const inLedCount = myInLed.length;

  // Quick lookup of all led-suit pairs and tractors.
  const ledPairs = findNKinds(myInLed, n);
  const ledTractors = findTractors(myInLed, ctx, n);

  // Tractor lead.
  if (leadCombo !== null && leadCombo.kind === 'tractor') {
    const k = leadCombo.length;
    const matching = ledTractors.filter((t) => t.length === k);
    if (matching.length > 0) {
      // Must include a length-k tractor in led suit.
      // Each candidate core: one tractor, possibly extended with led-suit
      // fillers up to leadCount cards.
      for (const t of matching) {
        const baseCore = [...t.cards];
        if (baseCore.length >= leadCount) {
          cores.push(baseCore.slice(0, leadCount));
        } else {
          // Fill remaining slots with led-suit cards (any choice; we'll
          // enumerate fill variants later via the filler step). For the
          // core enumeration here, just emit the bare tractor — the
          // filler step will add led-suit fillers preferentially.
          cores.push(baseCore);
        }
        if (cores.length >= maxCores) return cores;
      }
      return cores;
    }
    // No matching length-k tractor: must include all available pairs in
    // led suit (up to leadCombo.length pairs), plus singles to fill.
    // Enumerate which pair subsets to include.
    const requiredPairs = Math.min(ledPairs.length, k);
    const pairsToInclude = chooseSubsets(ledPairs, requiredPairs, maxCores);
    for (const pairChoice of pairsToInclude) {
      const baseCore = pairChoice.flatMap((p) => [...p.cards]);
      // Include the pair cards as the core's required portion. The rest
      // of the led-suit cards (singles) and any fillers come later.
      cores.push(baseCore);
      if (cores.length >= maxCores) return cores;
    }
    if (cores.length === 0) {
      // No pairs in led suit either — just need to play min(inLedCount,
      // leadCount) led-suit cards. Enumerate which subset.
      enumerateLedSuitSubsets(myInLed, leadCount, cores, maxCores);
    }
    return cores;
  }

  // Pair lead.
  if (leadCombo !== null && leadCombo.kind === 'nkind' && leadCombo.n === 2) {
    if (ledPairs.length > 0) {
      // Must include one pair.
      for (const p of ledPairs) {
        cores.push([...p.cards]);
        if (cores.length >= maxCores) return cores;
      }
      return cores;
    }
    // No pairs — must include min(inLedCount, 2) led-suit cards.
    enumerateLedSuitSubsets(myInLed, Math.min(inLedCount, leadCount), cores, maxCores);
    return cores;
  }

  // Throw lead or unclassified lead — fall back to "play all led-suit
  // cards as required core" (the simplest safe interpretation).
  enumerateLedSuitSubsets(myInLed, Math.min(inLedCount, leadCount), cores, maxCores);
  return cores;
}

// Enumerate subsets of `cards` of exactly size `k`. Used when the rules
// don't pin down which led-suit cards must be in the core.
function enumerateLedSuitSubsets(
  cards: ReadonlyArray<Card>,
  k: number,
  out: Card[][],
  maxCores: number,
): void {
  if (k <= 0) {
    out.push([]);
    return;
  }
  if (k > cards.length) return;
  if (k === cards.length) {
    out.push([...cards]);
    return;
  }
  const cur: Card[] = [];
  const recurse = (start: number): void => {
    if (out.length >= maxCores) return;
    if (cur.length === k) {
      out.push([...cur]);
      return;
    }
    const need = k - cur.length;
    for (let i = start; i + need - 1 < cards.length; i++) {
      cur.push(cards[i]!);
      recurse(i + 1);
      cur.pop();
      if (out.length >= maxCores) return;
    }
  };
  recurse(0);
}

// Choose `k` items from `pool`. If pool.length ≤ k, return [pool] (must
// include them all). If `k === 0`, return [[]]. Otherwise enumerate
// k-subsets, capped at maxOut.
function chooseSubsets<T>(
  pool: ReadonlyArray<T>,
  k: number,
  maxOut: number,
): T[][] {
  if (k <= 0) return [[]];
  if (pool.length <= k) return [pool.slice()];
  const out: T[][] = [];
  const cur: T[] = [];
  const recurse = (start: number): void => {
    if (out.length >= maxOut) return;
    if (cur.length === k) {
      out.push([...cur]);
      return;
    }
    const need = k - cur.length;
    for (let i = start; i + need - 1 < pool.length; i++) {
      cur.push(pool[i]!);
      recurse(i + 1);
      cur.pop();
      if (out.length >= maxOut) return;
    }
  };
  recurse(0);
  return out;
}

// Enumerate filler subsets of size `fillerCount` from `pool`. Strategy:
//   - Prefer led-suit fillers first (they're often required to satisfy
//     led-suit obligation up to inLedCount).
//   - For the rest, allow any combination; cap by maxCandidates so we
//     don't blow up on big hands.
function enumerateFillers(
  seat: Seat,
  core: ReadonlyArray<Card>,
  pool: ReadonlyArray<Card>,
  fillerCount: number,
  myCards: ReadonlyArray<Card>,
  ledSuit: EffectiveSuit,
  leadCombo: import('../../classifier').ClassifiedCombo | null,
  ctx: TrumpContext,
  n: number,
  opts: Required<EnumerationOptions>,
  out: EnumeratedAction[],
): void {
  // Cap the number of fillers we're willing to enumerate for a single core.
  const localCap = Math.max(8, Math.floor(opts.maxCandidates / 8));

  // Stratify: any led-suit cards in the pool MUST be played first (up
  // to fillerCount), per follow-suit rule. Compute how many led-suit
  // fillers are mandatory.
  const ledInPool = pool.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const nonLedInPool = pool.filter((c) => effectiveSuit(c, ctx) !== ledSuit);
  const mandatoryLed = Math.min(ledInPool.length, fillerCount);

  if (mandatoryLed > 0) {
    // Choose which `mandatoryLed` led-suit cards to use, then choose
    // `fillerCount - mandatoryLed` non-led fillers.
    const ledSubsets = chooseSubsets(ledInPool, mandatoryLed, localCap);
    const nonLedNeeded = fillerCount - mandatoryLed;
    for (const ls of ledSubsets) {
      if (nonLedNeeded === 0) {
        const cards = [...core, ...ls];
        emitIfValid(seat, cards, myCards, ledSuit, leadCombo, ctx, n, out);
        if (out.length >= opts.maxCandidates) return;
        continue;
      }
      const nlSubsets = chooseSubsets(nonLedInPool, nonLedNeeded, localCap);
      for (const nls of nlSubsets) {
        const cards = [...core, ...ls, ...nls];
        emitIfValid(seat, cards, myCards, ledSuit, leadCombo, ctx, n, out);
        if (out.length >= opts.maxCandidates) return;
      }
    }
  } else {
    // No led-suit fillers required — fill from anywhere. Equivalent to
    // choosing `fillerCount` from the full pool.
    const subsets = chooseSubsets(pool, fillerCount, localCap);
    for (const s of subsets) {
      const cards = [...core, ...s];
      emitIfValid(seat, cards, myCards, ledSuit, leadCombo, ctx, n, out);
      if (out.length >= opts.maxCandidates) return;
    }
  }
}

function emitIfValid(
  seat: Seat,
  cards: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ledSuit: EffectiveSuit,
  leadCombo: import('../../classifier').ClassifiedCombo | null,
  ctx: TrumpContext,
  n: number,
  out: EnumeratedAction[],
): void {
  if (leadCombo !== null) {
    const v = validateFollow(leadCombo, cards, hand, ctx, n);
    if (!v.ok) return;
  }
  out.push({
    kind: 'play_cards',
    seat,
    cards,
    tags: tagsFor(cards, ledSuit, ctx, leadCombo, n),
  });
}

// ----------- Helpers -----------------------------------------------------

function tagsFor(
  cards: ReadonlyArray<Card>,
  ledSuit: EffectiveSuit,
  ctx: TrumpContext,
  leadCombo: import('../../classifier').ClassifiedCombo | null,
  n: number,
): EnumeratedAction['tags'] {
  if (leadCombo === null) {
    // Free play (e.g., throw lead with no validate context). Use led-suit
    // as the discriminator.
    if (cards.every((c) => effectiveSuit(c, ctx) === ledSuit)) {
      return ['follow-led-suit'];
    }
    return ['follow-discard'];
  }
  const inLedCount = cards.filter((c) => effectiveSuit(c, ctx) === ledSuit).length;
  if (inLedCount === cards.length) return ['follow-led-suit'];
  if (cards.every((c) => isTrump(c, ctx)) && ledSuit !== 'trump') {
    // Could be a trump-beat. Check structure-match.
    const cls = classifyCombo(cards, ctx, n);
    if (cls.ok && cls.combo.effectiveSuit === 'trump') {
      return ['follow-trump'];
    }
  }
  return ['follow-discard'];
}

function collectOpponentHands(
  state: GameState,
  leaderSeat: Seat,
): ReadonlyArray<ReadonlyArray<Card>> {
  if (state.currentHand === null) return [];
  const leaderTeam = teamOf(state, leaderSeat);
  const out: ReadonlyArray<Card>[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (teamOf(state, s) === leaderTeam) continue;
    out.push(handFor(state, s));
  }
  return out;
}

function identityKey(c: Card): string {
  if (c.kind === 'joker') return `J${c.rank}-${c.deckId}`;
  return `S${c.suit}${c.rank}-${c.deckId}`;
}

function signature(cards: ReadonlyArray<Card>): string {
  return cards.map(identityKey).sort().join('|');
}

function cardSetContains(set: ReadonlyArray<Card>, c: Card): boolean {
  for (const x of set) if (cardEquals(x, c)) return true;
  return false;
}

function dedupBySignature(
  actions: EnumeratedAction[],
  cap: number,
): EnumeratedAction[] {
  const seen = new Set<string>();
  const out: EnumeratedAction[] = [];
  for (const a of actions) {
    const sig = signature(a.cards);
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push(a);
    if (out.length >= cap) break;
  }
  return out;
}

// Touch compareCards to keep the import live for future ordering work.
const _ = compareCards;
void _;
