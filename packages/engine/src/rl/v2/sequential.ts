// Sequential (card-by-card) action space for play decisions.
//
// A play is built one card at a time. At each sub-step the seat holds a
// partial play (a prefix) and either adds one more card or, for a lead
// that is already a legal play, submits it. Follows have a size fixed by
// the lead and complete themselves when they reach it.
//
// Three properties make this a faithful replacement for an enumerated
// candidate list, and the tests check each one against a brute-force
// oracle (`oracle.ts`):
//
//   * Complete: every fair-play legal move is reachable.
//   * Sound:    nothing else is, and no prefix is a dead end.
//   * Unique:   each move is reached by exactly one sequence.
//
// Uniqueness comes from a canonical order: cards are added in
// non-increasing `orderKey`, and the two copies of a card are
// interchangeable (the action is a card *key*, a slot). A k-card play then
// has one sequence rather than k!, so its value is not split across
// orderings of the same set.
//
// Masks are exact, not heuristic. A card is offered only if some legal
// move still extends the prefix with it, given that every later card must
// rank at or below it in the canonical order.
//
//   * Follows use counting. The rules constrain only the led-suit part of
//     a follow (how many led cards, how many pairs, a tractor when the
//     hand has one), so feasibility reduces to a few counts per prefix.
//   * Leads are enumerated once per decision -- singles, pairs, tractors
//     and every throw a fair-play agent may risk -- into a prefix tree,
//     and each mask is a lookup.
//
// Fair play: a throw is offered only if no component can be beaten even
// by a single opponent holding every card this seat has not seen. That
// opponent is the worst case, because beating ability only grows with
// holdings, so an offered throw can never draw the forced-lead penalty.

import { classifyCombo, type ClassifiedCombo } from '../../classifier';
import { chainPosition, findTractors } from '../../combos';
import { effectiveSuit, isTrump } from '../../ordering';
import { type GameState, type Seat } from '../../state';
import { decomposePlay, decomposeThrow, isThrowSafe } from '../../throws';
import { SUITS, type Card, type EffectiveSuit, type TrumpContext } from '../../types';

import { unseenCounts } from './candidates';
import { cardForSlot, cardSlot, NUM_SLOTS } from './cards';
import { MAX_CANDIDATE_CARDS } from './record';

// ---------------------------------------------------------------------------
// Canonical order
// ---------------------------------------------------------------------------

// Larger keys are added first. Trumps come before the plain suits, a suit
// is ordered strongest first by chain position, and the slot breaks ties
// between distinct cards of equal strength (the off-suit trump-rank tier).
// Any fixed total order gives uniqueness; strongest-first is the one that
// reads naturally, since a play's highest card usually decides it.
export function orderKey(slot: number, ctx: TrumpContext): number {
  const card = cardForSlot(slot);
  const group = isTrump(card, ctx)
    ? 4
    : 3 - SUITS.indexOf((card as { suit: (typeof SUITS)[number] }).suit);
  return group * 10_000 + chainPosition(card, ctx) * 100 + slot;
}

// ---------------------------------------------------------------------------
// Fair-play lead legality (shared with the oracle)
// ---------------------------------------------------------------------------

// Every card this seat has not seen, as one hand.
export function pooledUnseenHand(
  state: GameState,
  seat: Seat,
  myKitty?: ReadonlyArray<Card>,
): Card[] {
  const counts = unseenCounts(state, seat, myKitty);
  const out: Card[] = [];
  for (let slot = 0; slot < NUM_SLOTS; slot++) {
    for (let copy = 0; copy < counts[slot]!; copy++) {
      out.push({ ...cardForSlot(slot), deckId: copy } as Card);
    }
  }
  return out;
}

function componentsSafe(
  components: ReadonlyArray<ClassifiedCombo>,
  suit: EffectiveSuit,
  pooled: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): boolean {
  return isThrowSafe({ components, effectiveSuit: suit }, [pooled], ctx, n).safe;
}

// True iff `cards` is a lead a fair-play agent may submit: a single combo,
// or a throw no component of which the pooled opponent can beat.
export function isFairLead(
  cards: ReadonlyArray<Card>,
  pooled: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): boolean {
  if (cards.length === 0) return false;
  if (classifyCombo(cards, ctx, n).ok) return true;
  const d = decomposeThrow(cards, ctx, n);
  if (!d.ok) return false;
  return isThrowSafe(d.throw, [pooled], ctx, n).safe;
}

// ---------------------------------------------------------------------------
// The per-decision action space
// ---------------------------------------------------------------------------

export type SequentialOptions = {
  // Slots that may be added next, in canonical (descending) order.
  readonly next: ReadonlyArray<number>;
  // The prefix is itself a complete legal play. For a follow this is true
  // exactly when the prefix has reached the lead's size.
  readonly canSubmit: boolean;
};

export type SequentialBuildOptions = {
  readonly myKitty?: ReadonlyArray<Card>;
  readonly maxCards?: number;
};

type FollowPlan = {
  readonly k: number;
  readonly req: number; // led-suit cards the follow must contain
  readonly fill: number; // cards from outside the led suit
  readonly reqNK: number; // n-of-a-kinds the led part must contain
  // When the lead is a tractor the hand can match, the legal led parts are
  // exactly these multisets (slot -> count); otherwise null.
  readonly tractorParts: ReadonlyArray<ReadonlyMap<number, number>> | null;
};

export class SequentialPlay {
  readonly mode: 'lead' | 'follow' | 'none';
  // The follow size; null for leads.
  readonly size: number | null;

  private readonly ctx: TrumpContext;
  private readonly n: number;
  private readonly maxCards: number;
  private readonly physical = new Map<number, Card[]>();
  private readonly held = new Map<number, number>();
  private readonly key = new Map<number, number>();
  private readonly led = new Set<number>();
  // Held slots, highest order key first.
  private readonly order: number[];
  private readonly follow: FollowPlan | null = null;
  // Lead prefix tree: prefix signature -> next slots + whether terminal.
  private readonly trie: Map<string, { next: Set<number>; terminal: boolean }> | null = null;

  constructor(state: GameState, seat: Seat, opts: SequentialBuildOptions = {}) {
    const hand = state.currentHand;
    this.maxCards = opts.maxCards ?? MAX_CANDIDATE_CARDS;
    if (hand === null || hand.trumpRank === null || state.phase !== 'play') {
      this.mode = 'none';
      this.size = null;
      this.ctx = { trumpRank: '2', trumpSuit: null };
      this.n = state.config.numDecks;
      this.order = [];
      return;
    }
    this.ctx = { trumpRank: hand.trumpRank, trumpSuit: hand.trumpSuit };
    this.n = state.config.numDecks;
    const myCards = hand.hands[seat] ?? [];
    for (const c of myCards) {
      const s = cardSlot(c);
      const list = this.physical.get(s);
      if (list === undefined) this.physical.set(s, [c]);
      else list.push(c);
    }
    for (const [s, cards] of this.physical) {
      this.held.set(s, cards.length);
      this.key.set(s, orderKey(s, this.ctx));
    }
    this.order = [...this.held.keys()].sort((a, b) => this.key.get(b)! - this.key.get(a)!);

    const trick = hand.currentTrick;
    if (trick !== null && trick.plays.length > 0) {
      this.mode = 'follow';
      const plan = this.planFollow(trick.plays[0]!.cards, myCards);
      this.follow = plan;
      this.size = plan === null ? null : plan.k;
    } else {
      this.mode = 'lead';
      this.size = null;
      const pooled = pooledUnseenHand(state, seat, opts.myKitty);
      this.trie = this.buildLeadTrie(myCards, pooled);
    }
  }

  // Physical cards for a prefix of slots.
  cards(prefix: ReadonlyArray<number>): Card[] {
    const used = new Map<number, number>();
    const out: Card[] = [];
    for (const s of prefix) {
      const i = used.get(s) ?? 0;
      const c = this.physical.get(s)?.[i];
      if (c === undefined) throw new Error(`slot ${s} used more often than held`);
      out.push(c);
      used.set(s, i + 1);
    }
    return out;
  }

  options(prefix: ReadonlyArray<number>): SequentialOptions {
    if (this.mode === 'lead') {
      const node = this.trie!.get(prefix.join(','));
      if (node === undefined) return { next: [], canSubmit: false };
      const next = [...node.next].sort((a, b) => this.key.get(b)! - this.key.get(a)!);
      return { next, canSubmit: node.terminal };
    }
    if (this.mode === 'follow' && this.follow !== null) {
      return this.followOptions(prefix, this.follow);
    }
    return { next: [], canSubmit: false };
  }

  // ---- follows --------------------------------------------------------------

  private planFollow(lead: ReadonlyArray<Card>, myCards: ReadonlyArray<Card>): FollowPlan | null {
    const ctx = this.ctx;
    const n = this.n;
    const k = lead.length;
    // A follow longer than the wire format can carry is left to the caller
    // (the actor falls back to a bot, as it did for oversized candidates).
    if (k > this.maxCards) return null;
    const ledSuit = effectiveSuit(lead[0]!, ctx);
    const inLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);
    for (const c of inLed) this.led.add(cardSlot(c));
    const req = Math.min(inLed.length, k);

    let reqNK = 0;
    let tractorParts: Array<Map<number, number>> | null = null;
    const cls = classifyCombo(lead, ctx, n);
    // A lead that is not a single combo is a throw, which only obliges the
    // follower to match count and suit (`validateFollowThrow`).
    if (cls.ok && cls.combo.kind !== 'single') {
      const leadNK = cls.combo.kind === 'nkind' ? (cls.combo.n >= n ? 1 : 0) : cls.combo.length;
      const handNK = [...this.led].filter((s) => this.held.get(s)! >= n).length;
      reqNK = Math.min(handNK, leadNK);
      if (cls.combo.kind === 'tractor') {
        const L = cls.combo.length;
        if (findTractors(inLed, ctx, n).some((t) => t.length === L)) {
          // The hand can match the tractor, so the led part must be one.
          // Every led n-of-a-kind set of size L is a candidate; the engine's
          // own tractor finder decides which ones are tractors, which also
          // covers the alternative picks at the off-suit trump-rank tier
          // that a single `findTractors` call over the hand would collapse.
          const nkSlots = [...this.led].filter((s) => this.held.get(s)! >= n);
          tractorParts = [];
          for (const subset of subsetsOfSize(nkSlots, L)) {
            const cards = subset.flatMap((s) => this.physical.get(s)!.slice(0, n));
            if (findTractors(cards, ctx, n).some((t) => t.length === L)) {
              tractorParts.push(new Map(subset.map((s) => [s, n])));
            }
          }
        }
      }
    }
    return { k, req, fill: k - req, reqNK, tractorParts };
  }

  private followOptions(prefix: ReadonlyArray<number>, plan: FollowPlan): SequentialOptions {
    if (prefix.length >= plan.k) return { next: [], canSubmit: prefix.length === plan.k };
    const n = this.n;
    const inPrefix = new Map<number, number>();
    let a = 0;
    let b = 0;
    for (const s of prefix) {
      inPrefix.set(s, (inPrefix.get(s) ?? 0) + 1);
      if (this.led.has(s)) a += 1;
      else b += 1;
    }
    const lastKey = prefix.length > 0 ? this.key.get(prefix[prefix.length - 1]!)! : Infinity;

    // Suffix sums over `order` (descending key): cards still addable
    // strictly below position i, split by led / not led, plus the count of
    // led slots holding a full n-of-a-kind.
    const m = this.order.length;
    const belowL = new Array<number>(m + 1).fill(0);
    const belowN = new Array<number>(m + 1).fill(0);
    const belowNK = new Array<number>(m + 1).fill(0);
    for (let i = m - 1; i >= 0; i--) {
      const s = this.order[i]!;
      const h = this.held.get(s)!;
      const isLed = this.led.has(s);
      belowL[i] = belowL[i + 1]! + (isLed ? h : 0);
      belowN[i] = belowN[i + 1]! + (isLed ? 0 : h);
      belowNK[i] = belowNK[i + 1]! + (isLed && h >= n ? 1 : 0);
    }
    let nkInPrefix = 0;
    for (const [s, c] of inPrefix) if (this.led.has(s) && c >= n) nkInPrefix += 1;

    const next: number[] = [];
    for (let i = 0; i < m; i++) {
      const c = this.order[i]!;
      const kc = this.key.get(c)!;
      if (kc > lastKey) continue;
      const usedC = inPrefix.get(c) ?? 0;
      const heldC = this.held.get(c)!;
      if (usedC >= heldC) continue;
      // Allowing a repeat of the last slot only: a slot above the last one
      // is excluded by the key check, and an equal key means the same slot.
      const cLed = this.led.has(c);
      const a2 = a + (cLed ? 1 : 0);
      const b2 = b + (cLed ? 0 : 1);
      if (a2 > plan.req || b2 > plan.fill) continue;
      const usedC2 = usedC + 1;
      const restC = heldC - usedC2; // further copies of c still addable
      const availL = belowL[i + 1]! + (cLed ? restC : 0);
      const availN = belowN[i + 1]! + (cLed ? 0 : restC);
      if (a2 + availL < plan.req || b2 + availN < plan.fill) continue;

      if (plan.tractorParts !== null) {
        if (!this.tractorFeasible(inPrefix, c, kc, plan.tractorParts)) continue;
      } else if (plan.reqNK > 0) {
        // Most n-of-a-kinds reachable: those already complete, topping up c
        // if it is a led slot one short, then whole n-of-a-kinds below c.
        const budget = plan.req - a2;
        let nk = nkInPrefix + (cLed && usedC2 === n ? 1 : 0);
        let left = budget;
        if (cLed && usedC2 < n && heldC >= n && left >= n - usedC2) {
          nk += 1;
          left -= n - usedC2;
        }
        nk += Math.min(belowNK[i + 1]!, Math.floor(left / n));
        if (nk < plan.reqNK) continue;
      }
      next.push(c);
    }
    return { next, canSubmit: false };
  }

  private tractorFeasible(
    inPrefix: ReadonlyMap<number, number>,
    c: number,
    kc: number,
    parts: ReadonlyArray<ReadonlyMap<number, number>>,
  ): boolean {
    const after = new Map(inPrefix);
    after.set(c, (after.get(c) ?? 0) + 1);
    outer: for (const part of parts) {
      // The prefix (plus c) must sit inside the part...
      for (const [s, cnt] of after) {
        if ((part.get(s) ?? 0) < cnt) continue outer;
      }
      // ...and whatever the part still needs must rank at or below c.
      for (const [s, cnt] of part) {
        const have = after.get(s) ?? 0;
        if (cnt > have && this.key.get(s)! > kc) continue outer;
      }
      return true;
    }
    return false;
  }

  // ---- leads ----------------------------------------------------------------

  private buildLeadTrie(
    myCards: ReadonlyArray<Card>,
    pooled: ReadonlyArray<Card>,
  ): Map<string, { next: Set<number>; terminal: boolean }> {
    const ctx = this.ctx;
    const n = this.n;
    const trie = new Map<string, { next: Set<number>; terminal: boolean }>();
    const node = (sig: string) => {
      let v = trie.get(sig);
      if (v === undefined) {
        v = { next: new Set(), terminal: false };
        trie.set(sig, v);
      }
      return v;
    };
    node('');
    const insert = (slots: number[]): void => {
      slots.sort((x, y) => this.key.get(y)! - this.key.get(x)!);
      for (let i = 0; i < slots.length; i++) {
        node(slots.slice(0, i).join(',')).next.add(slots[i]!);
      }
      node(slots.join(',')).terminal = true;
    };

    const bySuit = new Map<EffectiveSuit, number[]>();
    for (const s of this.held.keys()) {
      const es = effectiveSuit(cardForSlot(s), ctx);
      const list = bySuit.get(es);
      if (list === undefined) bySuit.set(es, [s]);
      else list.push(s);
    }

    for (const [suit, slots] of bySuit) {
      // Singles are always legal leads.
      for (const s of slots) insert([s]);

      // Keys safe to throw as singles: nothing unseen beats them.
      const safeSingle = slots.filter((s) => {
        const cls = classifyCombo([this.physical.get(s)![0]!], ctx, n);
        return cls.ok && componentsSafe([cls.combo], suit, pooled, ctx, n);
      });
      const nkSlots = slots.filter((s) => this.held.get(s)! >= n);
      // A throw is its n-of-a-kind part plus single copies of safe-single
      // keys, so enumerate the n-of-a-kind part, keep it if every
      // component it decomposes into is safe, then add safe singles.
      const limit = Math.min(nkSlots.length, 14);
      for (let mask = 0; mask < 1 << limit; mask++) {
        const nkPart: number[] = [];
        for (let j = 0; j < limit; j++) if (mask & (1 << j)) nkPart.push(nkSlots[j]!);
        const nkCards = nkPart.flatMap((s) => this.physical.get(s)!.slice(0, n));
        if (nkCards.length > this.maxCards) continue;
        let partSafe = true;
        if (nkCards.length > 0) {
          if (classifyCombo(nkCards, ctx, n).ok) {
            insert(nkPart.flatMap((s) => new Array<number>(n).fill(s)));
          }
          const d = decomposePlay(nkCards, ctx, n);
          partSafe = d !== null && componentsSafe(d.components, suit, pooled, ctx, n);
        }
        if (!partSafe) continue;
        const singles = safeSingle.filter((s) => !nkPart.includes(s));
        for (let sm = 0; sm < 1 << singles.length; sm++) {
          const picked: number[] = [];
          for (let j = 0; j < singles.length; j++) if (sm & (1 << j)) picked.push(singles[j]!);
          if (picked.length + nkCards.length < 2) continue;
          if (picked.length + nkCards.length > this.maxCards) continue;
          if (picked.length === 0) continue; // the n-of-a-kind part alone was handled above
          const cards = [...nkCards, ...picked.map((s) => this.physical.get(s)![0]!)];
          if (classifyCombo(cards, ctx, n).ok) continue;
          if (isFairLead(cards, pooled, ctx, n)) {
            insert([...nkPart.flatMap((s) => new Array<number>(n).fill(s)), ...picked]);
          }
        }
        // An n-of-a-kind part that is not itself a combo is a throw of
        // pairs and tractors with no singles.
        if (nkCards.length > 0 && !classifyCombo(nkCards, ctx, n).ok) {
          if (isFairLead(nkCards, pooled, ctx, n)) {
            insert(nkPart.flatMap((s) => new Array<number>(n).fill(s)));
          }
        }
      }
    }
    return trie;
  }
}

function* subsetsOfSize<T>(items: ReadonlyArray<T>, k: number): Generator<T[]> {
  const cur: T[] = [];
  function* rec(start: number): Generator<T[]> {
    if (cur.length === k) {
      yield [...cur];
      return;
    }
    for (let i = start; i <= items.length - (k - cur.length); i++) {
      cur.push(items[i]!);
      yield* rec(i + 1);
      cur.pop();
    }
  }
  yield* rec(0);
}
