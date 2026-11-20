// smart-v35: the fair search bot meant as a harder benchmark than smart-v34.
//
// Strength is measured elsewhere (research/src/botMatch.ts, duplicate
// deals against smart-v34). What is pinned here is what a benchmark has
// to be: registered, legal on every move, and fair -- its decision is a
// function of what its seat can see, so re-dealing every card it cannot
// see must not change what it plays.

import { describe, expect, it } from 'vitest';
import {
  BOT_TYPES,
  VISIBLE_BOT_TYPES,
  makeSmartV35Agent,
  pickBotAction,
  smartV35Worlds,
  type BotAction,
} from '../src/bot';
import { buildDeck } from '../src/deck';
import { runHand } from '../src/eval';
import { applyEvent } from '../src/reducer';
import { effectiveSuit } from '../src/ordering';
import { mulberry32, shuffle } from '../src/prng';
import { sumPoints } from '../src/scoring';
import { createGame, defaultV1Config, type GameState, type Seat } from '../src/state';
import { cardKey, type Card, type EffectiveSuit, type TrumpContext } from '../src/types';

const identity = (c: Card): string => `${cardKey(c)}#${c.deckId}`;
const keysOf = (cards: ReadonlyArray<Card>): string => cards.map(cardKey).sort().join(' ');
const playedCards = (a: BotAction): string => (a.kind === 'play_cards' ? keysOf(a.cards) : a.kind);

// Positions where seat 0 or 2 must play, from hands played by the cheap
// v1 bot (the positions are the point, not who reached them).
function playPositions(seeds: ReadonlyArray<number>): Array<{ state: GameState; seat: Seat }> {
  const out: Array<{ state: GameState; seat: Seat }> = [];
  for (const seed of seeds) {
    const agent = (s: GameState, seat: Seat): BotAction => {
      if (s.phase === 'play' && seat % 2 === 0) out.push({ state: s, seat });
      return pickBotAction(s, seat, 'smart');
    };
    runHand({ agents: [agent, agent, agent, agent], seed });
  }
  return out;
}

// The same position with every card `seat` cannot see dealt again: the
// other hands (sizes kept) and, unless `seat` is the landlord, the kitty.
function redeal(state: GameState, seat: Seat, seed: number): GameState {
  const hand = state.currentHand!;
  const others = [0, 1, 2, 3].filter((s) => s !== seat);
  const seesKitty = hand.landlord === seat;
  const pool = shuffle(
    [...others.flatMap((s) => hand.hands[s] ?? []), ...(seesKitty ? [] : hand.kitty)],
    mulberry32(seed),
  );
  const hands: Record<Seat, ReadonlyArray<Card>> = { [seat]: hand.hands[seat] ?? [] };
  let at = 0;
  for (const s of others) {
    const n = (hand.hands[s] ?? []).length;
    hands[s] = pool.slice(at, at + n);
    at += n;
  }
  const kitty = seesKitty ? hand.kitty : pool.slice(at);
  return { ...state, currentHand: { ...hand, hands, kitty } };
}

// Shown voids, from public play only (rules.md §7.3), computed here
// independently of the bot.
function shownVoids(state: GameState, ctx: TrumpContext): Array<Set<EffectiveSuit>> {
  const voids = [0, 1, 2, 3].map(() => new Set<EffectiveSuit>());
  const hand = state.currentHand!;
  const tricks = [...hand.tricks.map((t) => t.plays), ...(hand.currentTrick ? [hand.currentTrick.plays] : [])];
  for (const plays of tricks) {
    const led = effectiveSuit(plays[0]!.cards[0]!, ctx);
    for (const p of plays.slice(1)) {
      if (p.cards.some((c) => effectiveSuit(c, ctx) !== led)) voids[p.seat]!.add(led);
    }
  }
  return voids;
}

describe('smart-v35', () => {
  it('is registered for evaluation but not offered in the lobby', () => {
    expect(BOT_TYPES).toContain('smart-v35');
    expect(VISIBLE_BOT_TYPES).not.toContain('smart-v35');
  });

  it('plays whole hands with every move accepted by the engine', () => {
    // runHand throws on any refused move or a stalled hand.
    for (const seed of [11, 12]) {
      const { summary } = runHand({
        agents: ['smart-v35', 'smart-v34', 'smart-v35', 'smart'],
        seed,
      });
      expect(summary.attackerPoints).toBeGreaterThanOrEqual(0);
    }
    const cheap = makeSmartV35Agent({ budget: 400, maxWorlds: 8, minWorlds: 2 });
    for (const seed of [21, 22, 23]) {
      runHand({ agents: [cheap, cheap, cheap, cheap], seed });
    }
  }, 120_000);

  it('decides from what its seat can see: re-dealing hidden cards changes nothing', () => {
    const positions = playPositions([31, 32]);
    // Every fourth position: early, middle and late tricks, both seats.
    const picked = positions.filter((_, i) => i % 4 === 0).slice(0, 14);
    expect(picked.length).toBeGreaterThan(8);
    let redealtDiffers = 0;
    for (const [i, { state, seat }] of picked.entries()) {
      const before = playedCards(pickBotAction(state, seat, 'smart-v35'));
      for (let k = 0; k < 2; k++) {
        const other = redeal(state, seat, 1000 * i + k);
        const otherSeat = (seat + 1) % 4;
        if (keysOf(other.currentHand!.hands[otherSeat]!) !== keysOf(state.currentHand!.hands[otherSeat]!)) {
          redealtDiffers++;
        }
        expect(playedCards(pickBotAction(other, seat, 'smart-v35'))).toBe(before);
      }
    }
    // The control: the re-deal really moved hidden cards.
    expect(redealtDiffers).toBeGreaterThan(picked.length);
  }, 120_000);

  it('deals worlds that respect hand sizes, shown voids and declarations', () => {
    const positions = playPositions([41, 42, 43]);
    let checked = 0;
    let voidsSeen = 0;
    for (const { state, seat } of positions.filter((_, i) => i % 3 === 0)) {
      const hand = state.currentHand!;
      const ctx: TrumpContext = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };
      const iAmLandlord = hand.landlord === seat;
      // What this seat cannot see, by physical card.
      const seen = new Set<string>((hand.hands[seat] ?? []).map(identity));
      for (const t of hand.tricks) for (const p of t.plays) for (const c of p.cards) seen.add(identity(c));
      for (const p of hand.currentTrick?.plays ?? []) for (const c of p.cards) seen.add(identity(c));
      if (iAmLandlord) for (const c of hand.kitty) seen.add(identity(c));
      const unseen = buildDeck(2).map(identity).filter((k) => !seen.has(k)).sort();
      const voids = shownVoids(state, ctx);
      for (const world of smartV35Worlds(state, seat, 12)) {
        const dealt: string[] = [];
        for (let s = 0; s < 4; s++) {
          if (s === seat) {
            expect(world.hands[s]!.length).toBe(0);
            continue;
          }
          expect(world.hands[s]!.length).toBe((hand.hands[s] ?? []).length);
          for (const c of world.hands[s]!) {
            expect(voids[s]!.has(effectiveSuit(c, ctx))).toBe(false);
            dealt.push(identity(c));
          }
          voidsSeen += voids[s]!.size;
        }
        expect(world.kitty.length).toBe(iAmLandlord ? 0 : hand.kitty.length);
        for (const c of world.kitty) dealt.push(identity(c));
        expect(dealt.sort()).toEqual(unseen);
        // A card shown when declaring and not played since is still with
        // its declarer (or in the kitty, if the declarer is the landlord).
        const where = new Map<string, number>();
        world.hands.forEach((h, s) => h.forEach((c) => where.set(identity(c), s)));
        world.kitty.forEach((c) => where.set(identity(c), -1));
        for (const d of hand.declarationLog) {
          for (const c of d.cards) {
            if (seen.has(identity(c)) || d.seat === seat) continue;
            const at = where.get(identity(c));
            expect(at === d.seat || (at === -1 && d.seat === hand.landlord)).toBe(true);
          }
        }
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(50);
    expect(voidsSeen).toBeGreaterThan(0);
  }, 60_000);

  it("buries only cards it holds, choosing between v34's burial and a point-keeping one", () => {
    let v35Points = 0;
    let v34Points = 0;
    let hands = 0;
    let differs = 0;
    for (let seed = 1; seed <= 30; seed++) {
      let s = createGame(defaultV1Config());
      s = applyEvent(s, { kind: 'start_hand', seed }).state;
      for (let i = 0; i < 50 && s.phase !== 'kitty_replace'; i++) {
        if (s.phase === 'declare_lockout') {
          let r = applyEvent(s, { kind: 'release_lockout', seed });
          for (const seat of [0, 1, 2, 3]) {
            const a = pickBotAction(s, seat, 'smart-v34');
            if (a.kind !== 'declare') continue;
            const d = applyEvent(s, { kind: 'declare', seat, cards: a.cards });
            if (d.ok) r = d;
            break;
          }
          if (!r.ok) break;
          s = r.state;
          continue;
        }
        if (s.phase === 'declare_open') {
          s = applyEvent(s, { kind: 'pick_up_kitty', seat: s.currentHand!.landlord! }).state;
        }
      }
      if (s.phase !== 'kitty_replace') continue;
      const landlord = s.currentHand!.landlord!;
      const mine = pickBotAction(s, landlord, 'smart-v35');
      const theirs = pickBotAction(s, landlord, 'smart-v34');
      expect(mine.kind).toBe('replace_kitty');
      if (mine.kind !== 'replace_kitty' || theirs.kind !== 'replace_kitty') continue;
      expect(mine.cards.length).toBe(8);
      expect(new Set(mine.cards.map(identity)).size).toBe(8);
      expect(applyEvent(s, { kind: 'replace_kitty', seat: landlord, cards: mine.cards }).ok).toBe(true);
      v35Points += sumPoints(mine.cards);
      v34Points += sumPoints(theirs.cards);
      if (keysOf(mine.cards) !== keysOf(theirs.cards)) differs++;
      hands++;
    }
    expect(hands).toBeGreaterThan(20);
    // The alternative is chosen on some hands, not all, and it only ever
    // keeps points back.
    expect(differs).toBeGreaterThan(0);
    expect(differs).toBeLessThan(hands);
    expect(v35Points).toBeLessThan(v34Points);
  }, 60_000);
});
