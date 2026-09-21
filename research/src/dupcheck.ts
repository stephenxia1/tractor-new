// Verifies the duplicate-deal seeding scheme:  npx tsx src/dupcheck.ts
//
// The whole point of duplicate scoring is that the two halves of a pair are
// the *same cards*. That is an assumption worth testing rather than
// trusting, so this compares the dealt hands card for card between two
// slots that share a --deal-seed-base but differ in every other way that
// could plausibly perturb an RNG stream: which seats the learner controls,
// which bot fills the rest, and how the learner picks among candidates.
//
// It also checks the negative direction (a different base gives different
// cards) and that the 32-bit deal digest carried on the wire agrees with
// the card-for-card comparison, since the Python arena verifies pairs by
// digest and not by cards.

import { mulberry32, type BotType } from '@tractor/engine';

import { GameSlot, type DealtCards } from './env';

const argOf = (name: string, fallback: number): number => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? Number(process.argv[i + 1]) : fallback;
};
const HANDS = argOf('hands', 12);
const GAMES = argOf('games', 3);
// The opponent bot only has to be *different* between the two halves to
// prove the deal does not depend on it. smart-v18 is ~50x slower per hand
// than lowball, so the default keeps the check quick; --heavy runs the
// paired sides against the smart bots the arena actually uses.
const HEAVY = process.argv.includes('--heavy');
const BOT_A = HEAVY ? 'smart-v18' : 'lowball';
const BOT_B = HEAVY ? 'smart-v34' : 'random';

type Observed = {
  readonly key: string; // `${gameId}:${handId}`
  readonly deal: DealtCards;
  readonly digest: number;
};

// Runs one slot for `hands` hands, recording the deal at the start of each.
// The learner policy is a seeded random choice among candidates; passing a
// different policySeed is what makes the two halves of a pair diverge in
// play while (we claim) sharing a deal.
function runSlot(opts: {
  gameId: number;
  dealSeedBase: number;
  learnerSeats: ReadonlyArray<boolean>;
  baselineBot: BotType;
  policySeed: number;
  slotSeed: number;
  hands: number;
}): Observed[] {
  const slot = new GameSlot({
    gameId: opts.gameId,
    seed: opts.slotSeed,
    learnerSeats: opts.learnerSeats,
    baselineBot: opts.baselineBot,
    prePlayBot: 'smart-v34',
    dealSeedBase: opts.dealSeedBase,
    captureDeal: true,
  });
  const rng = mulberry32(opts.policySeed);
  const seen: Observed[] = [];
  const recorded = new Set<string>();

  let guard = 0;
  while (seen.length < opts.hands) {
    if (guard++ > 2_000_000) throw new Error('dupcheck made no progress');
    const { pending } = slot.advance();
    // advance() has already started the hand, so the deal is visible even
    // before the first learner decision of that hand.
    const deal = slot.dealt;
    if (deal !== null) {
      const key = `${slot.gameId}:${slot.handId}`;
      if (!recorded.has(key)) {
        recorded.add(key);
        seen.push({ key, deal, digest: slot.dealDigest });
      }
    }
    if (pending === null) continue;
    slot.choose(Math.floor(rng() * pending.candidates.length));
  }
  return seen;
}

// Cards only: the kitty and all four hands, with landlord and trump rank
// deliberately excluded. A negative control must use this one, because
// landlord and trumpRank come off the same dealRng stream as the shuffle
// and so differ first; comparing them would let the control pass without a
// single card ever being looked at, turning "different base differs" into a
// statement about two bytes of metadata.
function cardsEqual(a: DealtCards, b: DealtCards): boolean {
  if (a.kitty.join(',') !== b.kitty.join(',')) return false;
  if (a.hands.length !== b.hands.length) return false;
  for (let s = 0; s < a.hands.length; s++) {
    if (a.hands[s]!.join(',') !== b.hands[s]!.join(',')) return false;
  }
  return true;
}

function dealEquals(a: DealtCards, b: DealtCards): boolean {
  if (a.landlord !== b.landlord) return false;
  if (a.trumpRank !== b.trumpRank) return false;
  return cardsEqual(a, b);
}

function describe(d: DealtCards): string {
  return `landlord ${d.landlord} trump ${d.trumpRank} seat0 ${d.hands[0]!.slice(0, 6).join(' ')}...`;
}

function main(): void {
  let checkedHands = 0;
  let checkedCards = 0;

  for (let game = 0; game < GAMES; game++) {
    const base = 4242;
    // Side A: learner on seats 0 and 2, smart-v18 filling 1 and 3.
    const sideA = runSlot({
      gameId: game,
      dealSeedBase: base,
      learnerSeats: [true, false, true, false],
      baselineBot: BOT_A,
      policySeed: 11 + game,
      slotSeed: 777 + game,
      hands: HANDS,
    });
    // Side B: learner on the other two seats, a different bot filling the
    // rest, a different policy stream, and a different slot RNG seed.
    const sideB = runSlot({
      gameId: game,
      dealSeedBase: base,
      learnerSeats: [false, true, false, true],
      baselineBot: BOT_B,
      policySeed: 999_983 - game,
      slotSeed: 123_457 + game,
      hands: HANDS,
    });

    if (sideA.length !== sideB.length) {
      throw new Error(`hand counts differ: ${sideA.length} vs ${sideB.length}`);
    }
    for (let i = 0; i < sideA.length; i++) {
      const a = sideA[i]!;
      const b = sideB[i]!;
      if (a.key !== b.key) {
        throw new Error(`hand keys out of step: ${a.key} vs ${b.key}`);
      }
      if (!dealEquals(a.deal, b.deal)) {
        throw new Error(
          `deal mismatch at ${a.key}\n  A: ${describe(a.deal)}\n  B: ${describe(b.deal)}`,
        );
      }
      // The wire digest must agree with the card-for-card verdict, since
      // the Python arena only ever sees the digest.
      if (a.digest !== b.digest) {
        throw new Error(
          `cards match but digests differ at ${a.key}: ${a.digest} vs ${b.digest}`,
        );
      }
      checkedHands += 1;
      for (const h of a.deal.hands) checkedCards += h.length;
      checkedCards += a.deal.kitty.length;
    }
  }

  // Negative control: a different base must give different cards, or the
  // positive result above would be vacuous.
  const other = runSlot({
    gameId: 0,
    dealSeedBase: 4243,
    learnerSeats: [true, false, true, false],
    baselineBot: BOT_A,
    policySeed: 11,
    slotSeed: 777,
    hands: HANDS,
  });
  const reference = runSlot({
    gameId: 0,
    dealSeedBase: 4242,
    learnerSeats: [true, false, true, false],
    baselineBot: BOT_A,
    policySeed: 11,
    slotSeed: 777,
    hands: HANDS,
  });
  let identical = 0;
  for (let i = 0; i < other.length; i++) {
    if (cardsEqual(other[i]!.deal, reference[i]!.deal)) identical += 1;
  }
  if (identical > 0) {
    throw new Error(
      `${identical}/${other.length} deals identical across different bases`,
    );
  }

  // Distinct game slots under one base must also differ, otherwise the
  // arena would be replaying one deal in parallel.
  const g0 = reference[0]!.deal;
  const g1 = runSlot({
    gameId: 1,
    dealSeedBase: 4242,
    learnerSeats: [true, false, true, false],
    baselineBot: BOT_A,
    policySeed: 11,
    slotSeed: 777,
    hands: 1,
  })[0]!.deal;
  if (cardsEqual(g0, g1)) {
    throw new Error('game 0 and game 1 were dealt the same cards');
  }

  // Digests must be distinctive too, or digest-equality would prove nothing.
  const digests = new Set(reference.map((o) => o.digest));
  if (digests.size !== reference.length) {
    throw new Error(
      `digest collisions: ${reference.length} deals -> ${digests.size} digests`,
    );
  }

  console.log(
    `paired deals verified   ${checkedHands} (${GAMES} game slots x ${HANDS} hands, cards + landlord + trump)`,
  );
  console.log(`cards compared          ${checkedCards}`);
  console.log(`different base differs   yes (0/${other.length} identical on cards alone)`);
  console.log(`different game differs   yes (cards alone)`);
  console.log(`digests unique           ${digests.size}/${reference.length}`);
  console.log('dupcheck OK');
}

main();
