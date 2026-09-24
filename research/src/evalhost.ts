// An NDJSON host, so a Python-side network can play duplicate-deal
// matches against the engine's heuristic bots:
//
//   npx tsx research/src/evalhost.ts --deals 96 --seed-base 12345 --bot smart-v34
//
// The learner already meets the TypeScript environment through
// src/worker.ts, but that speaks a packed binary frame format built for
// throughput: many games per round trip, a fixed-width encoding, and the
// feature layout of packages/engine/src/rl/v2 baked into it. The Python
// research stack has its own simulator and its own encoder, and what it
// needs from TypeScript is not features but *opponents* -- the heuristic
// bots, which exist nowhere else. So this host ships the position rather
// than an encoding of it, in the plainest form both sides already agree
// on: card multisets over the 54-slot index of rl/v2/cards.ts. The Python
// side rebuilds a PlayState from that and encodes it itself, which is
// what keeps the measured policy identical to the trained one.
//
// One hand per slot, two slots per deal. Half 0 puts the network on seats
// 0 and 2 and the bot on 1 and 3; half 1 swaps them. Both halves share a
// --seed-base and a gameId, so `GameSlot`'s duplicate mode deals them the
// identical cards and the part of the result that came from the cards
// cancels in the pair. That claim is checked rather than trusted: see
// `fingerprintPlayStart` below.
//
// stdout carries the protocol and nothing else. Anything worth saying
// goes to stderr.

import { createInterface } from 'node:readline';

import { NATURAL_RANKS, type BotType, type Card } from '@tractor/engine';

import { NUM_SLOTS, cardSlot } from './cards';
import { trumpSuitIndex, type EpisodeSummary } from './encode';
import { GameSlot } from './env';

const NUM_SEATS = 4;

const argOf = (name: string, fallback: number): number => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? Number(process.argv[i + 1]) : fallback;
};
const strOf = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
};

// Counts per slot index, which is how every card multiset crosses the
// wire. Two physical copies of a card share a slot, so a hand, a play and
// a candidate are all the same shape and the Python side needs one reader.
function plane(cards: ReadonlyArray<Card>): number[] {
  const out = new Array<number>(NUM_SLOTS).fill(0);
  for (const c of cards) {
    const slot = cardSlot(c);
    out[slot] = (out[slot] ?? 0) + 1;
  }
  return out;
}

// Sum of two count vectors, for reconstructing a seat's dealt hand from
// what it still holds plus what it has played.
function addPlanes(a: number[], b: ReadonlyArray<number>): number[] {
  for (let i = 0; i < a.length; i++) a[i] = (a[i] ?? 0) + (b[i] ?? 0);
  return a;
}

// --- protocol i/o ---------------------------------------------------------

// Lines arriving on stdin, one await at a time. `readline` handles the
// chunk reassembly and the trailing-newline cases; the async iterator is
// pulled explicitly rather than looped over because the host reads a
// reply at a point of its choosing, not whenever one happens to arrive.
function lineReader(): () => Promise<string | null> {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const it = rl[Symbol.asyncIterator]();
  return async () => {
    const step = await it.next();
    return step.done === true ? null : step.value;
  };
}

function writeLine(obj: unknown): Promise<void> {
  return new Promise((resolve) => {
    const ok = process.stdout.write(`${JSON.stringify(obj)}\n`);
    if (ok) resolve();
    else process.stdout.once('drain', () => resolve());
  });
}

// The client closing its end is the normal shutdown signal and can land
// mid-write, surfacing as an unhandled EPIPE. Exit quietly on those so a
// clean shutdown does not look like a crash; a real error still does.
function installPipeGuards(): void {
  const quiet = (err: NodeJS.ErrnoException): void => {
    if (err.code === 'EPIPE' || err.code === 'EOF') process.exit(0);
    throw err;
  };
  process.stdout.on('error', quiet);
  process.stdin.on('error', quiet);
}

// --- one hand -------------------------------------------------------------

type HalfResult = {
  readonly digest: number;
  // Everything about the position at the moment play began: the dealt
  // hands, the burial, the landlord and the trump context. See
  // `fingerprintPlayStart`.
  readonly playStart: string;
  readonly episode: EpisodeSummary;
  readonly decisions: number;
};

// The deal digest is taken right after the shuffle, so it cannot see the
// declaration or the burial -- and those are what decide the trump suit
// and which eight cards leave the landlord's hand. Two halves could
// therefore agree on the digest and still have entered the play phase
// with different trump or a different kitty, which would break the
// pairing in a way the digest is structurally unable to report.
//
// This closes that gap. A seat's dealt hand is what it still holds plus
// everything it has played, so the whole play-start position can be
// recovered from any later position and compared between the halves even
// though they reach their first learner decision at different points.
function fingerprintPlayStart(
  hands: ReadonlyArray<ReadonlyArray<number>>,
  playedBySeat: ReadonlyArray<ReadonlyArray<number>>,
  kitty: ReadonlyArray<number>,
  landlord: number,
  levelRank: number,
  trumpSuit: number,
): string {
  const dealt: number[][] = [];
  for (let seat = 0; seat < NUM_SEATS; seat++) {
    dealt.push(addPlanes([...(hands[seat] ?? [])], playedBySeat[seat] ?? []));
  }
  return JSON.stringify([dealt, kitty, landlord, levelRank, trumpSuit]);
}

// Plays exactly one hand in one slot, emitting a `pending` for every
// learner decision and reading the reply before continuing.
async function playHand(
  slot: GameSlot,
  game: number,
  half: number,
  readLine: () => Promise<string | null>,
): Promise<HalfResult> {
  let decisions = 0;
  let playStart: string | null = null;

  for (;;) {
    let advanced;
    try {
      advanced = slot.advance();
    } catch (err) {
      // A move the engine refused. Report the position rather than the
      // message alone: which seat, what was led, and what the refused
      // play held is the whole of the diagnosis.
      const hs = slot.gameState.currentHand;
      const trick = hs?.currentTrick ?? null;
      const lead = trick?.plays?.[0];
      process.stderr.write(
        'evalhost: rejected move. ' +
          JSON.stringify({
            leader: trick?.leader ?? null,
            plays: trick?.plays?.length ?? 0,
            leadCards: lead ? lead.cards.map((c: any) => cardSlot(c)) : null,
            leadSize: lead ? lead.cards.length : null,
            trumpSuit: hs?.trumpSuit ?? null,
            trumpRank: hs?.trumpRank ?? null,
          }) + String.fromCharCode(10),
      );
      throw err;
    }
    const { episodes, pending } = advanced;
    if (episodes.length > 0) {
      const episode = episodes[0]!;
      if (playStart === null) throw new Error('hand ended with no learner decision');
      return { digest: episode.dealDigest, playStart, episode, decisions };
    }
    if (pending === null) throw new Error('slot stalled with nothing pending');

    const state = slot.gameState;
    const hand = state.currentHand;
    if (hand === null || hand.trumpRank === null || hand.landlord === null) {
      throw new Error('pending decision outside a live hand');
    }

    const hands: number[][] = [];
    const playedBySeat: number[][] = [];
    for (let seat = 0; seat < NUM_SEATS; seat++) {
      hands.push(plane(hand.hands[seat] ?? []));
      playedBySeat.push(new Array<number>(NUM_SLOTS).fill(0));
    }

    const tricks = hand.tricks.map((t) => {
      const plays: number[][] = [];
      for (let i = 0; i < t.plays.length; i++) {
        const play = t.plays[i]!;
        // The Python rebuild attributes play i of a trick to seat
        // (leader + i) % 4. That is how the engine fills a trick, but a
        // silent change to either side would mis-attribute every card
        // played, so it is asserted rather than assumed.
        if (play.seat !== (t.leader + i) % NUM_SEATS) {
          throw new Error(
            `trick play ${i} is seat ${play.seat}, expected ${(t.leader + i) % NUM_SEATS}`,
          );
        }
        const counts = plane(play.cards);
        addPlanes(playedBySeat[play.seat]!, counts);
        plays.push(counts);
      }
      return { leader: t.leader, winner: t.winner, points: t.points, plays };
    });

    const open = hand.currentTrick;
    const trickPlays: number[][] = [];
    if (open !== null) {
      for (let i = 0; i < open.plays.length; i++) {
        const play = open.plays[i]!;
        if (play.seat !== (open.leader + i) % NUM_SEATS) {
          throw new Error(
            `open play ${i} is seat ${play.seat}, expected ${(open.leader + i) % NUM_SEATS}`,
          );
        }
        const counts = plane(play.cards);
        addPlanes(playedBySeat[play.seat]!, counts);
        trickPlays.push(counts);
      }
    }

    const kitty = plane(hand.kitty);
    const levelRank = NATURAL_RANKS.indexOf(hand.trumpRank);
    const trumpSuit = trumpSuitIndex(hand);
    if (playStart === null) {
      playStart = fingerprintPlayStart(
        hands,
        playedBySeat,
        kitty,
        hand.landlord,
        levelRank,
        trumpSuit,
      );
    }

    await writeLine({
      t: 'pending',
      game,
      half,
      seat: pending.seat,
      level_rank: levelRank,
      trump_suit: trumpSuit,
      num_decks: state.config.numDecks,
      landlord: hand.landlord,
      to_act: pending.seat,
      // A trick that has not opened yet is led by whoever is to act.
      trick_leader: open === null ? pending.seat : open.leader,
      hands,
      kitty,
      tricks,
      trick_plays: trickPlays,
      candidates: pending.candidates.map((c) => plane(c.cards)),
    });

    const reply = await readLine();
    if (reply === null) throw new Error('client closed the stream mid-hand');
    const parsed = JSON.parse(reply) as { t?: string; index?: number };
    if (parsed.t !== 'choose' || typeof parsed.index !== 'number') {
      throw new Error(`expected a choose message, got ${reply.slice(0, 120)}`);
    }
    slot.choose(parsed.index);
    decisions += 1;
  }
}

// --- main -----------------------------------------------------------------

async function main(): Promise<void> {
  installPipeGuards();
  const deals = argOf('deals', 96);
  const seedBase = argOf('seed-base', 12345);
  const bot = strOf('bot', 'smart-v34') as BotType;
  const maxCandidates = argOf('max-candidates', 128);
  // Who declares and buries. `random` is the training generator's draw,
  // so neither side plays from a position its own heuristic set up;
  // `bot` hands both to the opponent bot at every seat, the network's
  // included, which is how every measurement before 2026-09-23 was taken.
  const prePlay = strOf('pre-play', 'random');
  if (prePlay !== 'random' && prePlay !== 'bot') {
    throw new Error(`--pre-play must be random or bot, got ${prePlay}`);
  }
  // Shard: this host plays the games congruent to `shard` modulo
  // `shards`. A deal is a pure function of the seed base and its game
  // id, so the shards are independent and their union is the whole set
  // -- which is what lets several hosts run at once. The heuristic bots
  // search, and at better than half a second a decision one host is far
  // too slow to measure anything with.
  const shards = argOf('shards', 1);
  const shard = argOf('shard', 0);
  if (!Number.isInteger(deals) || deals < 1) {
    throw new Error(`--deals must be a positive integer, got ${deals}`);
  }
  if (!Number.isInteger(shards) || shards < 1) {
    throw new Error(`--shards must be a positive integer, got ${shards}`);
  }
  if (!Number.isInteger(shard) || shard < 0 || shard >= shards) {
    throw new Error(`--shard must lie in [0, ${shards}), got ${shard}`);
  }

  const readLine = lineReader();
  // Seats the network controls, by half. Each half is one whole team, so
  // "the network's result" on a hand is a single number.
  const halves: ReadonlyArray<ReadonlyArray<boolean>> = [
    [true, false, true, false],
    [false, true, false, true],
  ];

  const started = Date.now();
  let decisions = 0;
  for (let game = shard; game < deals; game += shards) {
    const results: HalfResult[] = [];
    for (let half = 0; half < halves.length; half++) {
      const slot = new GameSlot({
        gameId: game,
        // Only reached for the kitty perturbation (off here) and the
        // lockout-release draw, neither of which can move the deal. It
        // differs between the halves on purpose: a deal that depended on
        // it would then show up as a mismatch below rather than hiding.
        seed: seedBase * 1_000_003 + game * 7919 + half,
        learnerSeats: halves[half]!,
        baselineBot: bot,
        prePlayBot: bot,
        randomPrePlay: prePlay === 'random',
        maxCandidates,
        randomize: false,
        dealSeedBase: seedBase,
      });
      const result = await playHand(slot, game, half, readLine);
      decisions += result.decisions;
      results.push(result);

      const e = result.episode;
      await writeLine({
        t: 'hand',
        game,
        half,
        landlord: e.landlordSeat,
        attacker_points: e.attackerPoints,
        defender_gain: e.defendersWon ? e.levelDelta : 0,
        attacker_gain: e.defendersWon ? 0 : e.levelDelta,
        digest: result.digest,
      });
    }

    const [a, b] = results as [HalfResult, HalfResult];
    if (a.digest !== b.digest) {
      throw new Error(
        `deal ${game}: halves were dealt different cards (digest ${a.digest} vs ${b.digest})`,
      );
    }
    if (a.playStart !== b.playStart) {
      throw new Error(
        `deal ${game}: halves entered play from different positions ` +
          `(same shuffle, different declaration or burial)`,
      );
    }
  }

  await writeLine({ t: 'done', deals });
  const elapsed = (Date.now() - started) / 1000;
  process.stderr.write(
    `evalhost: ${deals} deals (${2 * deals} hands, ${decisions} learner decisions) ` +
      `vs ${bot} in ${elapsed.toFixed(1)}s (${(elapsed / deals).toFixed(2)}s/deal)\n`,
  );
}

main().catch((err: unknown) => {
  process.stderr.write(`evalhost: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(1);
});
