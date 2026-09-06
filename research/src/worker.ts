// Self-play worker. Holds `--games` independent game slots, advances all
// of them to their next learner decision, ships that batch to the learner
// over stdout, and applies the chosen actions that come back on stdin.
//
// Batching many games per round-trip is what makes the process boundary
// affordable: one forward pass in Python covers every game in flight, and
// the pipe carries ~0.6 KB per decision instead of a tensor.
//
// Usage (the Python side spawns this; the defaults let you run it by hand):
//   tsx src/worker.ts --games 32 --seed 7 --learner-seats 15
//
// For duplicate-deal evaluation, --deal-seed-base makes each slot's deals a
// pure function of (base, gameId, handIndex), so two workers given the same
// base play the same cards while controlling different seats.

import { GameSlot } from './env';
import { writeDecision, writeEpisode } from './encode';
import { NATURAL_RANKS, type BotType, type NaturalRank, type Seat } from '@tractor/engine';
import {
  ByteWriter,
  BATCH_HEADER_BYTES,
  FRAME_HEADER_BYTES,
} from './record';

type Args = {
  games: number;
  seed: number;
  learnerSeats: boolean[];
  baseline: BotType;
  preplay: BotType;
  maxCandidates: number;
  randomize: boolean;
  dealSeedBase: number | undefined;
  fixedLandlord: Seat | undefined;
  fixedLevel: NaturalRank | undefined;
  baselineBySeat: Array<BotType | undefined>;
  // Training only: emit the other seats' hands for a centralised critic.
  privileged: boolean;
  // Value-guided search: worlds sampled and candidates searched per
  // decision. 0 worlds disables it.
  searchWorlds: number;
  searchTop: number;
  searchOpponent: BotType;
};

function parseArgs(argv: ReadonlyArray<string>): Args {
  const get = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  // Seat mask: bit i set means seat i is learner-controlled. 15 = all four.
  const mask = Number(get('learner-seats') ?? '15');
  // Duplicate mode: deals become a pure function of (base, gameId, handIndex).
  const dealBaseRaw = get('deal-seed-base');
  const landlordRaw = get('fixed-landlord');
  const levelRaw = get('fixed-level');
  if (levelRaw !== undefined && !NATURAL_RANKS.includes(levelRaw as NaturalRank)) {
    throw new Error(`--fixed-level ${levelRaw} is not a natural rank`);
  }
  // A second bot for a chosen set of seats, so the two teams can be filled
  // by different heuristics (bot-vs-bot duplicate matches).
  const altBot = get('baseline-alt');
  const altMask = Number(get('alt-seats') ?? '0');
  const baselineBySeat: Array<BotType | undefined> = [0, 1, 2, 3].map((s) =>
    altBot !== undefined && (altMask & (1 << s)) !== 0
      ? (altBot as BotType)
      : undefined,
  );
  return {
    games: Number(get('games') ?? '32'),
    seed: Number(get('seed') ?? '1'),
    learnerSeats: [0, 1, 2, 3].map((s) => (mask & (1 << s)) !== 0),
    baseline: (get('baseline') ?? 'smart-v34') as BotType,
    preplay: (get('preplay') ?? 'smart-v34') as BotType,
    maxCandidates: Number(get('max-cands') ?? '48'),
    randomize: (get('randomize') ?? '1') !== '0',
    dealSeedBase: dealBaseRaw === undefined ? undefined : Number(dealBaseRaw),
    fixedLandlord:
      landlordRaw === undefined ? undefined : (Number(landlordRaw) as Seat),
    fixedLevel: levelRaw === undefined ? undefined : (levelRaw as NaturalRank),
    baselineBySeat,
    privileged: (get('privileged') ?? '0') === '1',
    searchWorlds: Number(get('search-worlds') ?? '0'),
    searchTop: Number(get('search-top') ?? '6'),
    searchOpponent: (get('search-opponent') ?? 'smart-v2') as BotType,
  };
}

// Reassembles length-prefixed frames arriving on a stream.
class FrameReader {
  private chunks: Buffer[] = [];
  private buffered = 0;
  private waiter: ((frame: Buffer | null) => void) | null = null;
  private closed = false;

  constructor(stream: NodeJS.ReadableStream) {
    stream.on('data', (chunk: Buffer) => {
      this.chunks.push(chunk);
      this.buffered += chunk.length;
      this.pump();
    });
    stream.on('end', () => {
      this.closed = true;
      this.pump();
    });
  }

  // Resolves with the next complete frame payload, or null once the
  // stream ends (the learner exiting is the worker's shutdown signal).
  next(): Promise<Buffer | null> {
    return new Promise((resolve) => {
      this.waiter = resolve;
      this.pump();
    });
  }

  private pump(): void {
    if (this.waiter === null) return;
    if (this.buffered < FRAME_HEADER_BYTES) {
      if (this.closed) this.finish(null);
      return;
    }
    const head = this.peek(FRAME_HEADER_BYTES);
    const payloadBytes = head.readUInt32LE(0);
    if (this.buffered < FRAME_HEADER_BYTES + payloadBytes) {
      if (this.closed) this.finish(null);
      return;
    }
    const all = this.take(FRAME_HEADER_BYTES + payloadBytes);
    this.finish(all.subarray(FRAME_HEADER_BYTES));
  }

  private finish(frame: Buffer | null): void {
    const waiter = this.waiter;
    this.waiter = null;
    if (waiter !== null) waiter(frame);
  }

  private peek(n: number): Buffer {
    if (this.chunks.length === 1) return this.chunks[0]!;
    const merged = Buffer.concat(this.chunks, this.buffered);
    this.chunks = [merged];
    return merged.subarray(0, n);
  }

  private take(n: number): Buffer {
    const merged =
      this.chunks.length === 1 ? this.chunks[0]! : Buffer.concat(this.chunks, this.buffered);
    const out = merged.subarray(0, n);
    const rest = merged.subarray(n);
    this.chunks = rest.length > 0 ? [rest] : [];
    this.buffered = rest.length;
    return out;
  }
}

function writeAll(buf: Uint8Array): Promise<void> {
  return new Promise((resolve) => {
    const ok = process.stdout.write(buf);
    if (ok) resolve();
    else process.stdout.once('drain', () => resolve());
  });
}

// The learner closing its end is the normal shutdown signal, but it can
// land mid-write and surface as an unhandled EPIPE/EOF on stdout. Exit
// quietly on those so a clean shutdown does not look like a crash (and a
// real error still does).
function installPipeGuards(): void {
  const quiet = (err: NodeJS.ErrnoException): void => {
    if (err.code === 'EPIPE' || err.code === 'EOF') process.exit(0);
    throw err;
  };
  process.stdout.on('error', quiet);
  process.stdin.on('error', quiet);
}

async function main(): Promise<void> {
  installPipeGuards();
  const args = parseArgs(process.argv.slice(2));
  const slots: GameSlot[] = [];
  for (let i = 0; i < args.games; i++) {
    slots.push(
      new GameSlot({
        gameId: i,
        seed: args.seed * 1_000_003 + i * 7919,
        learnerSeats: args.learnerSeats,
        baselineBot: args.baseline,
        prePlayBot: args.preplay,
        maxCandidates: args.maxCandidates,
        randomize: args.randomize,
        dealSeedBase: args.dealSeedBase,
        fixedLandlord: args.fixedLandlord,
        fixedLevel: args.fixedLevel,
        baselineBotBySeat: args.baselineBySeat,
      }),
    );
  }

  const reader = new FrameReader(process.stdin);
  const writer = new ByteWriter(1 << 20);
  const pendingSeat = new Map<number, number>(); // gameId -> pending index guard

  for (;;) {
    writer.reset();
    const lenAt = writer.reserve(FRAME_HEADER_BYTES);
    const countsAt = writer.reserve(BATCH_HEADER_BYTES);
    const payloadStart = writer.length;

    let nDecisions = 0;
    const episodes: Array<Parameters<typeof writeEpisode>[1]> = [];
    pendingSeat.clear();

    for (const slot of slots) {
      const { episodes: finished, pending } = slot.advance();
      for (const e of finished) episodes.push(e);
      if (pending === null) continue;
      writeDecision(
        writer,
        slot.gameState,
        {
          gameId: slot.gameId,
          handId: slot.handId,
          seat: pending.seat,
          myKitty: slot.kitty,
          privileged: args.privileged,
        },
        pending.candidates,
      );
      pendingSeat.set(slot.gameId, pending.candidates.length);
      nDecisions += 1;

      // Leaves ride the same frame as ordinary records with no candidates,
      // tagged with the candidate they inform.
      if (args.searchWorlds > 0) {
        const result = slot.searchLeaves({
          worlds: args.searchWorlds,
          topCandidates: args.searchTop,
          opponentModel: args.searchOpponent,
          seed: (slot.gameId + 1) * 7919 + slot.handId,
        });
        if (result !== null) {
          for (const leaf of result.leaves) {
            writeDecision(
              writer,
              leaf.state,
              {
                gameId: slot.gameId,
                handId: slot.handId,
                seat: pending.seat,
                myKitty: slot.kitty,
                privileged: false,
                leafOf: leaf.candidateIndex,
                leafPoints: leaf.points,
              },
              [],
            );
            nDecisions += 1;
          }
        }
      }
    }
    for (const e of episodes) writeEpisode(writer, e);

    const payloadBytes = writer.length - payloadStart + BATCH_HEADER_BYTES;
    writer.patchU32(lenAt, payloadBytes);
    writer.patchU16(countsAt, nDecisions);
    writer.patchU16(countsAt + 2, episodes.length);
    await writeAll(writer.taken());

    const reply = await reader.next();
    if (reply === null) return; // learner closed the pipe

    const nActions = reply.readUInt16LE(0);
    let off = 2;
    for (let i = 0; i < nActions; i++) {
      const gameId = reply.readUInt16LE(off);
      const candidateIndex = reply.readUInt8(off + 2);
      off += 3;
      const slot = slots[gameId];
      if (slot === undefined) throw new Error(`unknown gameId ${gameId}`);
      slot.choose(candidateIndex);
    }
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`worker failed: ${String(err)}\n`);
  process.exit(1);
});
