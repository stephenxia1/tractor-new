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

import { GameSlot } from './env';
import { writeDecision, writeEpisode } from './encode';
import type { BotType } from '@tractor/engine';
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
};

function parseArgs(argv: ReadonlyArray<string>): Args {
  const get = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  // Seat mask: bit i set means seat i is learner-controlled. 15 = all four.
  const mask = Number(get('learner-seats') ?? '15');
  return {
    games: Number(get('games') ?? '32'),
    seed: Number(get('seed') ?? '1'),
    learnerSeats: [0, 1, 2, 3].map((s) => (mask & (1 << s)) !== 0),
    baseline: (get('baseline') ?? 'smart-v34') as BotType,
    preplay: (get('preplay') ?? 'smart-v34') as BotType,
    maxCandidates: Number(get('max-cands') ?? '48'),
    randomize: (get('randomize') ?? '1') !== '0',
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

async function main(): Promise<void> {
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
        },
        pending.candidates,
      );
      pendingSeat.set(slot.gameId, pending.candidates.length);
      nDecisions += 1;
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
