// Dumps real decision records together with the TypeScript featuriser's
// and network's output, so `py/test_ts_parity.py` can diff them against
// the Python ones the weights were actually trained under.
//
//   npx tsx src/dumpParity.ts --decisions 200 > parity.jsonl
//
// A disagreement here does not crash anything -- it just makes the shipped
// bot play a slightly different game from the one that was measured, which
// is precisely the sort of thing that never gets noticed. Hence the test.

import { GameSlot } from './env';
import { writeDecision } from './encode';
import { ByteWriter, H_ROLE, H_TRUMP_RANK, H_TRUMP_SUIT } from './record';
import {
  actionFeatures,
  historyFeatures,
  stateFeatures,
} from '@tractor/engine/rl/v2/features';
import { loadWeights, scoreCandidates } from '@tractor/engine/rl/v2/net';
import { CANDIDATE_BYTES, STATE_BYTES } from '@tractor/engine/rl/v2/record';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const target = Number(arg('decisions', '200'));
const seed = Number(arg('seed', '20260907'));
const withScores = arg('scores', '1') !== '0';

const weights = withScores ? loadWeights() : null;
const slot = new GameSlot({
  gameId: 0,
  seed,
  learnerSeats: [true, true, true, true],
  baselineBot: 'smart-v34',
  prePlayBot: 'smart-v34',
  maxCandidates: 48,
  randomize: true,
});

let emitted = 0;
let guard = 0;
while (emitted < target && guard++ < 200_000) {
  const { pending } = slot.advance();
  if (pending === null) continue;

  const w = new ByteWriter(1 << 16);
  writeDecision(
    w,
    slot.gameState,
    {
      gameId: slot.gameId,
      handId: slot.handId,
      seat: pending.seat,
      myKitty: slot.kitty,
      privileged: false,
    },
    pending.candidates,
  );
  // writeDecision prefixes the record with its u16 length.
  const framed = w.taken();
  const recBytes = framed[0]! + framed[1]! * 256;
  const record = framed.subarray(2, 2 + recBytes);
  const state = record.subarray(0, STATE_BYTES);

  const feats = stateFeatures(state);
  const hist = historyFeatures(state);
  const rank = state[H_TRUMP_RANK]!;
  const suit = state[H_TRUMP_SUIT]!;
  const role = state[H_ROLE]!;

  const nCands = (recBytes - STATE_BYTES) / CANDIDATE_BYTES;
  const actions: Float32Array[] = [];
  for (let c = 0; c < nCands; c++) {
    const at = STATE_BYTES + c * CANDIDATE_BYTES;
    actions.push(actionFeatures(record.subarray(at, at + CANDIDATE_BYTES), rank, suit));
  }

  const scores =
    weights === null
      ? null
      : Array.from(scoreCandidates(weights, feats, hist.tokens, hist.mask, actions, role));

  process.stdout.write(
    JSON.stringify({
      record: Buffer.from(record).toString('base64'),
      role,
      numCandidates: nCands,
      state: Array.from(feats),
      tokens: Array.from(hist.tokens),
      mask: Array.from(hist.mask),
      actions: actions.map((a) => Array.from(a)),
      scores,
    }) + '\n',
  );
  emitted += 1;

  // Keep the game moving: any legal choice will do, and varying it keeps
  // the sample from being one bot's narrow trajectory.
  slot.choose(emitted % pending.candidates.length);
}

process.stderr.write(`dumped ${emitted} decisions\n`);
