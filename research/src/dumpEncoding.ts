// Dumps the engine's chain-position table so the Python featuriser's copy
// can be diffed against it. Run: npx tsx src/dumpEncoding.ts > /tmp/enc.json
import { ENC_OF_RAW, CAPACITY } from '@tractor/engine/rl/v2/encoding';

const out = {
  encOfRaw: ENC_OF_RAW.map((byRank) => byRank.map((m) => Array.from(m))),
  capacity: CAPACITY.map((byRank) => byRank.map((m) => Array.from(m))),
};
process.stdout.write(JSON.stringify(out));
