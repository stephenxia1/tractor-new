// Loads the weights `research/py/tractor/export_ts.py` writes: the play,
// declaration and burial networks of one PPO checkpoint, float16 on disk,
// widened to float32 once at load.
//
// Regenerate with:
//   cd research/py && python -m tractor.export_ts <checkpoint.pt> ../../packages/engine/src/rl/zero

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export type TensorSpec = { readonly shape: number[]; readonly offset: number };

export type ZeroModelConfig = {
  readonly d_model: number;
  readonly n_layers_history: number;
  readonly n_layers_query: number;
  readonly n_heads: number;
  readonly d_ff: number;
  readonly max_history: number;
  readonly max_candidates: number;
  readonly scalar_d: number;
  readonly ffn: string;
  readonly [key: string]: unknown;
};

export type ZeroDeclareConfig = {
  readonly width: number;
  readonly depth: number;
  readonly residual: boolean;
  readonly ffn: string;
};

export type ZeroBuryConfig = {
  readonly d: number;
  readonly layers: number;
  readonly heads: number;
  readonly value_width: number;
  readonly ffn: string;
};

export type ZeroManifest = {
  readonly format: number;
  readonly source: string;
  readonly step: number;
  readonly elements: number;
  readonly model_config: ZeroModelConfig;
  readonly declare_config: ZeroDeclareConfig;
  readonly bury_config: ZeroBuryConfig;
  readonly tensors: Record<string, TensorSpec>;
};

export type ZeroWeights = {
  readonly manifest: ZeroManifest;
  /** The tensor called `name`, row-major. Throws when it is missing or,
   *  given `shape`, shaped otherwise: a broken export must not quietly
   *  play as a weaker bot. */
  get(name: string, ...shape: number[]): Float32Array;
  has(name: string): boolean;
};

// IEEE-754 half -> single, through a lookup table: 64k entries once,
// rather than Math.pow for each of eleven million values.
let HALF: Float32Array | null = null;
function halfTable(): Float32Array {
  if (HALF !== null) return HALF;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = (h & 0x8000) !== 0 ? -1 : 1;
    const exponent = (h & 0x7c00) >> 10;
    const fraction = h & 0x03ff;
    t[h] =
      exponent === 0
        ? sign * Math.pow(2, -14) * (fraction / 1024)
        : exponent === 0x1f
          ? fraction === 0
            ? sign * Infinity
            : NaN
          : sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  }
  HALF = t;
  return t;
}

const BASE = 'weights-zero';
const cache = new Map<string, ZeroWeights>();

/** The weights in `dir` (default: this module's directory). Loaded once
 *  per directory. */
export function loadZeroWeights(dir?: string): ZeroWeights {
  const here = dir ?? dirname(fileURLToPath(import.meta.url));
  const hit = cache.get(here);
  if (hit !== undefined) return hit;
  const manifest = JSON.parse(readFileSync(join(here, `${BASE}.json`), 'utf-8')) as ZeroManifest;
  if (manifest.format !== 1) throw new Error(`weights-zero: unknown format ${manifest.format}`);
  if (manifest.model_config.ffn !== 'gelu') {
    throw new Error(`weights-zero: the port implements GELU networks, not ${manifest.model_config.ffn}`);
  }
  const raw = readFileSync(join(here, `${BASE}.bin`));
  if (raw.byteLength !== manifest.elements * 2) {
    throw new Error(`weights-zero: ${raw.byteLength} bytes, expected ${manifest.elements * 2}`);
  }
  // A view when the buffer is aligned (a file this size is never pooled,
  // so it is), a copy otherwise: the ws server has 256 MB to hold this in.
  const halves =
    raw.byteOffset % 2 === 0
      ? new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2)
      : new Uint16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const table = halfTable();
  const all = new Float32Array(manifest.elements);
  for (let i = 0; i < all.length; i++) all[i] = table[halves[i]!]!;

  const loaded: ZeroWeights = {
    manifest,
    has: (name) => manifest.tensors[name] !== undefined,
    get(name: string, ...shape: number[]): Float32Array {
      const spec = manifest.tensors[name];
      if (spec === undefined) throw new Error(`weights-zero: no tensor named ${name}`);
      if (shape.length > 0 && shape.join('x') !== spec.shape.join('x')) {
        throw new Error(`weights-zero: ${name} is ${spec.shape.join('x')}, expected ${shape.join('x')}`);
      }
      const size = spec.shape.reduce((a, b) => a * b, 1);
      return all.subarray(spec.offset, spec.offset + size);
    },
  };
  cache.set(here, loaded);
  return loaded;
}
