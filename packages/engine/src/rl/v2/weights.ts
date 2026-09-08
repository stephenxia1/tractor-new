// Loads the exported v2 weights.
//
// Stored as float16 (5.9 MB rather than 11.8) because the file ships
// inside the ws server's Docker image and that machine has 256 MB of RAM.
// They are widened to float32 once at load; the forward pass is float32
// throughout, so half precision costs storage only.
//
// Regenerate with:
//   python research/py/export_v2_weights.py --checkpoint <ckpt>

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export type TensorSpec = { readonly shape: number[]; readonly offset: number };

export type WeightManifest = {
  readonly format: string;
  readonly count: number;
  readonly steps: number;
  readonly hands: number;
  readonly net_config: Record<string, unknown>;
  readonly tensors: Record<string, TensorSpec>;
};

export type LoadedWeights = {
  readonly manifest: WeightManifest;
  /** Throws rather than returning undefined: a missing tensor is a broken
   *  export, and silently substituting zeros would look like a weak bot. */
  get(name: string, ...shape: number[]): Float32Array;
};

// IEEE-754 half -> single. Node has no Float16Array, and the conversion
// runs once per tensor at startup, so a straightforward decode is fine.
function halfToFloat(h: number): number {
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exponent = (h & 0x7c00) >> 10;
  const fraction = h & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

let cached: LoadedWeights | null = null;

export function loadWeights(dir?: string): LoadedWeights {
  if (cached !== null && dir === undefined) return cached;
  const here = dir ?? dirname(fileURLToPath(import.meta.url));
  const manifest = JSON.parse(
    readFileSync(join(here, 'weights-v2.json'), 'utf-8'),
  ) as WeightManifest;
  if (manifest.format !== 'float16') {
    throw new Error(`unexpected weight format ${manifest.format}`);
  }

  const raw = readFileSync(join(here, 'weights-v2.bin'));
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const all = new Float32Array(manifest.count);
  for (let i = 0; i < manifest.count; i++) all[i] = halfToFloat(view.getUint16(i * 2, true));

  const cache = new Map<string, Float32Array>();
  const loaded: LoadedWeights = {
    manifest,
    get(name: string, ...shape: number[]): Float32Array {
      const hit = cache.get(name);
      if (hit !== undefined) return hit;
      const spec = manifest.tensors[name];
      if (spec === undefined) throw new Error(`weights: no tensor named ${name}`);
      const size = spec.shape.reduce((a, b) => a * b, 1);
      if (shape.length > 0) {
        const want = shape.join('x');
        const got = spec.shape.join('x');
        if (want !== got) throw new Error(`weights: ${name} is ${got}, expected ${want}`);
      }
      const slice = all.subarray(spec.offset, spec.offset + size);
      cache.set(name, slice);
      return slice;
    },
  };
  if (dir === undefined) cached = loaded;
  return loaded;
}
