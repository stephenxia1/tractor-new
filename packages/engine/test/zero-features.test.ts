// The rl-zero play featuriser against what the Python encoder the weights
// were trained under makes of the same positions
// (fixtures/zero-features.json, from research/py/test_zero_parity.py
// --fixture). Bit-identical: a drift here would not crash anything, the
// bot would just play a different game from the one measured. The full
// check over thousands of engine positions is research/src/zeroParity.ts
// with research/py/test_zero_parity.py.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { encodePosition, type ZeroPosition } from '../src/rl/zero/features';

type Case = {
  position: ZeroPosition;
  expected: Record<'history' | 'context' | 'unseen' | 'planes' | 'cards', number[]> & { historyLen: number };
};

const here = dirname(fileURLToPath(import.meta.url));
const { cases } = JSON.parse(readFileSync(join(here, 'fixtures/zero-features.json'), 'utf-8')) as {
  cases: Case[];
};

describe('rl-zero play features', () => {
  it('has a spread of positions', () => {
    expect(cases.length).toBeGreaterThanOrEqual(8);
  });
  for (const [i, c] of cases.entries()) {
    it(`matches the Python encoder on case ${i}`, () => {
      const got = encodePosition(c.position);
      expect(got.historyLen).toBe(c.expected.historyLen);
      for (const key of ['history', 'context', 'unseen', 'planes', 'cards'] as const) {
        expect(Array.from(got[key]), key).toEqual(c.expected[key]);
      }
    });
  }
});
