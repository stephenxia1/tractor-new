// smart-v18 / v28 / v34 are the research track's standing benchmark, with
// historical records against them, and smart-v35 was added beside them in
// the same file. This holds them to the moves the engine made before
// smart-v35 existed (commit 8d173cb), at every declaration and burial and
// at a spread of play decisions -- see fixtures/fairBotMoves.ts for how
// the fixture is made.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cardKey, pickBotAction, runHand } from '../src';
import { collectFairBotMoves, type Recorded } from './fixtures/fairBotMoves';

const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/fair-bot-moves.json', import.meta.url), 'utf8'),
) as Recorded[];

describe('the fair benchmark bots are unchanged', () => {
  it('smart-v18 / v28 / v34 make the same moves as before smart-v35', () => {
    const now = collectFairBotMoves({ runHand: runHand as never, pickBotAction: pickBotAction as never, cardKey });
    expect(now.length).toBe(fixture.length);
    const phases = new Set(fixture.map((r) => r.phase));
    expect(phases.has('play')).toBe(true);
    expect(phases.has('kitty_replace')).toBe(true);
    expect(fixture.filter((r) => r.phase === 'play').length).toBeGreaterThan(30);
    for (let i = 0; i < fixture.length; i++) {
      expect(now[i]).toEqual(fixture[i]);
    }
  }, 600_000);
});
