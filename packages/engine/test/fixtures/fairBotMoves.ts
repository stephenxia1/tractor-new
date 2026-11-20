// The standing benchmark bots' moves on a fixed set of positions, so a
// change to bot.ts can be checked to leave them exactly as they were.
//
// Positions come from whole hands played by the cheap v1 bot (`smart`)
// from fixed seeds; at every declaration and burial decision, and at
// every PLAY_STRIDE-th play decision, smart-v18 / v28 / v34 are asked
// what they would do. The engine is a parameter so the same walk can be
// run against an older checkout of the engine to make the fixture:
//
//   git archive 8d173cb packages/engine | tar -x -C /tmp/base
//   npx tsx packages/engine/test/fixtures/fairBotMoves.ts /tmp/base/packages/engine/src \
//     > packages/engine/test/fixtures/fair-bot-moves.json
//
// (8d173cb is the commit smart-v35 was branched from.)

export const FAIR_BOTS = ['smart-v18', 'smart-v28', 'smart-v34'] as const;
export const SEEDS = [101, 202, 303] as const;
export const PLAY_STRIDE = 8;

export type Recorded = {
  readonly seed: number;
  readonly step: number;
  readonly phase: string;
  readonly seat: number;
  readonly moves: Record<string, string>;
};

// Only what is used, so an older engine's module satisfies it too.
type Engine = {
  runHand: (opts: { agents: ReadonlyArray<unknown>; seed: number }) => unknown;
  pickBotAction: (state: any, seat: number, type: string) => any;
  cardKey: (c: any) => string;
};

function describe(action: any, cardKey: (c: any) => string): string {
  if (action.kind === 'play_cards' || action.kind === 'declare' || action.kind === 'replace_kitty') {
    const cards = (action.cards as any[]).map((c) => `${cardKey(c)}#${c.deckId}`).sort();
    return `${action.kind}:${cards.join(',')}`;
  }
  return action.kind;
}

export function collectFairBotMoves(engine: Engine): Recorded[] {
  const out: Recorded[] = [];
  for (const seed of SEEDS) {
    let step = 0;
    let plays = 0;
    const agent = (state: any, seat: number): unknown => {
      const here = step++;
      const isPlay = state.phase === 'play';
      if (!isPlay || plays++ % PLAY_STRIDE === 0) {
        const moves: Record<string, string> = {};
        for (const bot of FAIR_BOTS) {
          moves[bot] = describe(engine.pickBotAction(state, seat, bot), engine.cardKey);
        }
        out.push({ seed, step: here, phase: state.phase, seat, moves });
      }
      return engine.pickBotAction(state, seat, 'smart');
    };
    engine.runHand({ agents: [agent, agent, agent, agent], seed });
  }
  return out;
}

// CLI: print the fixture for the engine whose src/ directory is given.
const isMain = process.argv[1] !== undefined && /fairBotMoves\.ts$/.test(process.argv[1]);
if (isMain) {
  const dir = process.argv[2];
  if (dir === undefined) throw new Error('usage: fairBotMoves.ts <engine src dir>');
  const engine = (await import(`${dir}/index.ts`)) as Engine;
  process.stdout.write(JSON.stringify(collectFairBotMoves(engine), null, 0) + '\n');
}
