// Installs the PPO research agent trained from zero as the `rl-zero` bot.
//
// Import this once, for its side effect, from any Node entry point that
// plays games (the ws server, offline matches):
//
//   import '@tractor/engine/rl/zero/register';
//
// Separate from `bot.ts` because loading weights needs `node:fs`, while
// `bot.ts` is reachable from the web app's client bundle. Selecting
// `rl-zero` without this import throws with that instruction.

import { registerZeroAgent, registerZeroSearchAgent } from '../../bot';

import { makeZeroAgent, makeZeroSearchAgent } from './agent';
import { DEFAULT_SEARCH } from './search';

// TRACTOR_ZERO_WEIGHTS points at another export's directory (tests, a
// checkpoint being tried out); unset, the weights shipped beside this file.
const weightsDir = process.env.TRACTOR_ZERO_WEIGHTS || undefined;
registerZeroAgent(makeZeroAgent({ weightsDir }));

// `rl-zero-search`. Its cost is linear in worlds x top-k; the defaults are
// the setting the research evaluation measured (8 worlds, top 3, searched
// below 0.99 confidence), overridable without a new bot type.
const envNum = (name: string, fallback: number): number => {
  const raw = process.env[name];
  const n = raw === undefined || raw === '' ? NaN : Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
};
registerZeroSearchAgent(
  makeZeroSearchAgent({
    weightsDir,
    search: {
      worlds: Math.max(1, Math.floor(envNum('TRACTOR_ZERO_SEARCH_WORLDS', DEFAULT_SEARCH.worlds))),
      topK: Math.max(2, Math.floor(envNum('TRACTOR_ZERO_SEARCH_TOP_K', DEFAULT_SEARCH.topK))),
      extraTricks: Math.floor(envNum('TRACTOR_ZERO_SEARCH_EXTRA_TRICKS', DEFAULT_SEARCH.extraTricks)),
      confident: envNum('TRACTOR_ZERO_SEARCH_CONFIDENT', DEFAULT_SEARCH.confident),
    },
  }),
);
