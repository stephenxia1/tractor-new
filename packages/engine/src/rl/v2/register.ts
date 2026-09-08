// Installs the trained v2 agent as the `rl-v2` bot.
//
// Import this once, for its side effect, from any Node entry point that
// plays games — the ws server, the research workers, offline tournaments:
//
//   import '@tractor/engine/rl/v2/register';
//
// It is separate from `bot.ts` because loading weights needs `node:fs`,
// while `bot.ts` is reachable from the web app's client bundle, where a
// `node:` import fails the build outright. Selecting `rl-v2` without this
// import throws with that instruction rather than quietly playing as some
// other bot.

import { pickBotAction, registerV2Agent, registerV2SearchAgents } from '../../bot';

import { makeV2Agent } from './agent';
import { DEFAULT_BELIEF_WEIGHTS } from './beliefSampler';
import { makeV2SearchAgent } from './searchAgent';

registerV2Agent(
  makeV2Agent({
    // Declaring, kitty burial and the incremental deal were smart-v34 in
    // every training game, so they stay smart-v34 here.
    fallback: (state, seat) => pickBotAction(state, seat, 'smart-v34'),
  }),
);

const searchFallback = (state: import('../../state').GameState, seat: import('../../state').Seat) =>
  pickBotAction(state, seat, 'smart-v34');

// Search cost is linear in worlds x candidates, and the right point on that
// curve is an empirical question rather than a constant: at 2 worlds the
// search was measurably worth nothing (+0.82 [-4.52, +6.16] against the
// bare network). Overridable so a sweep does not need a new bot type per
// setting; the defaults are what the app plays with.
const envInt = (name: string, fallbackValue: number): number => {
  const raw = process.env[name];
  const n = raw === undefined ? NaN : Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallbackValue;
};
const worlds = envInt('TRACTOR_SEARCH_WORLDS', 4);
const maxSearched = envInt('TRACTOR_SEARCH_MAX_CANDIDATES', 16);

registerV2SearchAgents(
  makeV2SearchAgent({ worlds, maxSearched, belief: DEFAULT_BELIEF_WEIGHTS, fallback: searchFallback }),
  // Identical but for the sampler, so any difference between the two is
  // the belief prior and nothing else.
  makeV2SearchAgent({ worlds, maxSearched, belief: null, fallback: searchFallback }),
);
