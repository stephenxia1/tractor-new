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

import { registerZeroAgent } from '../../bot';

import { makeZeroAgent } from './agent';

// TRACTOR_ZERO_WEIGHTS points at another export's directory (tests, a
// checkpoint being tried out); unset, the weights shipped beside this file.
registerZeroAgent(makeZeroAgent({ weightsDir: process.env.TRACTOR_ZERO_WEIGHTS || undefined }));
