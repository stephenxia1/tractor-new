// AUTO-GENERATED placeholder weights for the DMC agent.
//
// This file is overwritten by `python/dmc/export_weights.py` after
// training. The initial committed version contains zero weights so the
// build succeeds and the `rl-dmc` bot can be loaded — but with zero
// weights, the bot's Q-values are all identical and it just picks the
// first enumerated candidate. Train + export before deploying.
//
// Schema is defined in `./weights.ts` (`SerializedWeights`). Sizes are
// derived from `./inference.ts` and must match `python/dmc/network.py`.

import type { SerializedWeights } from './weights';

const ENCODING_LEN = 108;
const PER_STEP_DIM = ENCODING_LEN + 4 + 1; // 113
const SCALAR_DIM = 16;
const STATIC_DIM = 7 * ENCODING_LEN; // 756
const STATIC_FC_IN = STATIC_DIM + SCALAR_DIM; // 772
const HIDDEN_STATIC_1 = 512;
const HIDDEN_STATIC = 256;
const HIDDEN_HIST = 128;
const HIDDEN_ACT = 64;
const HIDDEN_HEAD_1 = 256;
const HIDDEN_HEAD_2 = 128;
const HEAD_IN = HIDDEN_STATIC + HIDDEN_HIST + HIDDEN_ACT;
const ACTION_DIM = ENCODING_LEN + 5 + 2; // 115

function zeros(n: number): number[] {
  const a = new Array<number>(n);
  for (let i = 0; i < n; i++) a[i] = 0;
  return a;
}

function zeroRole() {
  return {
    staticW1: zeros(HIDDEN_STATIC_1 * STATIC_FC_IN),
    staticB1: zeros(HIDDEN_STATIC_1),
    staticW2: zeros(HIDDEN_STATIC * HIDDEN_STATIC_1),
    staticB2: zeros(HIDDEN_STATIC),
    lstmWi: zeros(4 * HIDDEN_HIST * PER_STEP_DIM),
    lstmWh: zeros(4 * HIDDEN_HIST * HIDDEN_HIST),
    lstmBi: zeros(4 * HIDDEN_HIST),
    lstmBh: zeros(4 * HIDDEN_HIST),
    actionW: zeros(HIDDEN_ACT * ACTION_DIM),
    actionB: zeros(HIDDEN_ACT),
    headW1: zeros(HIDDEN_HEAD_1 * HEAD_IN),
    headB1: zeros(HIDDEN_HEAD_1),
    headW2: zeros(HIDDEN_HEAD_2 * HIDDEN_HEAD_1),
    headB2: zeros(HIDDEN_HEAD_2),
    headW3: zeros(1 * HIDDEN_HEAD_2),
    headB3: zeros(1),
  };
}

export const EMBEDDED_DMC_WEIGHTS: SerializedWeights = {
  schema_version: 1,
  landlord: zeroRole(),
  defender: zeroRole(),
  attacker: zeroRole(),
};
