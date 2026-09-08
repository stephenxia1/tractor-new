// Weight loading + serialization for the DMC Q-networks.
//
// Storage format: a plain JSON object with one entry per role:
//
//   {
//     "schema_version": 1,
//     "landlord": { "staticW1": [...], ..., "headB3": [...] },
//     "defender": { ... },
//     "attacker": { ... }
//   }
//
// Float values are stored as JSON number arrays. Per-role payload is
// ~3.2 MB raw, ~6 MB after JSON-stringifying (number-per-array overhead).
// For the deployed bot we load lazily on first invocation; the full
// bundle is gated behind the agent factory to keep cold-start cheap.

import { ACTION_ENCODING_LEN } from './actions';
import { ENCODING_LEN } from './cardEncoding';
import { PER_STEP_DIM, SCALAR_DIM, STATIC_CHANNELS } from './observation';
import type { DmcWeights, RoleWeights } from './inference';

const SCHEMA_VERSION = 1;

// Same layer sizes as inference.ts — re-derive to avoid coupling.
const STATIC_DIM = STATIC_CHANNELS * ENCODING_LEN; // 756
const STATIC_FC_IN = STATIC_DIM + SCALAR_DIM;      // 772
const HIDDEN_STATIC_1 = 512;
const HIDDEN_STATIC = 256;
const HIDDEN_HIST = 128;
const HIDDEN_ACT = 64;
const HIDDEN_HEAD_1 = 256;
const HIDDEN_HEAD_2 = 128;
const HEAD_IN = HIDDEN_STATIC + HIDDEN_HIST + HIDDEN_ACT;

type SerializedRole = {
  readonly staticW1: number[]; readonly staticB1: number[];
  readonly staticW2: number[]; readonly staticB2: number[];
  readonly lstmWi: number[];   readonly lstmWh: number[];
  readonly lstmBi: number[];   readonly lstmBh: number[];
  readonly actionW: number[];  readonly actionB: number[];
  readonly headW1: number[];   readonly headB1: number[];
  readonly headW2: number[];   readonly headB2: number[];
  readonly headW3: number[];   readonly headB3: number[];
};

export type SerializedWeights = {
  readonly schema_version: number;
  readonly landlord: SerializedRole;
  readonly defender: SerializedRole;
  readonly attacker: SerializedRole;
};

function f32(arr: number[], expectedLen: number, name: string): Float32Array {
  if (arr.length !== expectedLen) {
    throw new Error(
      `weight tensor ${name} has wrong length: got ${arr.length}, expected ${expectedLen}`,
    );
  }
  return Float32Array.from(arr);
}

function deserializeRole(s: SerializedRole, role: string): RoleWeights {
  const ctx = (n: string) => `${role}.${n}`;
  return {
    staticW1: f32(s.staticW1, HIDDEN_STATIC_1 * STATIC_FC_IN, ctx('staticW1')),
    staticB1: f32(s.staticB1, HIDDEN_STATIC_1, ctx('staticB1')),
    staticW2: f32(s.staticW2, HIDDEN_STATIC * HIDDEN_STATIC_1, ctx('staticW2')),
    staticB2: f32(s.staticB2, HIDDEN_STATIC, ctx('staticB2')),
    lstmWi: f32(s.lstmWi, 4 * HIDDEN_HIST * PER_STEP_DIM, ctx('lstmWi')),
    lstmWh: f32(s.lstmWh, 4 * HIDDEN_HIST * HIDDEN_HIST, ctx('lstmWh')),
    lstmBi: f32(s.lstmBi, 4 * HIDDEN_HIST, ctx('lstmBi')),
    lstmBh: f32(s.lstmBh, 4 * HIDDEN_HIST, ctx('lstmBh')),
    actionW: f32(s.actionW, HIDDEN_ACT * ACTION_ENCODING_LEN, ctx('actionW')),
    actionB: f32(s.actionB, HIDDEN_ACT, ctx('actionB')),
    headW1: f32(s.headW1, HIDDEN_HEAD_1 * HEAD_IN, ctx('headW1')),
    headB1: f32(s.headB1, HIDDEN_HEAD_1, ctx('headB1')),
    headW2: f32(s.headW2, HIDDEN_HEAD_2 * HIDDEN_HEAD_1, ctx('headW2')),
    headB2: f32(s.headB2, HIDDEN_HEAD_2, ctx('headB2')),
    headW3: f32(s.headW3, 1 * HIDDEN_HEAD_2, ctx('headW3')),
    headB3: f32(s.headB3, 1, ctx('headB3')),
  };
}

export function deserializeWeights(s: SerializedWeights): DmcWeights {
  if (s.schema_version !== SCHEMA_VERSION) {
    throw new Error(`weights schema mismatch: got ${s.schema_version}, expected ${SCHEMA_VERSION}`);
  }
  return {
    landlord: deserializeRole(s.landlord, 'landlord'),
    defender: deserializeRole(s.defender, 'defender'),
    attacker: deserializeRole(s.attacker, 'attacker'),
  };
}

// Build a fresh weight set with all zeros. Useful for smoke-testing the
// inference path before real weights are exported. Note: actual training
// uses PyTorch's Kaiming-style init; zeros are NOT a good starting point
// for training — only for "does the forward pass run end-to-end" checks.
export function zeroWeights(): DmcWeights {
  const zRole = (): RoleWeights => ({
    staticW1: new Float32Array(HIDDEN_STATIC_1 * STATIC_FC_IN),
    staticB1: new Float32Array(HIDDEN_STATIC_1),
    staticW2: new Float32Array(HIDDEN_STATIC * HIDDEN_STATIC_1),
    staticB2: new Float32Array(HIDDEN_STATIC),
    lstmWi: new Float32Array(4 * HIDDEN_HIST * PER_STEP_DIM),
    lstmWh: new Float32Array(4 * HIDDEN_HIST * HIDDEN_HIST),
    lstmBi: new Float32Array(4 * HIDDEN_HIST),
    lstmBh: new Float32Array(4 * HIDDEN_HIST),
    actionW: new Float32Array(HIDDEN_ACT * ACTION_ENCODING_LEN),
    actionB: new Float32Array(HIDDEN_ACT),
    headW1: new Float32Array(HIDDEN_HEAD_1 * HEAD_IN),
    headB1: new Float32Array(HIDDEN_HEAD_1),
    headW2: new Float32Array(HIDDEN_HEAD_2 * HIDDEN_HEAD_1),
    headB2: new Float32Array(HIDDEN_HEAD_2),
    headW3: new Float32Array(1 * HIDDEN_HEAD_2),
    headB3: new Float32Array(1),
  });
  return { landlord: zRole(), defender: zRole(), attacker: zRole() };
}
