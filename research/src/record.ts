// Moved into the engine so the trainer and the shipped bot share one
// implementation: see packages/engine/src/rl/v2/record.ts. A second copy
// here would drift, and a drift between the encoding training saw and the
// encoding inference produces is silent -- the bot simply plays worse.
export * from '@tractor/engine/rl/v2/record';
