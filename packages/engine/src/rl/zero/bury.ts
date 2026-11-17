// The PPO bot's kitty burial: research/py/tractor/nn/bury_net.py's v5
// network, eight greedy picks of one card each (sim/bury.py::
// policy_burials with greedy=True), asked as the research evaluation host
// asks it (research/src/evalhost.ts 'bury', eval/bots.py::_bury_choice).
//
// Each pick reads one token per card type: the card's static features in
// the hand's trump context, then the copies still held, already buried,
// and in the landlord's 33, each over the deck count. A context vector
// holds the level, the trump suit, the pick index, the standing
// declaration's tier and its declarer relative to the landlord.

import type { GameState, Seat } from '../../state';
import { NATURAL_RANKS, SUITS, type Card } from '../../types';
import { cardSlot } from '../v2/cards';

import { D_STATIC, NO_TRUMP, NUM_TYPES, staticCardFeatures, trumpCtx } from './ctx';
import { standing } from './declare';
import { layerNormRows, linearRows, maskedArgmax, selfAttention } from './preplayNn';
import type { ZeroWeights } from './weights';

export const BURY_STEPS = 8;
export const BURY_CARD_DIM = D_STATIC + 3; // 31
export const BURY_CTX_DIM = 13 + 5 + BURY_STEPS + 5 + 4; // 35

/** What the evaluation host's `bury` message carries, in the form
 *  `_bury_choice` reads it. */
export type BuryView = {
  readonly levelRank: number;
  /** 0..3, or 4 for no-trump. */
  readonly trumpSuit: number;
  readonly standingTier: number;
  /** The standing declarer relative to the landlord, -1 for none. */
  readonly declarerRel: number;
  /** The landlord's 33 cards as 54 slot counts. */
  readonly pool: ReadonlyArray<number>;
  readonly numDecks: number;
};

/** sim/bury.py::bury_context. */
export function buryContext(v: BuryView, step: number): Float32Array {
  const out = new Float32Array(BURY_CTX_DIM);
  out[v.levelRank] = 1;
  out[13 + v.trumpSuit] = 1;
  out[18 + step] = 1;
  out[18 + BURY_STEPS + v.standingTier] = 1;
  if (v.declarerRel >= 0) out[18 + BURY_STEPS + 5 + v.declarerRel] = 1;
  return out;
}

/** sim/bury.py::_step_inputs for one landlord: card tokens [54, 31] and
 *  the held mask. */
export function buryCardTokens(
  v: BuryView,
  buried: ReadonlyArray<number>,
): { cards: Float32Array; mask: boolean[] } {
  const stat = staticCardFeatures(trumpCtx(v.levelRank, v.trumpSuit));
  const n = v.numDecks;
  const cards = new Float32Array(NUM_TYPES * BURY_CARD_DIM);
  const mask: boolean[] = [];
  for (let t = 0; t < NUM_TYPES; t++) {
    const row = t * BURY_CARD_DIM;
    for (let k = 0; k < D_STATIC; k++) cards[row + k] = stat[t * D_STATIC + k]!;
    const held = v.pool[t]! - buried[t]!;
    cards[row + D_STATIC] = Math.fround(held / n);
    cards[row + D_STATIC + 1] = Math.fround(buried[t]! / n);
    cards[row + D_STATIC + 2] = Math.fround(v.pool[t]! / n);
    mask.push(held > 0);
  }
  return { cards, mask };
}

/** BuryNet.forward's logits [54] (before masking). v5 only: the
 *  pre-norm `nn.TransformerEncoder` with ReLU feed-forwards. */
export function buryLogits(w: ZeroWeights, cards: Float32Array, ctx: Float32Array): Float64Array {
  const cfg = w.manifest.bury_config;
  if (cfg.ffn !== 'gelu') throw new Error('zero bury: only the v5 burial network is ported');
  const d = cfg.d;
  const ff = 2 * d;
  const cardIn = linearRows(
    Float64Array.from(cards),
    NUM_TYPES,
    w.get('bury.card_in.weight', d, BURY_CARD_DIM),
    w.get('bury.card_in.bias', d),
    BURY_CARD_DIM,
    d,
  );
  const ctxIn = linearRows(
    Float64Array.from(ctx),
    1,
    w.get('bury.ctx_in.weight', d, BURY_CTX_DIM),
    w.get('bury.ctx_in.bias', d),
    BURY_CTX_DIM,
    d,
  );
  const embed = w.get('bury.card_embed', NUM_TYPES, d);
  let x = new Float64Array(NUM_TYPES * d);
  for (let t = 0; t < NUM_TYPES; t++) {
    for (let k = 0; k < d; k++) x[t * d + k] = cardIn[t * d + k]! + embed[t * d + k]! + ctxIn[k]!;
  }
  for (let l = 0; l < cfg.layers; l++) {
    const p = `bury.encoder.layers.${l}`;
    // x = x + SA(norm1(x)); x = x + FF(norm2(x))   (norm_first=True)
    const h1 = layerNormRows(x, NUM_TYPES, d, w.get(`${p}.norm1.weight`, d), w.get(`${p}.norm1.bias`, d));
    const sa = selfAttention(
      h1,
      NUM_TYPES,
      d,
      cfg.heads,
      w.get(`${p}.self_attn.in_proj_weight`, 3 * d, d),
      w.get(`${p}.self_attn.in_proj_bias`, 3 * d),
      w.get(`${p}.self_attn.out_proj.weight`, d, d),
      w.get(`${p}.self_attn.out_proj.bias`, d),
    );
    for (let i = 0; i < x.length; i++) x[i] = x[i]! + sa[i]!;
    const h2 = layerNormRows(x, NUM_TYPES, d, w.get(`${p}.norm2.weight`, d), w.get(`${p}.norm2.bias`, d));
    const a = linearRows(h2, NUM_TYPES, w.get(`${p}.linear1.weight`, ff, d), w.get(`${p}.linear1.bias`, ff), d, ff);
    for (let i = 0; i < a.length; i++) if (a[i]! < 0) a[i] = 0;
    const b = linearRows(a, NUM_TYPES, w.get(`${p}.linear2.weight`, d, ff), w.get(`${p}.linear2.bias`, d), ff, d);
    const next = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) next[i] = x[i]! + b[i]!;
    x = next;
  }
  const h = layerNormRows(x, NUM_TYPES, d, w.get('bury.norm.weight', d), w.get('bury.norm.bias', d));
  return linearRows(h, NUM_TYPES, w.get('bury.score.weight', 1, d), w.get('bury.score.bias', 1), d, 1);
}

/** The eight greedy picks, as 54 slot counts, and each pick's card type. */
export function buryChoice(w: ZeroWeights, v: BuryView): { counts: number[]; picks: number[] } {
  const buried = new Array<number>(NUM_TYPES).fill(0);
  const picks: number[] = [];
  for (let step = 0; step < BURY_STEPS; step++) {
    const { cards, mask } = buryCardTokens(v, buried);
    const logits = buryLogits(w, cards, buryContext(v, step));
    const a = maskedArgmax(logits, mask);
    if (a < 0) throw new Error('zero bury: nothing left to bury');
    buried[a] = buried[a]! + 1;
    picks.push(a);
  }
  return { counts: buried, picks };
}

// --- the live engine position -----------------------------------------------

/** The network's view of the burial `seat` (the landlord) faces now, or
 *  null outside 'kitty_replace'. */
export function buryView(state: GameState, seat: Seat): BuryView | null {
  const hand = state.currentHand;
  if (state.phase !== 'kitty_replace' || hand === null) return null;
  if (hand.landlord !== seat || hand.trumpRank === null) return null;
  const pool = new Array<number>(NUM_TYPES).fill(0);
  for (const c of hand.hands[seat] ?? []) {
    const s = cardSlot(c);
    pool[s] = pool[s]! + 1;
  }
  const st = standing(state);
  return {
    levelRank: NATURAL_RANKS.indexOf(hand.trumpRank),
    trumpSuit: hand.trumpSuit === null ? NO_TRUMP : SUITS.indexOf(hand.trumpSuit),
    standingTier: st.tier,
    declarerRel: st.seat < 0 ? -1 : (((st.seat - seat) % 4) + 4) % 4,
    pool,
    numDecks: state.config.numDecks,
  };
}

/** The cards the landlord `seat` buries: the network's eight picks, as
 *  cards from its hand (env.ts `answerBury`). */
export function zeroBury(w: ZeroWeights, state: GameState, seat: Seat): Card[] {
  const view = buryView(state, seat);
  if (view === null) throw new Error(`zero bury: seat ${seat} has no burial to make`);
  if (state.config.kittySize !== BURY_STEPS) {
    throw new Error(`zero bury: the network buries ${BURY_STEPS}, the table ${state.config.kittySize}`);
  }
  const { counts } = buryChoice(w, view);
  const need = [...counts];
  const cards: Card[] = [];
  for (const c of state.currentHand!.hands[seat] ?? []) {
    const s = cardSlot(c);
    if ((need[s] ?? 0) > 0) {
      need[s] = need[s]! - 1;
      cards.push(c);
    }
  }
  if (cards.length !== BURY_STEPS || need.some((x) => x !== 0)) {
    throw new Error('zero bury: the picks do not match the landlord\'s hand');
  }
  return cards;
}
