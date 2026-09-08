// Action enumeration + encoding for the DMC agent.
//
// The play phase has two action regimes:
//
//   - **Follow**: the trick already has a lead. Enumerate all legal
//     follow plays (using the existing `allLegalPlayActions` helper);
//     each candidate is scored with one Q-evaluation and the argmax
//     wins. No sub-action: a follow's structure is forced by the lead.
//
//   - **Lead**: we choose a play from scratch. Following Shan §4.5,
//     we decompose multi-pattern leads (throws) into a sequence of
//     "pick one more pattern OR stop" sub-decisions. Each sub-decision
//     is its own Q-evaluation; cards committed in earlier sub-decisions
//     are surfaced to the network via the tentative-committed channel
//     in the observation.
//
//     Sub-action protocol:
//       1. The first sub-action enumerates all single-pattern leads
//          (singles, pairs, tractors) across all effective suits.
//       2. After ≥1 pattern is committed, sub-actions enumerate (a) a
//          STOP option that finalizes the lead, plus (b) any additional
//          single-pattern combos drawn from cards remaining in hand
//          AND restricted to the same effective suit as the committed
//          cards (throws are single-suit per rules §7.6).
//       3. Sub-actions cap at 3 patterns; the 4th forced sub-action
//          can only be STOP.
//
// Each enumerated candidate carries:
//   - `cards`: the cards added by this sub-action (empty for STOP)
//   - `encoding`: a fixed-length float vector for the Q-network
//   - `kind`: a discrete tag distinguishing follow / lead-pattern / stop
//   - `effectiveSuit`: where applicable, the suit the cards belong to
//
// The Python driver, on receiving the candidates, runs Q over each,
// argmaxes, and either commits the choice and recurses (for lead
// sub-actions) or finalizes the play_cards event (for follows / STOP).

import { findNKinds, findTractors } from '../../combos';
import { effectiveSuit } from '../../ordering';
import { handFor, nextToPlay } from '../../reducer';
import { type GameState, type Seat } from '../../state';
import { allLegalPlayActions } from '../mlp/actions';
import {
  cardEquals,
  cardKey,
  type Card,
  type EffectiveSuit,
  type TrumpContext,
} from '../../types';

import {
  encodeCardSet,
  ENCODING_LEN,
  type EncodingContext,
} from './cardEncoding';

// Shape one-hot dimensions. Indices line up with SHAPE_INDEX below.
export const SHAPE_DIM = 5;
// Final action vector: cards (108) + shape one-hot (5) + n_cards/10 (1)
// + all_trump bit (1).
export const ACTION_ENCODING_LEN = ENCODING_LEN + SHAPE_DIM + 2;

export type ActionKind =
  | 'follow'         // a full follow play (singletons / pairs / tractors / discards)
  | 'lead-single'    // first or additional lead pattern, single card
  | 'lead-pair'      // first or additional lead pattern, pair
  | 'lead-tractor'   // first or additional lead pattern, tractor
  | 'lead-stop';     // finalize the lead (commit accumulated patterns)

const SHAPE_INDEX: Record<ActionKind, number> = {
  'follow': 0,
  'lead-single': 1,
  'lead-pair': 2,
  'lead-tractor': 3,
  'lead-stop': 4,
};

export type Candidate = {
  readonly kind: ActionKind;
  // Cards this sub-action adds to the play (empty for lead-stop).
  readonly cards: ReadonlyArray<Card>;
  // For follow candidates: the structural tags from `allLegalPlayActions`
  // (e.g., 'follow-led-suit', 'follow-trump'). Empty otherwise.
  readonly tags: ReadonlyArray<string>;
  // Action feature vector, fixed length.
  readonly encoding: number[];
  // Effective suit of `cards`; null for STOP.
  readonly effectiveSuit: EffectiveSuit | null;
};

export type LeadInProgress = {
  // Cards committed in earlier sub-actions of this lead.
  readonly chosen: ReadonlyArray<Card>;
  // Effective suit pinned by the first committed pattern. Null if no
  // pattern committed yet.
  readonly suit: EffectiveSuit | null;
  // Number of patterns committed so far.
  readonly patternCount: number;
};

export const EMPTY_LEAD: LeadInProgress = {
  chosen: [],
  suit: null,
  patternCount: 0,
};

// Cap on patterns per lead (Shan §4.5).
const MAX_PATTERNS_PER_LEAD = 3;

// Build the action feature vector for a (cards, kind) pair.
function encodeAction(
  cards: ReadonlyArray<Card>,
  kind: ActionKind,
  ctx: EncodingContext,
  trumpCtx: TrumpContext,
): number[] {
  const out = new Float32Array(ACTION_ENCODING_LEN);
  if (cards.length > 0) {
    encodeCardSet(cards, ctx, out.subarray(0, ENCODING_LEN));
  }
  out[ENCODING_LEN + SHAPE_INDEX[kind]] = 1;
  out[ENCODING_LEN + SHAPE_DIM] = cards.length / 10;
  if (cards.length > 0) {
    let allTrump = true;
    for (const c of cards) {
      if (effectiveSuit(c, trumpCtx) !== 'trump') {
        allTrump = false;
        break;
      }
    }
    out[ENCODING_LEN + SHAPE_DIM + 1] = allTrump ? 1 : 0;
  }
  return Array.from(out);
}

// Cards remaining in hand after removing `committed` (matched by full
// identity, not key — multi-deck pairs share key but differ by deckId).
function handMinus(
  hand: ReadonlyArray<Card>,
  committed: ReadonlyArray<Card>,
): Card[] {
  const out = [...hand];
  for (const c of committed) {
    const idx = out.findIndex((m) => cardEquals(m, c));
    if (idx >= 0) out.splice(idx, 1);
  }
  return out;
}

// Enumerate all single-pattern lead candidates in `hand`. If `suitLock`
// is provided, restrict to that effective suit (used after the first
// pattern is committed — throws live in one suit per rules §7.6).
function enumerateSinglePatternLeads(
  hand: ReadonlyArray<Card>,
  trumpCtx: TrumpContext,
  numDecks: number,
  suitLock: EffectiveSuit | null,
): Array<{ kind: ActionKind; cards: ReadonlyArray<Card>; suit: EffectiveSuit }> {
  const out: Array<{ kind: ActionKind; cards: ReadonlyArray<Card>; suit: EffectiveSuit }> = [];
  // Singles.
  for (const c of hand) {
    const es = effectiveSuit(c, trumpCtx);
    if (suitLock !== null && es !== suitLock) continue;
    out.push({ kind: 'lead-single', cards: [c], suit: es });
  }
  // Pairs.
  for (const p of findNKinds(hand, 2)) {
    const es = effectiveSuit(p.cards[0]!, trumpCtx);
    if (suitLock !== null && es !== suitLock) continue;
    out.push({ kind: 'lead-pair', cards: p.cards, suit: es });
  }
  // Tractors (every sub-run, not just maximal).
  for (const t of findTractors(hand, trumpCtx, numDecks)) {
    const es = effectiveSuit(t.cards[0]!, trumpCtx);
    if (suitLock !== null && es !== suitLock) continue;
    out.push({ kind: 'lead-tractor', cards: t.cards, suit: es });
  }
  return out;
}

// Dedup helper — two single-pattern candidates can be equivalent up to
// `deckId` (e.g., two copies of 5♠ as "single"; we want one candidate).
function dedupByKey(
  cands: Array<{ kind: ActionKind; cards: ReadonlyArray<Card>; suit: EffectiveSuit }>,
): typeof cands {
  const seen = new Set<string>();
  const out: typeof cands = [];
  for (const c of cands) {
    const sig = c.kind + '|' + c.cards.map(cardKey).sort().join(',');
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push(c);
  }
  return out;
}

// Top-level: enumerate candidates for the current decision point.
//
// - If trick is in flight (someone has led): enumerate follow candidates,
//   ignoring `lead`.
// - If trick has no lead: enumerate sub-action lead candidates given
//   the current `lead` state. The first sub-action gets all single-pattern
//   options; subsequent sub-actions also get a STOP option.
export function enumerateCandidates(
  state: GameState,
  seat: Seat,
  lead: LeadInProgress = EMPTY_LEAD,
): Candidate[] {
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null || hand.trumpSuit === null) {
    return [];
  }
  if (state.phase !== 'play') return [];
  if (nextToPlay(state) !== seat) return [];

  const encCtx: EncodingContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const trumpCtx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const trick = hand.currentTrick;

  // Follow branch — full plays via `allLegalPlayActions`.
  if (trick !== null && trick.plays.length > 0) {
    const acts = allLegalPlayActions(state, seat);
    return acts.map((a) => ({
      kind: 'follow' as const,
      cards: a.cards,
      tags: a.tags,
      encoding: encodeAction(a.cards, 'follow', encCtx, trumpCtx),
      effectiveSuit: effectiveSuit(a.cards[0]!, trumpCtx),
    }));
  }

  // Lead branch — sub-action enumeration.
  const fullHand = handFor(state, seat);
  const remaining = handMinus(fullHand, lead.chosen);

  // Cap reached: only STOP is legal.
  if (lead.patternCount >= MAX_PATTERNS_PER_LEAD) {
    return [stopCandidate(encCtx, trumpCtx)];
  }

  // Patterns available from the remaining hand, locked to lead.suit if any.
  const patternOptions = dedupByKey(
    enumerateSinglePatternLeads(remaining, trumpCtx, state.config.numDecks, lead.suit),
  );
  const out: Candidate[] = patternOptions.map((p) => ({
    kind: p.kind,
    cards: p.cards,
    tags: [],
    encoding: encodeAction(p.cards, p.kind, encCtx, trumpCtx),
    effectiveSuit: p.suit,
  }));

  // STOP option, only valid after at least one pattern committed.
  if (lead.patternCount >= 1) {
    out.push(stopCandidate(encCtx, trumpCtx));
  }

  return out;
}

function stopCandidate(
  encCtx: EncodingContext,
  trumpCtx: TrumpContext,
): Candidate {
  return {
    kind: 'lead-stop',
    cards: [],
    tags: [],
    encoding: encodeAction([], 'lead-stop', encCtx, trumpCtx),
    effectiveSuit: null,
  };
}

// Commit a chosen sub-action candidate into a new LeadInProgress.
// `lead-stop` returns null (signal to finalize).
export function applyLeadSubAction(
  lead: LeadInProgress,
  chosen: Candidate,
): LeadInProgress | null {
  if (chosen.kind === 'lead-stop') return null;
  return {
    chosen: [...lead.chosen, ...chosen.cards],
    suit: lead.suit ?? chosen.effectiveSuit,
    patternCount: lead.patternCount + 1,
  };
}
