// Smoke-test for allLegalPlayActions:
//  1. Plays a few hands using built-in bots, calling allLegalPlayActions
//     at every play decision.
//  2. Records the candidate count distribution.
//  3. Verifies every emitted candidate round-trips through applyEvent.
//  4. Compares against legalActions to confirm we're producing a
//     superset (or at least a wider set).

import { applyEvent } from '../../reducer';
import { runHand, type Agent } from '../../eval';
import { legalActions } from '../../legalActions';
import { nextToPlay } from '../../reducer';
import { pickBotAction, type BotAction } from '../../bot';
import type { GameState, Seat } from '../../state';
import { allLegalPlayActions } from './actions';

const NUM_HANDS = parseInt(process.env.NUM_HANDS ?? '5', 10);
const SEED = parseInt(process.env.SEED ?? '42', 10);

const stats = {
  decisions: 0,
  totalCandidates: 0,
  totalLegalActionsCandidates: 0,
  candidateCountHistogram: new Map<number, number>(),
  invalidCandidates: 0,
  illegalSamplesShown: 0,
  leadDecisions: 0,
  followDecisions: 0,
  // Tag tally
  tags: new Map<string, number>(),
};

function bumpHistogram(map: Map<number, number>, key: number): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function spyAgent(): (state: GameState, seat: Seat) => BotAction {
  // We delegate playing to 'lowball' but we WATCH every play decision
  // by calling allLegalPlayActions and recording diagnostics.
  return (state, seat) => {
    if (
      state.phase === 'play' &&
      nextToPlay(state) === seat &&
      state.currentHand?.trumpRank !== null
    ) {
      const enumerated = allLegalPlayActions(state, seat);
      const fromLegalActions = legalActions(state, seat).filter(
        (a) => a.kind === 'play_cards',
      );
      stats.decisions++;
      stats.totalCandidates += enumerated.length;
      stats.totalLegalActionsCandidates += fromLegalActions.length;
      bumpHistogram(stats.candidateCountHistogram, enumerated.length);
      const isLeading = state.currentHand?.currentTrick === null;
      if (isLeading) stats.leadDecisions++;
      else stats.followDecisions++;

      for (const a of enumerated) {
        for (const t of a.tags) {
          stats.tags.set(t, (stats.tags.get(t) ?? 0) + 1);
        }
      }

      // Round-trip check: every candidate must be applyable.
      for (const a of enumerated) {
        const r = applyEvent(state, {
          kind: 'play_cards',
          seat,
          cards: a.cards,
        });
        if (!r.ok) {
          stats.invalidCandidates++;
          if (stats.illegalSamplesShown < 5) {
            stats.illegalSamplesShown++;
            console.error(
              `INVALID CANDIDATE (decision ${stats.decisions}, leading=${isLeading}): ` +
              `tags=${a.tags.join(',')} error=${r.error}`,
            );
            console.error(`  cards=${JSON.stringify(a.cards)}`);
            console.error(`  hand=${JSON.stringify(state.currentHand?.hands[seat])}`);
            if (state.currentHand?.currentTrick) {
              console.error(`  lead=${JSON.stringify(state.currentHand.currentTrick.plays[0]?.cards)}`);
            }
          }
        }
      }
    }
    // Always play via lowball — we're just instrumenting.
    return pickBotAction(state, seat, 'lowball');
  };
}

function main(): void {
  console.log(`Smoke-testing allLegalPlayActions over ${NUM_HANDS} hands...`);
  const agents: Agent[] = [spyAgent(), spyAgent(), spyAgent(), spyAgent()];
  for (let i = 0; i < NUM_HANDS; i++) {
    runHand({ agents, seed: SEED + i });
  }

  console.log(`\nDecisions observed: ${stats.decisions}`);
  console.log(`  Leading: ${stats.leadDecisions}`);
  console.log(`  Following: ${stats.followDecisions}`);
  console.log(`Avg candidates/decision (allLegalPlayActions): ${
    (stats.totalCandidates / Math.max(1, stats.decisions)).toFixed(1)
  }`);
  console.log(`Avg candidates/decision (legalActions, baseline): ${
    (stats.totalLegalActionsCandidates / Math.max(1, stats.decisions)).toFixed(1)
  }`);
  console.log(`Widening factor: ${
    (stats.totalCandidates / Math.max(1, stats.totalLegalActionsCandidates)).toFixed(1)
  }x`);
  console.log(`Invalid candidates: ${stats.invalidCandidates}`);

  console.log(`\nCandidate-count histogram:`);
  const sortedKeys = [...stats.candidateCountHistogram.keys()].sort((a, b) => a - b);
  for (const k of sortedKeys) {
    const v = stats.candidateCountHistogram.get(k)!;
    const bar = '#'.repeat(Math.min(60, Math.round(v / Math.max(1, stats.decisions) * 200)));
    console.log(`  ${k.toString().padStart(4)}: ${v.toString().padStart(4)}  ${bar}`);
  }

  console.log(`\nTag distribution:`);
  const sortedTags = [...stats.tags.entries()].sort((a, b) => b[1] - a[1]);
  for (const [tag, count] of sortedTags) {
    console.log(`  ${tag.padEnd(20)} ${count}`);
  }
}

main();
