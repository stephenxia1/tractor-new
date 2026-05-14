import { describe, it } from 'vitest';
import { applyEvent, nextToPlay } from '../src/reducer';
import { pickBotAction } from '../src/bot';
import { createGame, defaultV1Config, teamOf, type GameState, type Seat } from '../src/state';
import { effectiveSuit, isTrump } from '../src/ordering';
import { sumPoints } from '../src/scoring';
import { classifyCombo, comboBeats } from '../src/classifier';
import type { GameEvent } from '../src/events';
import type { Card, TrumpContext } from '../src/types';

// Look for: tricks where v16 followed in led suit (didn't trump), the opp
// team won the trick worth ≥10 points, and v16 was void in led suit at
// time of play but had at least one trump.
function runAndAnalyze(seed: number) {
  const agents = ['smart-v16', 'smart', 'smart-v16', 'smart'] as const;
  let s: GameState = createGame(defaultV1Config());
  const start = applyEvent(s, { kind: 'start_hand', seed });
  if (!start.ok) throw new Error(start.error);
  s = start.state;

  // Drive declare/kitty.
  for (let i = 0; i < 500; i++) {
    if (s.phase === 'between_hands' || s.phase === 'game_over' || s.phase === 'play') break;
    if (s.phase === 'declare_lockout') {
      const r = applyEvent(s, { kind: 'release_lockout', seed: seed + 1 });
      if (!r.ok) break;
      s = r.state; continue;
    }
    if (s.phase === 'trick_review') {
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) break;
      s = r.state; continue;
    }
    let progressed = false;
    for (let seat = 0; seat < s.config.numPlayers; seat++) {
      const action = pickBotAction(s, seat, agents[seat]);
      if (action.kind === 'wait') continue;
      const ev: GameEvent | null =
        action.kind === 'declare' ? { kind: 'declare', seat, cards: action.cards } :
        action.kind === 'pick_up_kitty' ? { kind: 'pick_up_kitty', seat } :
        action.kind === 'replace_kitty' ? { kind: 'replace_kitty', seat, cards: action.cards } :
        null;
      if (ev === null) continue;
      const r = applyEvent(s, ev);
      if (!r.ok) continue;
      s = r.state; progressed = true; break;
    }
    if (!progressed) {
      if (s.phase === 'declare_open') {
        const land = s.currentHand!.landlord!;
        const r = applyEvent(s, { kind: 'pick_up_kitty', seat: land });
        if (!r.ok) break;
        s = r.state;
      } else break;
    }
  }

  let missedTrumpWins = 0;  // v16 had trump but didn't use it, opp won trick with points
  let missedTrumpPoints = 0;
  let lostHighPointTricks = 0;
  let lostHighPointPoints = 0;
  let totalPointsLost = 0;
  let totalPointsWon = 0;
  let trickCount = 0;

  for (let i = 0; i < 5000; i++) {
    if (s.phase !== 'play' && s.phase !== 'trick_review') break;
    if (s.phase === 'trick_review') {
      const trick = s.currentHand?.currentTrick;
      if (trick && trick.plays.length === s.config.numPlayers) {
        trickCount++;
        const ctx: TrumpContext = {
          trumpRank: s.currentHand!.trumpRank!,
          trumpSuit: s.currentHand!.trumpSuit,
        };
        const n = s.config.numDecks;
        const lead = trick.plays[0]!;
        const ledSuit = effectiveSuit(lead.cards[0]!, ctx);
        const leadCls = classifyCombo(lead.cards, ctx, n);

        // Resolve winner.
        let winner: Seat = lead.seat;
        let bestCombo = leadCls.ok ? leadCls.combo : null;
        for (let j = 1; j < trick.plays.length; j++) {
          const p = trick.plays[j]!;
          const cls = classifyCombo(p.cards, ctx, n);
          if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
            winner = p.seat;
            bestCombo = cls.combo;
          }
        }
        const points = sumPoints(trick.plays.flatMap(p => [...p.cards]));
        const v16Team = 'A';  // seats 0,2
        const v16Won = teamOf(s, winner) === v16Team;
        if (v16Won) totalPointsWon += points;
        else totalPointsLost += points;

        if (!v16Won && points >= 10) {
          lostHighPointTricks++;
          lostHighPointPoints += points;

          // For each v16 seat that FOLLOWED non-trump-trumping, check if
          // they had a trump and a led-suit card (i.e., they had to
          // follow led, but did the lead suit they followed cost a chance
          // to trump? Only relevant if they were void in led suit and
          // still didn't trump).
          for (let j = 1; j < trick.plays.length; j++) {
            const p = trick.plays[j]!;
            if (p.seat % 2 !== 0) continue;  // only v16 seats (0, 2)
            const handAfter = s.currentHand!.hands[p.seat] ?? [];
            const handBefore = [...handAfter, ...p.cards];
            const ledSuitInBefore = handBefore.filter(c => effectiveSuit(c, ctx) === ledSuit).length;
            const trumpsInBefore = handBefore.filter(c => isTrump(c, ctx)).length;
            const playedInLed = p.cards.filter(c => effectiveSuit(c, ctx) === ledSuit).length;
            // Void at play time = no led-suit cards in handBefore. (Actually if had any, would have had to follow.)
            // Check: was the seat void in led suit AND had trumps AND didn't play trump?
            const wasVoid = ledSuitInBefore === 0;
            const playedTrump = p.cards.some(c => isTrump(c, ctx));
            if (wasVoid && trumpsInBefore > 0 && !playedTrump) {
              missedTrumpWins++;
              missedTrumpPoints += points;
            }
          }
        }
      }
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) break;
      s = r.state;
      continue;
    }
    const seat = nextToPlay(s);
    if (seat === null) break;
    const action = pickBotAction(s, seat, agents[seat]);
    if (action.kind !== 'play_cards') break;
    const r = applyEvent(s, { kind: 'play_cards', seat, cards: action.cards });
    if (!r.ok) break;
    s = r.state;
  }

  return { trickCount, totalPointsWon, totalPointsLost, lostHighPointTricks, lostHighPointPoints, missedTrumpWins, missedTrumpPoints };
}

describe('v16 trump usage analysis', () => {
  it('aggregates "missed trump" stats across games', () => {
    let trickCount = 0, won = 0, lost = 0, hi = 0, hiPts = 0, missed = 0, missedPts = 0;
    for (let seed = 1; seed <= 5; seed++) {
      const r = runAndAnalyze(seed);
      trickCount += r.trickCount;
      won += r.totalPointsWon;
      lost += r.totalPointsLost;
      hi += r.lostHighPointTricks;
      hiPts += r.lostHighPointPoints;
      missed += r.missedTrumpWins;
      missedPts += r.missedTrumpPoints;
    }
    console.log(`Total tricks: ${trickCount}`);
    console.log(`v16 won: ${won} pts`);
    console.log(`v16 lost: ${lost} pts`);
    console.log(`High-point losses (≥10 pts): ${hi} tricks, ${hiPts} pts lost`);
    console.log(`Missed-trump opportunities (v16 void in led, had trumps, didn't trump, opp won 10+): ${missed} (${missedPts} pts lost)`);
  });
});
