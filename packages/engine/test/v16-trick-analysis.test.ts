import { describe, it } from 'vitest';
import { applyEvent, nextToPlay } from '../src/reducer';
import { pickBotAction } from '../src/bot';
import { createGame, defaultV1Config, teamOf, type GameState, type Seat } from '../src/state';
import { effectiveSuit, isTrump, compareCards } from '../src/ordering';
import { pointsForCard, sumPoints } from '../src/scoring';
import { classifyCombo, comboBeats } from '../src/classifier';
import { findNKinds } from '../src/combos';
import type { GameEvent } from '../src/events';
import type { Card, TrumpContext } from '../src/types';

type TrickAnalysis = {
  hand: number;
  trick: number;
  leader: Seat;
  leaderTeam: 'A' | 'B';
  ledSuit: string;
  leadStruct: 'single' | 'pair' | 'tractor' | 'other';
  winnerTeam: 'A' | 'B';
  points: number;
  v16Seats: ReadonlyArray<{
    seat: Seat;
    wasLeader: boolean;
    cards: ReadonlyArray<string>;
    role: 'lead' | 'follow';
    handSizeBefore: number;
    hadHigherInLed: boolean;
    hadAtLeastOneTrump: boolean;
    teammateWonTrick: boolean;
  }>;
};

function cardLabel(c: Card): string {
  if (c.kind === 'joker') return c.rank === 'big' ? 'JB' : 'JS';
  return `${c.rank}${c.suit[0]!.toUpperCase()}`;
}

// Run a hand with both teams using smart-v16; record per-trick analysis
// from the perspective of every v16-controlled seat. With both teams as
// v16 this gives a balanced sample.
function runHandWithLogging(seed: number, agents: ReadonlyArray<'smart-v16' | 'smart-v9' | 'smart' | 'random'>): TrickAnalysis[] {
  let s: GameState = createGame(defaultV1Config());
  const start = applyEvent(s, { kind: 'start_hand', seed });
  if (!start.ok) throw new Error(start.error);
  s = start.state;

  // Drive declare and kitty.
  for (let i = 0; i < 500; i++) {
    if (s.phase === 'between_hands' || s.phase === 'game_over' || s.phase === 'play') break;
    if (s.phase === 'declare_lockout') {
      const r = applyEvent(s, { kind: 'release_lockout', seed: seed + 1 });
      if (!r.ok) throw new Error(r.error);
      s = r.state;
      continue;
    }
    if (s.phase === 'trick_review') {
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) throw new Error(r.error);
      s = r.state;
      continue;
    }
    if (s.phase === 'declare_open' || s.phase === 'kitty_replace') {
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
        s = r.state;
        progressed = true;
        break;
      }
      if (!progressed) {
        if (s.phase === 'declare_open') {
          const land = s.currentHand!.landlord!;
          const r = applyEvent(s, { kind: 'pick_up_kitty', seat: land });
          if (!r.ok) throw new Error(r.error);
          s = r.state;
        } else break;
      }
      continue;
    }
  }

  // Trick play.
  const analyses: TrickAnalysis[] = [];
  for (let i = 0; i < 5000; i++) {
    if (s.phase !== 'play' && s.phase !== 'trick_review') break;
    if (s.phase === 'trick_review') {
      // Analyze the just-completed trick.
      const lastTrick = s.currentHand?.currentTrick;
      if (lastTrick && lastTrick.plays.length === s.config.numPlayers) {
        const ctx: TrumpContext = {
          trumpRank: s.currentHand!.trumpRank!,
          trumpSuit: s.currentHand!.trumpSuit,
        };
        const n = s.config.numDecks;
        const lead = lastTrick.plays[0]!;
        const ledSuit = effectiveSuit(lead.cards[0]!, ctx);
        const leadCls = classifyCombo(lead.cards, ctx, n);
        const leadStruct: 'single' | 'pair' | 'tractor' | 'other' =
          !leadCls.ok ? 'other' :
          leadCls.combo.kind === 'single' ? 'single' :
          leadCls.combo.kind === 'nkind' && leadCls.combo.n === 2 ? 'pair' :
          leadCls.combo.kind === 'tractor' ? 'tractor' : 'other';

        // Resolve winner.
        let winner: Seat = lead.seat;
        let bestCombo = leadCls.ok ? leadCls.combo : null;
        for (let j = 1; j < lastTrick.plays.length; j++) {
          const p = lastTrick.plays[j]!;
          const cls = classifyCombo(p.cards, ctx, n);
          if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
            winner = p.seat;
            bestCombo = cls.combo;
          }
        }
        const points = sumPoints(lastTrick.plays.flatMap(p => [...p.cards]));
        const leaderTeam = teamOf(s, lead.seat);
        const winnerTeam = teamOf(s, winner);

        // For each v16-controlled seat, record what they had/played.
        const v16Seats = lastTrick.plays
          .filter(p => agents[p.seat] === 'smart-v16')
          .map(p => {
            const role = p.seat === lead.seat ? 'lead' : 'follow';
            // Cards in hand BEFORE this trick = current hand + cards played by this seat in this trick.
            const handAfter = s.currentHand!.hands[p.seat] ?? [];
            const handBefore = [...handAfter, ...p.cards];
            // Did they have a higher-rank in led suit (non-trump)?
            const ledRankIdx = (() => {
              if (leadCls.ok && leadCls.combo.kind === 'single' && leadCls.combo.effectiveSuit !== 'trump') {
                const c = lead.cards[0]!;
                if (c.kind === 'standard') return c.rank;
              }
              return null;
            })();
            let hadHigherInLed = false;
            if (ledRankIdx !== null && lead.cards[0]!.kind === 'standard') {
              const ledCard = lead.cards[0]! as Extract<Card, { kind: 'standard' }>;
              for (const c of handBefore) {
                if (c.kind !== 'standard') continue;
                if (c.suit !== ledCard.suit) continue;
                if (compareCards(c, ledCard, ctx) > 0) { hadHigherInLed = true; break; }
              }
            }
            const hadAtLeastOneTrump = handBefore.some(c => isTrump(c, ctx));
            const teammateWonTrick = teamOf(s, winner) === teamOf(s, p.seat);
            return {
              seat: p.seat,
              wasLeader: p.seat === lead.seat,
              cards: p.cards.map(cardLabel),
              role,
              handSizeBefore: handBefore.length,
              hadHigherInLed,
              hadAtLeastOneTrump,
              teammateWonTrick,
            } as const;
          });

        analyses.push({
          hand: s.currentHand!.handNumber,
          trick: s.currentHand!.tricks.length, // since currentTrick still active
          leader: lead.seat,
          leaderTeam: leaderTeam as 'A' | 'B',
          ledSuit: ledSuit as string,
          leadStruct,
          winnerTeam: winnerTeam as 'A' | 'B',
          points,
          v16Seats,
        });
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
    if (!r.ok) throw new Error(`seat ${seat}: ${r.error}`);
    s = r.state;
  }
  return analyses;
}

describe('v16 trick analysis', () => {
  it('aggregates stats across 5 v16-vs-smart games', () => {
    // Mixed setup: v16 in seats 0/2, smart (no-search) in seats 1/3. So
    // v16 records its decisions against a known weaker baseline; tricks
    // where v16 still loses points are informative.
    const agents = ['smart-v16', 'smart', 'smart-v16', 'smart'] as const;
    const allAnalyses: TrickAnalysis[] = [];
    for (let seed = 1; seed <= 5; seed++) {
      const a = runHandWithLogging(seed, agents);
      allAnalyses.push(...a);
    }

    // Aggregate: for each v16 follow on a non-trump single lead where
    // opponent won the trick with points ≥ 10, did v16 have a higher
    // card in led suit (i.e., missed a beat)?
    let missedBeats = 0;
    let missedBeatPoints = 0;
    let totalLoss = 0;
    let totalLossPoints = 0;
    let totalWinPoints = 0;
    let trickCount = 0;

    for (const t of allAnalyses) {
      trickCount++;
      for (const seat of t.v16Seats) {
        const myTeamWon = seat.teammateWonTrick || seat.wasLeader === false
          ? t.winnerTeam === (seat.seat % 2 === 0 ? 'A' : 'B')
          : t.winnerTeam === (seat.seat % 2 === 0 ? 'A' : 'B');
        // simplified: v16 seats are 0,2 = team A.
        const v16WonTrick = t.winnerTeam === 'A';
        if (v16WonTrick) {
          totalWinPoints += t.points;
        } else {
          totalLoss++;
          totalLossPoints += t.points;
          if (seat.role === 'follow' && t.leadStruct === 'single' && seat.hadHigherInLed && t.points >= 10) {
            missedBeats++;
            missedBeatPoints += t.points;
          }
        }
      }
    }

    console.log(`tricks logged: ${trickCount}`);
    console.log(`v16 team won tricks worth: ${totalWinPoints} points`);
    console.log(`v16 team lost tricks worth: ${totalLossPoints} points`);
    console.log(`v16 follow-position missed-beat tricks (single lead, opp won 10+ pt, v16 had higher): ${missedBeats} (${missedBeatPoints} pts lost)`);

    // Also: count tricks where led suit was non-trump, opp followed without trumping, and v16 didn't beat.
    let nonTrumpLossesWhereV16Held: number[] = [];
    let leadStructCounts: Record<string, number> = {};
    let suitCounts: Record<string, number> = {};
    for (const t of allAnalyses) {
      leadStructCounts[t.leadStruct] = (leadStructCounts[t.leadStruct] ?? 0) + 1;
      suitCounts[t.ledSuit] = (suitCounts[t.ledSuit] ?? 0) + 1;
    }
    console.log(`lead structure distribution:`, leadStructCounts);
    console.log(`led suit distribution:`, suitCounts);
  });
});
