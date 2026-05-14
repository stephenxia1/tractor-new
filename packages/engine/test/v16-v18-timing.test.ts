import { describe, it } from 'vitest';
import { applyEvent, nextToPlay } from '../src/reducer';
import { pickBotAction, type BotType } from '../src/bot';
import { createGame, defaultV1Config, type GameState } from '../src/state';
import type { GameEvent } from '../src/events';

function runOneHand(seed: number, target: BotType): { leads: number[]; follows: number[] } {
  const agents: ReadonlyArray<BotType> = [target, 'smart', target, 'smart'];
  const leads: number[] = [];
  const follows: number[] = [];
  let s: GameState = createGame(defaultV1Config());
  const start = applyEvent(s, { kind: 'start_hand', seed });
  if (!start.ok) throw new Error(start.error);
  s = start.state;

  for (let i = 0; i < 5000; i++) {
    if (s.phase === 'between_hands' || s.phase === 'game_over') break;
    if (s.phase === 'declare_lockout') {
      const r = applyEvent(s, { kind: 'release_lockout', seed: seed + 1 });
      if (!r.ok) throw new Error(r.error);
      s = r.state; continue;
    }
    if (s.phase === 'trick_review') {
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) throw new Error(r.error);
      s = r.state; continue;
    }
    if (s.phase === 'declare_open' || s.phase === 'kitty_replace') {
      let progressed = false;
      for (let seat = 0; seat < s.config.numPlayers; seat++) {
        const action = pickBotAction(s, seat, agents[seat]!);
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
          if (!r.ok) throw new Error(r.error);
          s = r.state;
        } else break;
      }
      continue;
    }
    if (s.phase === 'play') {
      const seat = nextToPlay(s);
      if (seat === null) break;
      const isTarget = seat === 0 || seat === 2;
      const isLeading = s.currentHand?.currentTrick === null;
      const t0 = performance.now();
      const action = pickBotAction(s, seat, agents[seat]!);
      const dt = performance.now() - t0;
      if (isTarget && action.kind === 'play_cards') {
        (isLeading ? leads : follows).push(dt);
      }
      if (action.kind !== 'play_cards') break;
      const r = applyEvent(s, { kind: 'play_cards', seat, cards: action.cards });
      if (!r.ok) throw new Error(`seat ${seat}: ${r.error}`);
      s = r.state;
    }
  }
  return { leads, follows };
}

function stats(samples: number[]) {
  if (samples.length === 0) return { n: 0, mean: 0, median: 0, p95: 0, max: 0 };
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = sorted.reduce((s, v) => s + v, 0) / sorted.length;
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const p95 = sorted[Math.floor(sorted.length * 0.95)]!;
  const max = sorted[sorted.length - 1]!;
  return { n: sorted.length, mean, median, p95, max };
}

describe('v16/v17/v18 per-move timing', () => {
  for (const target of ['smart-v16', 'smart-v17', 'smart-v18'] as const) {
    it(`${target} (2 hands)`, () => {
      const allLeads: number[] = [];
      const allFollows: number[] = [];
      for (let seed = 1; seed <= 2; seed++) {
        const { leads, follows } = runOneHand(seed, target);
        allLeads.push(...leads);
        allFollows.push(...follows);
      }
      const L = stats(allLeads);
      const F = stats(allFollows);
      console.log(
        `${target}  LEADS:   n=${L.n}  mean=${L.mean.toFixed(1)}ms  median=${L.median.toFixed(1)}ms  p95=${L.p95.toFixed(1)}ms  max=${L.max.toFixed(1)}ms`,
      );
      console.log(
        `${target}  FOLLOWS: n=${F.n}  mean=${F.mean.toFixed(1)}ms  median=${F.median.toFixed(1)}ms  p95=${F.p95.toFixed(1)}ms  max=${F.max.toFixed(1)}ms`,
      );
    });
  }
});
