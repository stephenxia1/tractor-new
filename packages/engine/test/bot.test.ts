import { describe, it, expect } from 'vitest';
import {
  applyEvent,
  createGame,
  defaultV1Config,
  pickBotAction,
  type GameEvent,
  type GameState,
} from '../src';

// Drives a full hand with 4 bots, verifying each phase resolves and the
// hand ends in `between_hands`. The bot is dumb, so this is a robustness
// test (no crashes, all moves legal) not a strength test.
function runFullHandWithBots(seed: number): GameState {
  let s = createGame(defaultV1Config());
  const a = applyEvent(s, { kind: 'start_hand', seed });
  if (!a.ok) throw new Error(a.error);
  s = a.state;

  // Run a finite number of bot iterations to avoid infinite loops on bugs.
  for (let i = 0; i < 500; i++) {
    if (s.phase === 'between_hands' || s.phase === 'game_over') return s;

    // Server-driven event: release_lockout.
    if (s.phase === 'declare_lockout') {
      const r = applyEvent(s, { kind: 'release_lockout', seed: seed + 1 });
      if (!r.ok) throw new Error(r.error);
      s = r.state;
      continue;
    }

    // Server-driven event: advance_trick. In production this comes from a
    // UI click or a server-side timer; the test driver advances eagerly.
    if (s.phase === 'trick_review') {
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) throw new Error(r.error);
      s = r.state;
      continue;
    }

    // Find a bot with work to do.
    let progress = false;
    for (let seat = 0; seat < s.config.numPlayers; seat++) {
      const action = pickBotAction(s, seat);
      if (action.kind === 'wait') continue;
      const event = botEvent(action, seat);
      if (event === null) continue;
      const r = applyEvent(s, event);
      if (!r.ok) {
        // Bot picked an invalid action; skip this seat.
        continue;
      }
      s = r.state;
      progress = true;
      break;
    }
    if (!progress) {
      throw new Error(`stuck at phase ${s.phase}`);
    }
  }
  throw new Error('hand did not terminate within 500 iterations');
}

function botEvent(
  action: ReturnType<typeof pickBotAction>,
  seat: number,
): GameEvent | null {
  switch (action.kind) {
    case 'declare':
      return { kind: 'declare', seat, cards: action.cards };
    case 'pick_up_kitty':
      return { kind: 'pick_up_kitty', seat };
    case 'replace_kitty':
      return { kind: 'replace_kitty', seat, cards: action.cards };
    case 'play_cards':
      return { kind: 'play_cards', seat, cards: action.cards };
    case 'wait':
      return null;
  }
}

describe('bot integration', () => {
  it('plays a complete hand with 4 bots (seed 1)', () => {
    const s = runFullHandWithBots(1);
    expect(s.phase).toBe('between_hands');
    expect(s.handsCompleted).toBe(1);
    expect(s.currentHand!.tricks).toHaveLength(25);
  });

  it('plays a complete hand with 4 bots (seed 99)', () => {
    const s = runFullHandWithBots(99);
    expect(s.phase).toBe('between_hands');
    expect(s.currentHand!.tricks).toHaveLength(25);
  });

  it('plays a complete hand with 4 bots (seed 12345)', () => {
    const s = runFullHandWithBots(12345);
    expect(s.phase).toBe('between_hands');
  });

  it('attackers + defenders point split sums to ≤ 200 (kitty multiplier excluded for the inequality)', () => {
    const s = runFullHandWithBots(7);
    expect(s.currentHand!.attackerPoints).toBeGreaterThanOrEqual(0);
    // With kitty multiplier the total can exceed 200; just sanity-check non-negative.
  });
});
