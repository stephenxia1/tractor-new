# @tractor/engine — Bot Author API

This document is the contract another session (e.g., a bot or RL-training
pipeline) uses to drive the Tractor game engine. The engine is a pure,
synchronous TypeScript reducer with no network or DOM dependencies.

> All types live in `packages/engine/src/`. The public surface is
> re-exported from `packages/engine/src/index.ts`.

## 1. The mental model

Tractor is modeled as a finite-state reducer:

```
state = createGame(defaultV1Config())           // GameState
result = applyEvent(state, event)               // ApplyResult
if (result.ok) state = result.state
```

- **State** is immutable and JSON-serializable. Every event returns a new
  state (or an error + the unchanged state).
- **Events** are flat objects with a `kind` discriminator. Each event
  carries the `seat` it belongs to (no implicit "current player" on the
  reducer — callers pass it).
- **Phases** progress strictly forward:

  ```
  between_hands
    → start_hand → declare_lockout
    → declare → declare_open (after lockout)
    → pick_up_kitty → kitty_replace
    → replace_kitty → play
    → play_cards × N → trick_review
    → advance_trick → play (next trick) or finalize → between_hands / game_over
  ```

- **Card identity** includes a `deckId` (0 or 1 for v1's two decks), so
  two visually-identical cards are distinguishable.

## 2. Core types you need to know

| Type                | Where                       | Shape (abridged)                                                   |
| ------------------- | --------------------------- | ------------------------------------------------------------------ |
| `Card`              | `types.ts`                  | `{ kind: 'standard', suit, rank, deckId } \| { kind: 'joker', rank, deckId }` |
| `Seat`              | `state.ts`                  | `number` (0..numPlayers-1)                                         |
| `TeamId`            | `state.ts`                  | `string` (`'A'` or `'B'` in v1)                                    |
| `GameConfig`        | `state.ts`                  | `{ numPlayers, numDecks, kittySize, teamIds, seatToTeam, ... }`    |
| `GamePhase`         | `state.ts`                  | union of phase string literals                                     |
| `HandState`         | `state.ts`                  | per-hand: `hands`, `kitty`, `tricks`, `currentTrick`, `attackerPoints`, ... |
| `GameState`         | `state.ts`                  | top-level: `config`, `teamLevels`, `landlord`, `phase`, `currentHand`, ... |
| `GameEvent`         | `events.ts`                 | union of all event objects                                         |
| `ApplyResult`       | `reducer.ts`                | `{ ok: true, state, notice? } \| { ok: false, error, state }`     |
| `ClassifiedCombo`   | `classifier.ts`             | `{ kind: 'single' \| 'nkind' \| 'tractor', ... }`                  |
| `TrumpContext`      | `types.ts`                  | `{ trumpRank: NaturalRank, trumpSuit: Suit \| null }`              |

## 3. Driving a game

### 3.1 Create

```ts
import { createGame, defaultV1Config } from '@tractor/engine';

let state = createGame(defaultV1Config());
// state.phase === 'between_hands'
```

### 3.2 Apply an event

```ts
import { applyEvent } from '@tractor/engine';

const r = applyEvent(state, { kind: 'start_hand', seed: 12345 });
if (!r.ok) throw new Error(r.error);
state = r.state;
// state.phase === 'declare_lockout'; cards are dealt; hands accessible.
```

The reducer **never throws** for in-spec inputs. Validation failures show
up as `{ ok: false, error }` with the original state unchanged.

### 3.3 Event reference

```ts
type GameEvent =
  | { kind: 'start_hand'; seed: number }
  | { kind: 'declare'; seat: Seat; cards: ReadonlyArray<Card> }
  | { kind: 'release_lockout'; seed?: number } // server-driven; emitted after deal timer
  | { kind: 'pick_up_kitty'; seat: Seat }      // landlord only
  | { kind: 'replace_kitty'; seat: Seat; cards: ReadonlyArray<Card> } // 8 cards
  | { kind: 'play_cards'; seat: Seat; cards: ReadonlyArray<Card> }
  | { kind: 'advance_trick' };                 // after 4 plays in trick_review
```

### 3.4 Notices (forced lead)

If a leader's throw fails the safety check, the engine plays only the
smallest beatable component and returns `notice` on `ApplyResult`:

```ts
type ApplyNotice = {
  kind: 'forced_lead';
  leaderSeat: Seat;
  beaterSeat: Seat;
  playedCards: ReadonlyArray<Card>;    // what actually went into the trick
  returnedCards: ReadonlyArray<Card>;  // back to the leader's hand
};
```

## 4. Inspecting state for a bot

```ts
import {
  handFor,
  nextToPlay,
  teamOf,
  legalActions,
  legalEvents,
  classifyCombo,
  comboBeats,
  decomposeThrow,
  isThrowSafe,
} from '@tractor/engine';
```

- `handFor(state, seat) → ReadonlyArray<Card>` — that seat's cards.
- `nextToPlay(state) → Seat | null` — expected next seat during `play`.
- `teamOf(state, seat) → TeamId`.
- `state.currentHand.tricks` — completed tricks (winner + plays + points).
- `state.currentHand.currentTrick` — the in-flight trick, `null` between tricks.

## 5. Enumerating legal moves — the bot author's main entry point

```ts
const moves = legalActions(state, seat);
// e.g.
// [{ kind: 'declare', seat: 0, cards: [...], tier: 1 },
//  { kind: 'pick_up_kitty', seat: 0 }, ...]
```

`legalActions` returns a curated list of legal events for the seat at the
current phase. Pick one, call `applyEvent`.

`legalActions` covers:

- `declare_lockout` / `declare_open` — every legal declaration the seat
  can make (singles, pairs, joker pairs), plus `pick_up_kitty` if you're
  the landlord during `declare_open`.
- `kitty_replace` — landlord only. Returns one default burial
  (lowest 8 by trump ordering). Bots may construct their own
  burial event directly.
- `play` —
  - When **leading**: every single, every pair, every tractor in your
    hand, plus same-suit "throws" that pass the safety check (using
    full-information opponent hands — only suitable for self-play
    training, not partial-info inference).
  - When **following**: a small curated set (lowest-led-suit fill,
    highest-led-suit fill if it might beat the trick, plus pair matches
    for pair leads). Anything returned passes `validateFollow`.
- `trick_review` — `advance_trick`.

Returned actions are tagged so bots can reason about intent (e.g.
`'lead-pair'`, `'follow-trump'`, `'follow-discard'`).

If you want raw `GameEvent`s instead of the metadata-wrapped descriptions:

```ts
const events = legalEvents(state, seat); // ReadonlyArray<GameEvent>
```

> **Important:** `legalActions` is a starting point, not a guarantee of
> optimality or completeness. The full follow space (which cards to fill
> with after the led-suit obligation is met) is combinatorially large; the
> enumerator returns a sensible subset. To play arbitrary legal moves,
> construct events directly and validate via `applyEvent`. Anything that
> `applyEvent` returns `ok: true` for is legal.

## 6. Combo / structure helpers

```ts
import {
  classifyCombo,
  comboBeats,
  findNKinds,
  findTractors,
  validateFollow,
  validateFollowThrow,
  decomposeThrow,
  isThrowSafe,
  compareCards,
  compareCardsForDisplay,
  isTrump,
  effectiveSuit,
  trumpTier,
} from '@tractor/engine';
```

- `classifyCombo(cards, ctx, numDecks)` →
  `{ ok: true, combo: ClassifiedCombo } | { ok: false, reason }`.
- `comboBeats(candidate, target) → 1 | 0 | -1`.
- `validateFollow(lead, candidate, hand, ctx, n)` — checks pair / tractor
  follow obligations.
- `decomposeThrow(cards, ctx, n)` → throw components or rejection.
- `isThrowSafe(decomp, opponentHands, ctx, n)` — per rules §7.6. Note: a
  trump combo cannot block a non-trump throw (this engine's rule).
- `compareCards` orders cards by trump tier and rank. Use
  `compareCardsForDisplay` to break ties by suit (useful for sorting).

## 7. Scoring

```ts
import {
  pointsForCard,
  sumPoints,
  kittyMultiplier,
  computeHandResult,
  advanceLevel,
} from '@tractor/engine';
```

`computeHandResult(attackerPoints)` returns
`{ winner: 'defenders' | 'attackers', levelDelta: number }` per the
bracket table in rules.md §8.3. Levels max out at `'A'`. A defender win
*from* level `'A'` means the team wins the whole game (engine sets
`state.phase = 'game_over'`).

## 8. Self-play loop skeleton

```ts
import { createGame, defaultV1Config, applyEvent, legalActions, nextToPlay } from '@tractor/engine';

let state = createGame(defaultV1Config());
state = applyEvent(state, { kind: 'start_hand', seed: 1 }).state!;

// Drive declarations + kitty for all 4 bots until we're in `play`.
// (Server normally times out the lockout — in offline play, you can
//  send `release_lockout` yourself.)
state = applyEvent(state, { kind: 'release_lockout', seed: 2 }).state!;
// Now landlord may need to pick_up_kitty + replace_kitty before play.

while (state.phase !== 'between_hands' && state.phase !== 'game_over') {
  if (state.phase === 'trick_review') {
    state = applyEvent(state, { kind: 'advance_trick' }).state!;
    continue;
  }
  const seat = nextToPlay(state) ?? findCurrentActorSeat(state);
  const moves = legalActions(state, seat);
  const pick = policy.choose(moves, state, seat);  // your bot
  const r = applyEvent(state, pick);
  if (!r.ok) throw new Error(r.error);
  state = r.state;
}
// Inspect state.teamLevels, state.winner, state.handsCompleted.
```

## 9. Determinism

- All randomness flows through a `seed: number` argument on `start_hand`
  and (optionally) `release_lockout`. Same seed → same deal and same
  random landlord on kitty-flip fallback.
- The reducer otherwise contains no `Date.now()`, no I/O.
- Equality of two `GameState`s is structural; use `JSON.stringify` for
  cheap fingerprinting in tests.

## 9.5 Headless evaluation environment

For testing a candidate agent against a baseline (e.g. `'random'`),
import the eval driver. It plays games end-to-end with no UI, no
timers, and no network — synchronous reducer calls only.

```ts
import { runHand, runGame, runMatch, type Agent } from '@tractor/engine';
```

### 9.5.1 One hand

```ts
const { state, summary } = runHand({
  agents: ['random', 'lowball', 'random', 'lowball'],  // seats 0..3
  seed: 42,
});
console.log(summary);
// {
//   handNumber: 1,
//   landlord: 2,
//   trumpRank: '2',
//   attackerPoints: 75,
//   winner: 'defenders',
//   levelDelta: 1,
//   tricksPlayed: 25,
//   teamLevels: { A: '3', B: '2' },
// }
```

- `state` must be at `phase: 'between_hands'` (omit it for a fresh game).
- `seed` is the only randomness source — same seed reproduces the deal.
- An entry of `agents[seat]` may be either a built-in type string
  (`'lowball' | 'aggressive' | 'point-hoarder' | 'random'`) or a custom
  `(state, seat) => BotAction` function. Custom functions get full state
  access — appropriate for self-play training, not for partial-info
  inference.

### 9.5.2 A full game (hands until a team wins)

```ts
const { state, summary } = runGame({
  agents: ['my-agent', 'random', 'my-agent', 'random'],
  seed: 7,
  maxHands: 50,                  // safety cap; default 250
});
console.log(summary.winner, summary.handsPlayed, summary.finalLevels);
console.log(summary.hands);      // per-hand details
```

`summary.stoppedReason` is `'game_over'` (one team reached the win
condition) or `'max_hands_reached'`.

### 9.5.3 A match (many games)

```ts
const m = runMatch({
  agents: ['my-agent', 'random', 'my-agent', 'random'],
  numGames: 200,
  seed: 1,
  maxHandsPerGame: 50,
});
console.log(m.winsByTeam);          // { A: 137, B: 58 } — team A = my-agent
console.log(m.stalemates);          // games that hit the cap
console.log(m.avgHandsPerGame);     // useful comparative metric
console.log(m.avgAttackerPoints);   // attacker-points per hand average
```

### 9.5.4 Conventions for evaluation

- **Win rate**: with the default `defaultV1Config`, seats `0` & `2` are
  on team A and `1` & `3` are on team B. Put your candidate on team A
  and the baseline on team B (or vice versa) and read off
  `m.winsByTeam`.
- **Determinism**: every run is fully reproducible from `seed`. Save the
  seeds you used in your training reports.
- **Speed**: ~1000 hands/sec on a laptop. No async, no I/O.
- **Throughput tip**: if you only care about per-hand outcomes, prefer
  `runHand` in a loop with hand seeds you control — `runGame`/`runMatch`
  are convenience wrappers, not faster.

### 9.5.5 Plugging a custom agent in

A custom agent is a function:

```ts
function myAgent(state: GameState, seat: Seat): BotAction {
  // 1. Decide. You have full state access here — opponents' hands too.
  // 2. Return one of:
  //    { kind: 'play_cards', cards: [...] }
  //    { kind: 'declare',     cards: [...] }
  //    { kind: 'pick_up_kitty' }
  //    { kind: 'replace_kitty', cards: [...] }
  //    { kind: 'wait' }     // not my phase/turn
}
```

Anything you return is round-tripped through `applyEvent`; if it's
illegal the driver throws (so don't catch it — bugs surface
immediately). To stay legal during play, either consult `legalActions`
or read the rules helpers in §6.

Pass it inline as an `Agent`:

```ts
runMatch({
  agents: ['random', myAgent, 'random', myAgent],
  numGames: 500,
  seed: 1,
});
```

## 10. Existing baseline bot

`pickBotAction(state, seat) → BotAction` in `bot.ts` is a minimal greedy
heuristic. It's used by the WS server to fill seats; bot authors should
treat it as a baseline to beat, not as the production policy.

```ts
import { pickBotAction } from '@tractor/engine';
const action = pickBotAction(state, seat);
// { kind: 'play_cards' | 'declare' | 'replace_kitty' | 'pick_up_kitty' | 'wait', ... }
```

`'wait'` means this seat shouldn't act under the current phase / turn.

## 11. Cross-language usage

The engine is pure TypeScript with no native deps. Options for Python /
other languages:

- **Subprocess**: spawn `node` with a small JSON-line wrapper that reads
  `{state, event}` on stdin and writes `{ok, state | error}` on stdout.
- **HTTP**: wrap `applyEvent` in a tiny Express/uWS handler.
- **WASM**: not packaged today; ask the engine maintainer if needed.

State and events are plain JSON, so serializing across a process
boundary is straightforward.
