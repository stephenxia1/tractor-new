# Adding a new bot/agent

This guide walks through everything you need to touch to plug a new
playing strategy into the engine, the server, and the UI's per-seat
"Add bot" dropdown. The whole change is small — usually one new function
in `bot.ts` plus a couple of one-line edits.

> All file paths below are relative to the repo root.

## 1. What an agent is

An agent is a pure function

```ts
(state: GameState, seat: Seat) => BotAction
```

that picks a single legal action for `seat` at the current `state`. The
`BotAction` union is defined in `packages/engine/src/bot.ts`:

```ts
type BotAction =
  | { kind: 'declare';         cards: ReadonlyArray<Card> }
  | { kind: 'pick_up_kitty' }
  | { kind: 'replace_kitty';   cards: ReadonlyArray<Card> }
  | { kind: 'play_cards';      cards: ReadonlyArray<Card> }
  | { kind: 'wait' };
```

`'wait'` means *"not this seat's turn / not my phase"* — return it when
you have nothing to do.

The engine guarantees:

- `state` is immutable; you may safely sort / filter / clone its fields.
- Every helper exported from `@tractor/engine` is pure.
- Whatever your function returns will be passed back through
  `applyEvent` — illegal moves will be rejected with an error and the
  game will stall on your seat. See `API.md` for what makes a move legal.

A complete reference for state shape, events, and helper functions lives
in [`API.md`](./API.md). Read that first if you haven't already.

## 2. The shared pre-play boilerplate

All four built-in agents share the same logic for the non-play phases
(declarations, taking the kitty, burying 8 cards) via the
`defaultPrePlay` helper. Your new agent should reuse it too — agents
differ in **how to play cards**, not in declarations.

The flow inside `pickBotAction` is:

```ts
const pre = defaultPrePlay(state, seat);
if (pre !== null) return pre;          // declare / take kitty / bury / wait
// ...we're in the play phase, and it's our turn.
return myPlayStrategy(...);
```

If `defaultPrePlay` returns `null`, the phase is `play` and it is our
turn. Your strategy only needs to decide what to play.

## 3. Adding a new strategy — the recipe

### 3.1 Open `packages/engine/src/bot.ts`

Three changes here, in order.

**a) Extend the `BotType` union and the runtime list.**

```ts
export type BotType =
  | 'lowball'
  | 'aggressive'
  | 'point-hoarder'
  | 'random'
  | 'my-new-bot';   // ← add here

export const BOT_TYPES: ReadonlyArray<BotType> = [
  'lowball',
  'aggressive',
  'point-hoarder',
  'random',
  'my-new-bot',     // ← and here
];
```

**b) Add a human-readable label.**

```ts
export function botTypeLabel(t: BotType): string {
  switch (t) {
    // ...existing cases
    case 'my-new-bot': return 'My new bot';
  }
}
```

The compiler will force you to handle the new case — if you forget,
`tsc` fails the build.

**c) Implement and dispatch.**

```ts
function pickMyNewBot(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  // your strategy here — see §4 below for helpers
}

export function pickBotAction(
  state: GameState,
  seat: Seat,
  type: BotType = 'lowball',
): BotAction {
  const pre = defaultPrePlay(state, seat);
  if (pre !== null) return pre;

  const hand = state.currentHand!;
  const myCards = hand.hands[seat] ?? [];
  const ctx: TrumpContext = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  switch (type) {
    case 'lowball':       return pickLowball(myCards, trick, ctx, n);
    case 'aggressive':    return pickAggressive(myCards, trick, ctx, n);
    case 'point-hoarder': return pickPointHoarder(state, seat, myCards, trick, ctx, n);
    case 'random':        return pickRandom(myCards, trick, ctx, n);
    case 'my-new-bot':    return pickMyNewBot(myCards, trick, ctx, n);   // ← add
  }
}
```

That's it for the engine. No other engine file needs to change — the
type union changes propagate everywhere it's used.

### 3.2 Verify the build

```
npm run -ws --if-present typecheck
npm test --workspace @tractor/engine
```

The 134 existing tests cover the shared rules; your new agent will be
exercised by the WS server / UI as soon as someone picks it.

### 3.3 (Nothing else needed)

- The **WS server** reads the bot's type from `room.seating[seat].botType`
  and passes it straight to `pickBotAction`, so it picks up the new
  variant automatically.
- The **client UI** populates the per-seat dropdown from
  `BOT_TYPES` / `botTypeLabel` — your new strategy appears in the
  dropdown without any client edits.
- The **protocol** stores `botType` as a plain string; no schema bump.

## 4. Helpers you'll want

All exported from `@tractor/engine`.

| Helper                 | What it gives you                                                          |
| ---------------------- | -------------------------------------------------------------------------- |
| `compareCards(a,b,ctx)`| Total trump-aware order; sort your hand low → high                          |
| `effectiveSuit(c,ctx)` | `'spades' \| 'hearts' \| 'diamonds' \| 'clubs' \| 'trump'`                  |
| `isTrump(c,ctx)`       | Boolean                                                                    |
| `findNKinds(hand, n)`  | All n-of-a-kinds (e.g., pairs) in `hand`                                   |
| `findTractors(h,ctx,n)`| All tractors                                                               |
| `classifyCombo(cards,ctx,n)` | `{ok, combo}` — is this a single/pair/tractor?                       |
| `comboBeats(a, b)`     | `1`/`0`/`-1` — same-structure comparison                                   |
| `validateFollow(...)`  | Confirms a candidate satisfies follow-suit + pair/tractor obligations      |
| `decomposeThrow(...)`  | Tries to decompose a multi-combo lead into components                      |
| `isThrowSafe(...)`     | §7.6 safety check                                                          |
| `pointsForCard(c)`     | `5`/`10`/`0` (5s, 10s+Ks worth 10, everything else 0)                      |
| `teamOf(state, seat)`  | Team id ('A'/'B')                                                          |
| `nextToPlay(state)`    | Seat the engine expects to play next during `play`                         |

For richer enumeration of legal moves, `legalActions(state, seat)` /
`legalEvents(state, seat)` give a curated list of candidate events —
useful as a starting set for search-based agents.

## 5. Strategy sketches you can crib from

The four built-ins live in `bot.ts`. Each is short — 20-40 lines —
and they all share `fillFollow`, a helper that builds a legal follow
given a preference (`'low'` / `'high'` / `'point-dump'`). Look at
`pickAggressive` for the simplest non-trivial example; look at
`pickPointHoarder` for an example that reads the in-flight trick to
decide a per-card preference.

For a stronger agent, two patterns work well:

- **Heuristic with multiple knobs**: extend `fillFollow` with a new
  preference mode (e.g. `'save-trumps'`) and call it with the right
  preference per situation.
- **Search-based**: enumerate via `legalActions`, simulate each move by
  calling `applyEvent`, score the resulting state, pick the best.
  `applyEvent` is pure, so simulation is free of side effects.

## 6. Testing your agent

The engine ships an integration test that runs a whole hand with 4 bots:
`packages/engine/test/bot.test.ts`. To add a smoke test for your new
strategy, copy one of the existing seeds and swap in your type:

```ts
const state = runFullHandWithBots(1, 'my-new-bot');   // pass through to pickBotAction
expect(state.phase).toBe('between_hands');
```

You will need a small edit in `runFullHandWithBots` to take a `BotType`
parameter and forward it to `pickBotAction` (currently it uses the
default). That's the only test infrastructure change.

## 7. Off-process agents (Python, etc.)

If your agent lives in another language, the simplest pattern is:

1. Implement a thin Node wrapper that reads JSON `{state, seat}` on
   stdin, calls your remote service, and writes back the chosen
   `BotAction` (or a `GameEvent`).
2. Add a new built-in strategy (say `'remote'`) whose body in
   `pickBotAction` makes a synchronous call to the wrapper (or returns
   `'wait'` while an async fetch is in flight, depending on your
   server's tolerance).

State and events are plain JSON, so the boundary cost is just
serialization. See `API.md` §11 for more.

## 8. Checklist (copy / paste when shipping a new agent)

- [ ] New entry in `BotType` union
- [ ] New entry in `BOT_TYPES` array
- [ ] New case in `botTypeLabel`
- [ ] Implementation function
- [ ] New case in the `switch` inside `pickBotAction`
- [ ] `npm run -ws --if-present typecheck` is clean
- [ ] `npm test --workspace @tractor/engine` is green
- [ ] (Optional) Integration test or self-play seed in `bot.test.ts`
- [ ] Smoke-test it in the room UI: select the strategy from the
      dropdown, click "Add bot" four times, watch a full hand play out.
