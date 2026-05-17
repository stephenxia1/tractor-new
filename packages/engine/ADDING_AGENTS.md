# Adding a new bot/agent

This guide walks through everything you need to touch to plug a new
playing strategy into the engine, the server, and the UI's per-seat
"Add bot" dropdown. The whole change is small — usually one or two new
functions in `bot.ts` plus a few one-line edits.

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

## 2. Two patterns of agent

`bot.ts` ships two coexisting patterns. Pick whichever fits the agent
you're writing — they interoperate cleanly via `pickBotAction`.

### 2.a "Simple" agents — `(myCards, trick, ctx, n) => BotAction`

For agents that only care about *what to play* in the play phase and are
happy to inherit standard declaration / kitty / wait logic. Examples:
`pickLowball`, `pickAggressive`, `pickPointHoarder`, `pickRandom`.

Simple agents are dispatched **after** `defaultPrePlay`, which runs the
shared boilerplate (declare with rank-R, take kitty as landlord, bury
8 lowest non-points). If `defaultPrePlay` returns `null`, the phase is
`play` and it is the seat's turn; the dispatcher then calls the simple
function with `(myCards, trick, ctx, n)`.

### 2.b "Smart" agents — `(state, seat) => BotAction`

For agents that want richer control: custom declarations, custom kitty
burying, search across multiple phases, etc. Examples: `pickSmart`,
`pickSmartV2`, …, `pickSmartV34`. These functions handle phase routing
themselves — they call `smartDeclare` / `smartBury` / the appropriate
play-phase search depending on `state.phase`.

The naming convention is `pickSmart`/`pickSmartV<N>` for the dispatch
function; the `smartV<N>Search` and `scoreCandidateV<N>` helpers hold
the actual logic. You don't have to follow the naming, but it makes
diffs against the existing lineage easier.

## 3. Adding a new strategy — the recipe

### 3.1 Open `packages/engine/src/bot.ts`

Five changes here:

**a) Extend the `BotType` union.**

```ts
export type BotType =
  | 'lowball'
  | 'aggressive'
  | 'point-hoarder'
  | 'random'
  | 'smart-v0' | 'smart' | 'smart-v2' | /* … many smart-v* … */
  | 'smart-v34'
  | 'rl-mlp'
  | 'my-new-bot';   // ← add here
```

**b) Add to the runtime list `BOT_TYPES`.** Every value in `BotType` must
appear in this array. Used by tests and the WS protocol.

```ts
export const BOT_TYPES: ReadonlyArray<BotType> = [
  /* …existing entries… */
  'my-new-bot',
];
```

**c) Decide whether to surface the agent in the lobby UI.**
`VISIBLE_BOT_TYPES` is the user-facing subset — only agents listed here
show up in the per-seat "Add bot" dropdown. Currently it's a small
curated set (random + the latest peak smart variants). Adding to
`BOT_TYPES` alone keeps the agent reachable over the WS protocol
(handy for benchmarks and tournaments) without cluttering the dropdown.

```ts
export const VISIBLE_BOT_TYPES: ReadonlyArray<BotType> = [
  'random',
  'smart-v18',
  'smart-v34',
  // 'my-new-bot',  // ← add only when the agent is ready for users
];
```

**d) Add a human-readable label.**

```ts
export function botTypeLabel(t: BotType): string {
  switch (t) {
    /* …existing cases… */
    case 'my-new-bot': return 'My new bot';
  }
}
```

The compiler will force you to handle the new case — if you forget,
`tsc` fails the build.

**e) Implement and dispatch.** `pickBotAction` is laid out as:

```ts
export function pickBotAction(state, seat, type = 'lowball'): BotAction {
  // (1) Mid-deal declaration phase (real games with incrementalDeal: true).
  //     Smart bots get a partial-hand declaration strategy; everyone
  //     else waits until dealing finishes.
  if (state.phase === 'dealing') { … }

  // (2) Smart-bot dispatch — each smart variant has its own
  //     `pickSmartV<N>(state, seat)` and handles all phases itself.
  if (type === 'smart' || type === 'smart-v0') return pickSmart(state, seat);
  if (type === 'smart-v2')  return pickSmartV2(state, seat);
  /* …each smart variant in turn… */
  if (type === 'smart-v34') return pickSmartV34(state, seat);
  if (type === 'rl-mlp')    return rlMlpAgent()(state, seat);

  // (3) Simple-bot fallback — defaultPrePlay handles declare/kitty/wait;
  //     the play phase is dispatched via a switch.
  const pre = defaultPrePlay(state, seat);
  if (pre !== null) return pre;
  const hand = state.currentHand!;
  const myCards = hand.hands[seat] ?? [];
  const ctx = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;
  switch (type) {
    case 'lowball':       return pickLowball(myCards, trick, ctx, n);
    case 'aggressive':    return pickAggressive(myCards, trick, ctx, n);
    case 'point-hoarder': return pickPointHoarder(state, seat, myCards, trick, ctx, n);
    case 'random':        return pickRandom(myCards, trick, ctx, n);
  }
}
```

For a **simple agent**, write a function with the shape
`(myCards, trick, ctx, n) => BotAction` and add a `case` to the final
switch.

For a **smart agent**, write a function with the shape
`(state, seat) => BotAction` and add an `if (type === 'my-new-bot')
return pickMyNewBot(state, seat);` in section (2). Most smart agents
clone an existing `pickSmartV<N>` and change one piece (a scoring knob,
candidate generator, opponent model, etc.).

### 3.2 Verify the build

```
npm run -ws --if-present typecheck
npm test --workspace @tractor/engine
```

Existing engine tests cover the shared rules; your new agent will be
exercised by the WS server / UI as soon as someone picks it.

### 3.3 What you don't need to touch

- The **WS server** reads `room.seating[seat].botType` and passes it
  straight to `pickBotAction`, so it picks up the new variant
  automatically.
- The **client UI** populates the per-seat dropdown from
  `VISIBLE_BOT_TYPES` / `botTypeLabel` — so adding to
  `VISIBLE_BOT_TYPES` is all you need for the dropdown.
- The **protocol** stores `botType` as a plain string; no schema bump.

## 4. The 'dealing' phase — partial-hand declarations

When `GameConfig.incrementalDeal` is `true` (the WS server's default),
`start_hand` produces a new phase `'dealing'`. The server emits
`reveal_card` events round-robin at `DEAL_INTERVAL_MS`, and a player
sees only the prefix of their hand of length `cardsDealtBySeat[seat]`.
A `declare` event is accepted during `'dealing'` but only for cards in
that revealed prefix. After `finish_dealing`, the phase moves to
`declare_lockout` and everything proceeds as before.

Implications for new agents:

- Default config in `eval.ts` / benchmarks uses `incrementalDeal: false`
  (legacy fast-deal). `start_hand` goes straight to `declare_lockout`
  with all cards revealed, so simple-agent benchmarks are unaffected.
- In live games, your agent may be called with `state.phase === 'dealing'`.
  If you don't want to think about it, just return `{ kind: 'wait' }` —
  declarations will then happen after dealing completes, same as
  pre-feature behavior. `pickBotAction` does this automatically for
  non-smart bots.
- Smart bots route to `pickDealingPhaseDeclare(state, seat)`, which
  reads `hand.cardsDealtBySeat[seat]` to compute the revealed slice and
  decides when to declare. Strong shapes (joker pair, suit pair of
  rank-R) declare immediately on reveal; singles wait until ≥ half the
  hand is dealt (or an opponent has declared) before committing.

## 5. Helpers you'll want

All exported from `@tractor/engine`.

| Helper                       | What it gives you                                                |
| ---------------------------- | ---------------------------------------------------------------- |
| `compareCards(a,b,ctx)`      | Total trump-aware order; sort your hand low → high               |
| `effectiveSuit(c,ctx)`       | `'spades' \| 'hearts' \| 'diamonds' \| 'clubs' \| 'trump'`       |
| `isTrump(c,ctx)`             | Boolean                                                          |
| `findNKinds(hand, n)`        | All n-of-a-kinds (e.g., pairs) in `hand`                         |
| `findTractors(h,ctx,n)`      | All tractors                                                     |
| `classifyCombo(cards,ctx,n)` | `{ok, combo}` — is this a single/pair/tractor?                  |
| `comboBeats(a, b)`           | `1`/`0`/`-1` — same-structure comparison                         |
| `decomposePlay(...)`         | Decompose a play (single combo OR throw) into matchable parts    |
| `decomposedBeats(...)`       | Same-shape winner detection that allows trumps to beat throws    |
| `validateFollow(...)`        | Confirms a candidate satisfies follow-suit + pair/tractor rules  |
| `decomposeThrow(...)`        | Tries to decompose a multi-combo lead into components            |
| `isThrowSafe(...)`           | §7.6 safety check                                                |
| `pointsForCard(c)`           | `5`/`10`/`0` (5s, 10s+Ks worth 10, everything else 0)            |
| `teamOf(state, seat)`        | Team id ('A'/'B')                                                |
| `nextToPlay(state)`          | Seat the engine expects to play next during `play`               |

For richer enumeration of legal moves, `legalActions(state, seat)` /
`legalEvents(state, seat)` give a curated list of candidate events —
useful as a starting set for search-based agents.

> Note: when resolving who wins a trick containing a throw, use
> `decomposePlay` + `decomposedBeats`. The older `classifyCombo` +
> `comboBeats` path returns `ok: false` on multi-component plays and
> will silently award the lead to the leader on every throw.

## 6. Strategy sketches you can crib from

- **Simple heuristic**: see `pickAggressive` / `pickPointHoarder` for
  20-40 line agents that read the in-flight trick to choose a per-card
  preference. They all share `fillFollow`, which builds a legal follow
  given a preference (`'low'` / `'high'` / `'point-dump'`).
- **Search with determinization** (the smart-v13+ lineage): sample
  `N` consistent opponent hands via `maskOpponentHands`, simulate each
  candidate play with `simulateTricksAndWinsV18` (or a clone), average
  the score. v18 is the canonical version — clone it and change one
  knob to test a hypothesis.
- **Off-policy / learned**: see `rl-mlp` for a thin wrapper around a
  trained MLP policy that defers to a baseline for non-play phases.

## 7. Testing and benchmarking your agent

The engine ships an integration test that runs a whole hand with 4 bots:
`packages/engine/test/bot.test.ts`. To add a smoke test for a new
strategy, follow the existing pattern (call `runFullHandWithBots` and
assert the hand reaches `between_hands`).

For head-to-head benchmarks, `runMatch` (in `packages/engine/src/eval.ts`)
runs N seat-symmetric games and reports wins per team:

```ts
const m1 = runMatch({
  agents: ['my-new-bot', 'smart-v34', 'my-new-bot', 'smart-v34'],
  numGames: 8,
  seed: 1234,
  maxHandsPerGame: 50,
});
const m2 = runMatch({
  agents: ['smart-v34', 'my-new-bot', 'smart-v34', 'my-new-bot'],
  numGames: 8,
  seed: 1235,
  maxHandsPerGame: 50,
});
```

**Lessons from the lineage about benchmark sample size**: a 32-game
match has a 95% confidence interval roughly ±18%. Two prior variants
(`smart-v28`, `smart-v32`) initially benchmarked 56-59% in 32 games
and were promoted, then **flipped to 31-41% in a confirmation run with
fresh seeds**. The rule is:

> For deployment, run an initial 32-game match. If it suggests an
> improvement (≥ 17 wins), run a **confirmation** 32-game match with
> different seeds. Deploy only if BOTH halves are positive (or the
> 64-game combined total is ≥ 36 wins).

`v33` and `v34` followed this protocol and held up. `v28` and `v32`
would have been caught by this protocol.

## 8. Off-process agents (Python, etc.)

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

## 9. Checklist (copy / paste when shipping a new agent)

- [ ] New entry in `BotType` union
- [ ] New entry in `BOT_TYPES` array
- [ ] New entry in `VISIBLE_BOT_TYPES` (only if the agent is ready for users)
- [ ] New case in `botTypeLabel`
- [ ] Implementation function — simple `(myCards, trick, ctx, n)` shape, OR smart `(state, seat)` shape
- [ ] Dispatch in `pickBotAction`: either a new `case` in the simple-bot switch, or a new `if` branch in the smart-bot chain
- [ ] If the agent runs in live games, decide what it does in the
      `'dealing'` phase (or rely on `pickDealingPhaseDeclare`)
- [ ] `npm run -ws --if-present typecheck` is clean
- [ ] `npm test --workspace @tractor/engine` is green
- [ ] (Optional) Smoke test in `bot.test.ts`
- [ ] (For improvements over current peak) Run a 32-game `runMatch` vs
      the deployed peak. If ≥ 17 wins, run a confirmation match with
      fresh seeds before promoting / adding to `VISIBLE_BOT_TYPES`.
- [ ] Smoke-test the agent in the room UI: select the strategy from the
      dropdown, click "Add bot" four times, watch a full hand play out.
