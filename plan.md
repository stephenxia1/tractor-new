# Tractor — Project Plan

A full-stack web app for playing Tractor (升级/拖拉机), with self-play RL agents that users can eventually play against.

## Goals

1. A correct, well-tested game engine for Tractor.
2. A polished web app where users can play in real-time multiplayer rooms.
3. Bots that fill empty seats (heuristic at first, trained agents later).
4. A self-play RL pipeline that produces agents strong enough to challenge skilled human players.

## Tech stack

- **Frontend + backend**: Next.js (App Router) + TypeScript. Single language across the stack.
- **Realtime**: WebSocket server (standalone Node process or `ws` integrated with Next) — Next.js API routes alone aren't sufficient for game state push.
- **Database**: Postgres (Supabase or Neon for hosting) for users, match history, ratings.
- **Auth**: NextAuth with email/OAuth.
- **Agent training**: Python + PyTorch (separate `/agents` directory with its own toolchain).
- **Bot inference**: Python service exposing a small HTTP/gRPC endpoint the game server calls, OR ONNX export consumed directly in Node.
- **Infra**: Vercel for the Next app; Fly.io / Railway for the WebSocket server and bot inference service.

## Repository layout (target)

```
tractor-new/
├── apps/
│   ├── web/              # Next.js app (UI, auth, pages, API routes)
│   └── ws/               # WebSocket server (rooms, match orchestration)
├── packages/
│   ├── engine/           # Pure TS game engine (rules, state, scoring)
│   ├── protocol/         # Shared message types (client ↔ server, server ↔ bot)
│   └── ui/               # Shared React components
├── agents/
│   ├── engine_py/        # Python port of the engine
│   ├── training/         # Self-play, PPO/AlphaZero loops
│   ├── eval/             # Tournament harness, ELO tracking
│   └── serve/            # Inference service
└── plan.md
```

Use a pnpm workspace + Turborepo for the TS side. Keep `agents/` independent with its own `uv` / `pip` project.

## Engine design (the load-bearing piece)

Tractor is partnership trick-taking with imperfect information, level-up bidding, trump declaration, a kitty (底牌), and combo plays (singles, pairs, tractors). Getting the engine right is the most important early bet — everything else depends on it.

Hard requirements:
- **Pure & deterministic**: state in → state out, no hidden globals. RNG is an explicit input.
- **Serializable**: every state round-trips through JSON. Required for network sync, replays, and training data.
- **Legal-move enumeration**: must produce the full set of legal plays from any state. This is non-trivial because of combo-following rules (when you can't follow, when you must break a pair, etc.).
- **Two views**: a full "god-mode" state (used by the server and training) and a per-seat redacted view (sent to clients and to agents during play).
- **Replay/event log**: every state transition is driven by a typed event. Replays are just re-applying the log.
- **Parameterized over rule variants**: see `rules.md` §13. v1 ships one variant but the engine reads it from a config object, never hard-codes constants.

**Two abstractions that must be present from day one** (because retrofitting them is painful):

1. **Partnership model is per-hand, not global.** Even though v1 has fixed N+S vs E+W partnerships, the data model should treat team membership as a per-hand assignment that can be partial, hidden, or revealed over time. This is what makes the future Finding Friends (找朋友) variant viable without an engine rewrite. The per-seat redacted view must already understand "what I know about partnerships right now" as a first-class concept.
2. **Combos are parameterized by deck count.** "Pair" is really "2-of-a-kind"; with 3 decks it becomes "triplet"; tractors generalize from "consecutive pairs" to "consecutive n-of-a-kind." Write combo logic against an `n` parameter (deck count) from the start, even though v1 always uses `n=2`. Same for `numPlayers`, `kittySize`, and points-per-deck.

Rules scope to decide before coding:
- Number of decks (2 standard for 4 players).
- Variants: 找朋友 (find-a-friend) variants? Different kitty sizes? Different trump-declaration overrides (反主)? Pick one canonical variant for v1; design the engine to allow rule packs later.
- Scoring threshold per level, level-up math, defending-team kitty multiplier.

Testing strategy:
- Unit tests on each rule (follow rules, trump priority, combo comparison, kitty scoring).
- Property tests: legal moves never crash the engine; the sum of points across teams equals the deck total; replays of recorded games yield identical final state.
- Golden games: hand-authored full games checked into the repo as fixtures.

## Phased roadmap

Phases are sequential where they need to be and parallel where they can be. Each phase ends with something demonstrable.

### Phase 1 — Engine (TS)
- Card, deck, hand, seat, partnership primitives.
- State machine: deal → declare trump → kitty → play tricks → score → level up.
- Legal-move generator with combo support.
- Event log + replay.
- Test suite (unit + property + 2–3 golden games).

**Exit criteria**: can simulate a full game end-to-end from the CLI with random legal players.

### Phase 2 — Local prototype UI
- Single-page Next app, no auth, no server.
- Render hands, table, trick area, bidding panel, score.
- "Pass and play" hot-seat mode using the engine directly in the browser.
- Random-legal bot as a stand-in opponent.

**Exit criteria**: a friend can play a full hand of Tractor in the browser locally.

### Phase 3 — Backend, auth, multiplayer rooms
- WebSocket server: rooms, seats, ready-up, start, in-game message routing.
- Server is authoritative — clients send intents, server runs the engine, broadcasts redacted state.
- Auth (NextAuth), user accounts, persistent match history.
- Reconnect handling: client rejoins room and receives current redacted state.

**Exit criteria**: four browsers on four machines play a full game with disconnect/reconnect working.

### Phase 4 — Lobby & polish
- Lobby with open rooms, quick-match, private rooms via code.
- Card animations, sound, mobile layout.
- Spectator mode and replay viewer (free, since the event log already exists).
- Basic rating (Glicko-2 or simple ELO).

**Exit criteria**: shippable to a small group of friends.

### Phase 5 — Heuristic bot
- A rules-based bot that fills empty seats.
- Encodes obvious heuristics: lead trumps when partner is winning, save points, basic signaling.
- Lives behind the same "Player" interface that trained agents will later implement.

**Exit criteria**: bot plays legal, non-embarrassing Tractor and can fill any seat.

### Phase 6 — Python engine port + parity
- Port the engine to Python with the same event log and state shape.
- Shared test vectors: every TS golden game is also a Python fixture; both must produce byte-identical state.
- This is the wall between the app and the training stack — get parity right or the agent silently learns from a different game.

**Exit criteria**: TS and Python engines pass the same fixture suite.

### Phase 7 — Self-play RL training
- Choose a method: start with PPO over per-seat policies, with centralized critic for partnership credit assignment. AlphaZero-style is also viable but harder under imperfect information — would need ISMCTS or similar.
- State encoding: per-seat redacted state → tensor. This is where most of the engineering goes.
- Action space: legal plays only, masked at the policy head.
- Training infra: vectorized self-play, experience buffer, checkpoint cadence, evaluation against heuristic and prior checkpoints.

**Exit criteria**: trained agent beats the heuristic bot consistently in a 1000-game tournament.

### Phase 8 — Bot serving & user-vs-agent
- Export models (TorchScript or ONNX). Decide between a Python inference service vs ONNX-in-Node.
- WebSocket server treats bots as virtual seats: same protocol, just no human attached.
- Latency budget: bot moves should feel like a thinking human (~0.5–2s).

**Exit criteria**: users can queue into a room with three trained bots and play a full game.

### Phase 9 — Iteration
- Stronger agents: longer training, league play, population-based training.
- Difficulty tiers (different checkpoints) so newer players aren't crushed.
- Leaderboards, daily challenges, post-game analysis ("the bot would have played this card").

## Key risks & open questions

- **Engine duplication (TS + Python)**: the parity test suite is the only thing keeping these in sync. If divergence creeps in, agents learn the wrong game. Alternative: write the engine once in Rust and bind to both — more upfront cost, much less ongoing risk. **Decide before Phase 6.**
- **Imperfect information**: vanilla MCTS doesn't work cleanly. Self-play RL with a centralized critic is the safer path; revisit search-based methods only if RL stalls.
- **Rule variant lock-in**: every variant added later is engine + UI + training data churn. v1 ships one canonical variant (see `rules.md`), but the engine reads it from a config — see the two abstractions above. The biggest non-config future variants are **Finding Friends** (dynamic partnerships) and **non-(4,2) table dimensions**; if these aren't accounted for in the core data model, retrofitting them later is a rewrite.
- **Realtime infra cost**: WebSocket servers don't fit serverless well. Budget for a small always-on VM from day one.
- **Cheating / collusion**: partnership games are vulnerable to out-of-band coordination. Out of scope for v1; note as a future concern.

## Immediate next step

Phase 1 — start the TS engine. Specifically: card/deck primitives, the state machine skeleton, and the first golden-game fixture. That fixture is the contract everything else will be measured against.
