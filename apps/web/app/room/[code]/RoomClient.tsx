'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  classifyCombo,
  compareCardsForDisplay,
  comboBeats,
  decomposeThrow,
  isTrump,
  pointsForCard,
  validateFollow,
  validateFollowThrow,
  BOT_TYPES,
  botTypeLabel,
  type BotType,
  type Card,
  type NaturalRank,
  type Seat,
  type TrumpContext,
} from '@tractor/engine';
import { DEAL_INTERVAL_MS } from '@tractor/protocol';
import { cardId } from '../../../lib/cards';
import { getOrCreateSessionToken, useTractorSocket } from '../../../lib/ws';
import { CardSvg } from '../../../components/Cards';
import {
  availableDeclarations,
  declarationSuitClass,
  trumpRankForSeat,
  type AvailableDeclaration,
} from '../../../lib/declarations';

export default function RoomClient({ code }: { code: string }) {
  const router = useRouter();
  const { state, send } = useTractorSocket();
  const [nick, setNick] = useState('');
  const [selected, setSelected] = useState<ReadonlyArray<string>>([]);
  const [autoAdvance, setAutoAdvance] = useState(false);
  // Pre-play: when armed and it becomes my turn, the currently-selected
  // cards get sent automatically.
  const [autoPlayArmed, setAutoPlayArmed] = useState(false);
  const armedWinningSeatRef = useRef<Seat | null>(null);
  const [spectating, setSpectating] = useState(false);
  // Default bot type for the "Add bot" button on empty seats. Stored in
  // RoomClient so the choice persists across renders.
  const [defaultBotType, setDefaultBotType] = useState<BotType>('lowball');

  useEffect(() => {
    const saved = localStorage.getItem('tractor:autoAdvance');
    if (saved === '1') setAutoAdvance(true);
  }, []);
  useEffect(() => {
    localStorage.setItem('tractor:autoAdvance', autoAdvance ? '1' : '0');
  }, [autoAdvance]);

  useEffect(() => {
    const saved = localStorage.getItem('tractor:nickname');
    if (saved === null || saved.trim() === '') {
      router.replace('/');
      return;
    }
    setNick(saved);
  }, [router]);

  useEffect(() => {
    if (!state.connected || nick === '') return;
    if (state.room?.roomCode === code) return;
    send({
      kind: 'join',
      nickname: nick,
      roomCode: code,
      sessionToken: getOrCreateSessionToken(),
    });
  }, [state.connected, state.room?.roomCode, nick, code, send]);

  // Clear pre-selection only at hand boundaries — the user explicitly
  // clears it when they play / bury, and trick_review ↔ play transitions
  // must preserve the pending box so an armed Auto-play carries over.
  useEffect(() => {
    setSelected([]);
    setAutoPlayArmed(false);
  }, [state.room?.gameState.currentHand?.handNumber]);

  // Auto-advance: once the table is in trick_review, wait a moment so the
  // last play is visible, then send advance_trick. Re-fires for each new
  // trick (handNumber + tricks.length acts as the identity).
  const trickAdvancedRef = useRef<string | null>(null);
  useEffect(() => {
    const phase = state.room?.gameState.phase;
    const ch = state.room?.gameState.currentHand;
    const key =
      ch !== null && ch !== undefined
        ? `${ch.handNumber}:${ch.tricks.length}`
        : null;
    if (!autoAdvance || phase !== 'trick_review' || key === null) return;
    if (trickAdvancedRef.current === key) return;
    const t = setTimeout(() => {
      trickAdvancedRef.current = key;
      send({ kind: 'advance_trick' });
    }, 1500);
    return () => clearTimeout(t);
  }, [
    autoAdvance,
    state.room?.gameState.phase,
    state.room?.gameState.currentHand?.handNumber,
    state.room?.gameState.currentHand?.tricks.length,
    send,
  ]);

  const room = state.room;
  const priv = state.priv;
  const game = room?.gameState ?? null;
  const ch = game?.currentHand ?? null;

  const mySeat: Seat | null = useMemo(() => {
    if (room === null || state.clientId === null) return null;
    for (const s of room.seats) {
      if (s.occupant?.clientId === state.clientId) return s.seat;
    }
    return null;
  }, [room, state.clientId]);

  // Dealing animation, one card per seat per round. The animation effect
  // only listens to `handNumber` so a phase change mid-deal doesn't
  // re-trigger and freeze the interval before it finishes.
  const [dealtTotal, setDealtTotal] = useState<number>(Infinity);
  const lastHandRef = useRef<number | null>(null);
  useEffect(() => {
    const handNumber = ch?.handNumber ?? null;
    const phase = game?.phase;
    if (handNumber === null) {
      lastHandRef.current = null;
      setDealtTotal(Infinity);
      return;
    }
    if (lastHandRef.current === handNumber) return;
    lastHandRef.current = handNumber;
    if (phase !== 'declare_lockout') {
      setDealtTotal(Infinity);
      return;
    }
    setDealtTotal(0);
    const cfg = state.room!.gameState.config;
    const totalCards = cfg.numDecks * 54 - cfg.kittySize;
    const interval = setInterval(() => {
      setDealtTotal((c) => {
        const next = c + 1;
        if (next >= totalCards) {
          clearInterval(interval);
          return Infinity;
        }
        return next;
      });
    }, DEAL_INTERVAL_MS);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ch?.handNumber]);

  // Safety: outside the declare_lockout phase, all dealt cards should be
  // visible. This guarantees that pickup / play / between-hands never see
  // a stale finite `dealtTotal` (which would cap visible card counts).
  useEffect(() => {
    if (game?.phase !== 'declare_lockout') {
      setDealtTotal(Infinity);
    }
  }, [game?.phase]);

  const cardsPerPlayer = useMemo(() => {
    const cfg = room?.gameState.config;
    if (cfg === undefined) return 0;
    return Math.floor((cfg.numDecks * 54 - cfg.kittySize) / cfg.numPlayers);
  }, [room?.gameState.config]);

  // Round-robin dealing: card N goes to seat (N mod numPlayers). After
  // `dealtTotal` cards total, each seat has `floor(dealtTotal / numPlayers)`
  // cards plus 1 more for seats whose index is less than the remainder.
  function cardsDealtToSeat(seat: Seat, realCount: number): number {
    if (!isFinite(dealtTotal)) return realCount;
    const numPlayers = room?.gameState.config.numPlayers ?? 4;
    const rounds = Math.floor(dealtTotal / numPlayers);
    const remainder = dealtTotal - rounds * numPlayers;
    const got = rounds + (seat < remainder ? 1 : 0);
    return Math.min(realCount, cardsPerPlayer, got);
  }

  const rawCards = priv?.cards ?? [];

  const visibleRaw = useMemo(() => {
    if (mySeat === null) return rawCards;
    const dealt = cardsDealtToSeat(mySeat, rawCards.length);
    return rawCards.slice(0, dealt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawCards, dealtTotal, mySeat, cardsPerPlayer]);

  // Keep the hand sorted at all times. The engine deals random cards
  // (round-robin from a shuffled deck), so each new card arriving during
  // the deal animation is a random pick — but it slots into its sorted
  // position on insertion. End result: the player sees a sorted hand
  // growing card-by-card.
  const myCards = useMemo(() => {
    const trumpRank: NaturalRank =
      (ch?.trumpRank as NaturalRank | null) ?? '2';
    const ctx: TrumpContext = { trumpRank, trumpSuit: ch?.trumpSuit ?? null };
    return [...visibleRaw].sort((a, b) => compareCardsForDisplay(a, b, ctx));
  }, [visibleRaw, ch?.trumpRank, ch?.trumpSuit]);

  // The active trump context, or null until trump has been finalized for
  // this hand. Used to mark cards in the UI with a crown.
  const liveTrumpCtx: TrumpContext | null =
    ch !== null && ch.trumpFinalized && ch.trumpRank !== null
      ? { trumpRank: ch.trumpRank, trumpSuit: ch.trumpSuit }
      : null;
  const isTrumpCard = (c: Card): boolean =>
    liveTrumpCtx !== null && isTrump(c, liveTrumpCtx);

  // True when the local player can currently act on their cards (and so
  // clicking a card should move it into the playing box). Outside of these
  // states, the hand is read-only — no playing box, no click-to-select.
  const myTurn =
    mySeat !== null && game !== null && computeNextSeatToPlay(game) === mySeat;
  const isLandlordBurying =
    mySeat !== null &&
    game?.phase === 'kitty_replace' &&
    ch?.landlord === mySeat;
  const handIsActionable =
    (game?.phase === 'play' && myTurn) || isLandlordBurying;
  // Pre-play: in the play phase you can already arrange cards in the
  // playing box even when it isn't your turn — when the turn comes
  // around, the armed selection auto-fires. The playing box stays
  // mounted across trick_review too so it doesn't shift the page layout
  // between tricks.
  const canSelectCards =
    game?.phase === 'play' ||
    game?.phase === 'trick_review' ||
    isLandlordBurying;

  // Play a short beep when it becomes our turn (false → true edge).
  // We initialize the ref to true so the very first turn doesn't beep
  // before the user has interacted with the page (browsers block audio
  // until a gesture anyway, but this also avoids a stale-fire on join).
  const prevTurnRef = useRef<boolean>(true);
  useEffect(() => {
    const wasTurn = prevTurnRef.current;
    prevTurnRef.current = myTurn;
    if (myTurn && !wasTurn) {
      playTurnBeep();
    }
  }, [myTurn]);

  // Seat with the currently-winning combo in the active/just-finished
  // trick. The lead always seeds the comparison; later plays only take
  // over if they strictly beat it. Used to highlight that seat's play
  // box in red.
  const winningSeat: Seat | null = useMemo(() => {
    if (liveTrumpCtx === null || ch?.currentTrick === null || ch?.currentTrick === undefined) {
      return null;
    }
    const plays = ch.currentTrick.plays;
    if (plays.length === 0) return null;
    const n = game?.config.numDecks ?? 1;
    const leadClass = classifyCombo(plays[0]!.cards, liveTrumpCtx, n);
    if (!leadClass.ok) {
      // Throw lead auto-wins (engine skips this trick's beat check on
      // followers because none can match a throw structurally).
      return plays[0]!.seat;
    }
    let bestSeat = plays[0]!.seat;
    let bestCombo = leadClass.combo;
    for (let i = 1; i < plays.length; i++) {
      const cls = classifyCombo(plays[i]!.cards, liveTrumpCtx, n);
      if (!cls.ok) continue;
      if (comboBeats(cls.combo, bestCombo) > 0) {
        bestSeat = plays[i]!.seat;
        bestCombo = cls.combo;
      }
    }
    return bestSeat;
  }, [
    liveTrumpCtx,
    ch?.currentTrick,
    game?.config.numDecks,
  ]);

  function toggleSelect(id: string) {
    setSelected((sel) =>
      sel.includes(id) ? sel.filter((x) => x !== id) : [...sel, id],
    );
  }
  function clearSelection() {
    setSelected([]);
  }
  function selectedCards(): Card[] {
    const out: Card[] = [];
    for (const c of myCards) {
      if (selected.includes(cardId(c))) out.push(c);
    }
    return out;
  }

  // Auto-play arming: disarm if the seat that's currently winning the
  // trick changes from the seat that was winning when auto-play was
  // armed. Transitions to "no winner yet" (between tricks) are ignored
  // so the arm survives a trick advance.
  useEffect(() => {
    if (!autoPlayArmed) return;
    if (winningSeat !== null && winningSeat !== armedWinningSeatRef.current) {
      setAutoPlayArmed(false);
    }
  }, [autoPlayArmed, winningSeat]);

  // When the turn comes around and auto-play is armed, fire the play if
  // the selection is still a legal play; otherwise just disarm.
  useEffect(() => {
    if (!autoPlayArmed) return;
    if (!myTurn || mySeat === null || game === null) return;
    const cards = selectedCards();
    if (cards.length === 0) return;
    if (!isPlayValid(game, mySeat, myCards, cards)) {
      setAutoPlayArmed(false);
      return;
    }
    send({ kind: 'play_cards', cards });
    clearSelection();
    setAutoPlayArmed(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoPlayArmed, myTurn]);

  function armAutoPlay(): void {
    armedWinningSeatRef.current = winningSeat;
    setAutoPlayArmed(true);
  }
  function disarmAutoPlay(): void {
    setAutoPlayArmed(false);
  }

  return (
    <div className="room">
      <div className="header">
        <div>
          <span className="h2">Room {code}</span>{' '}
          <span className="muted">
            ({state.connected ? 'connected' : 'reconnecting…'})
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <label className="auto-advance">
            <input
              type="checkbox"
              checked={spectating}
              onChange={(e) => {
                const on = e.target.checked;
                setSpectating(on);
                send({ kind: 'set_spectate', on });
              }}
            />
            <span>Spectate (show all hands)</span>
          </label>
          <button
            onClick={() => {
              send({ kind: 'leave' });
              router.push('/');
            }}
          >
            Leave
          </button>
        </div>
      </div>

      <div className="room-main">
        {room === null ? (
          <div className="muted">Joining room…</div>
        ) : (
          <>
            <PokerTable
              room={room}
              game={game!}
              mySeat={mySeat}
              cardsDealtToSeat={cardsDealtToSeat}
              isTrumpCard={isTrumpCard}
              winningSeat={winningSeat}
              defaultBotType={defaultBotType}
              setDefaultBotType={setDefaultBotType}
              send={send}
            />

            <PointsMeter game={game!} />

            <div className="hand-row">
              <div className="hand-col">
                {mySeat !== null ? (
                  <ActionsBar
                    room={room}
                    game={game!}
                    mySeat={mySeat}
                    myCards={myCards}
                    selected={selectedCards()}
                    clearSelection={clearSelection}
                    send={send}
                    autoAdvance={autoAdvance}
                    setAutoAdvance={setAutoAdvance}
                    autoPlayArmed={autoPlayArmed}
                    armAutoPlay={armAutoPlay}
                    disarmAutoPlay={disarmAutoPlay}
                  />
                ) : (
                  <SpectatorActionsBar room={room} game={game!} send={send} />
                )}
                {spectating && state.spectateHands !== null ? (
                  <SpectatorHands
                    hands={state.spectateHands}
                    room={room}
                    isTrumpCard={isTrumpCard}
                    trumpCtx={liveTrumpCtx}
                    cardsDealtToSeat={cardsDealtToSeat}
                  />
                ) : (
                  <>
                    {canSelectCards && (
                      <PlayingBox
                        cards={myCards.filter((c) => selected.includes(cardId(c)))}
                        isTrumpCard={isTrumpCard}
                        onUnselect={(id) => toggleSelect(id)}
                      />
                    )}
                    <div style={{ marginTop: 8 }}>
                      <HandView
                        cards={
                          canSelectCards
                            ? myCards.filter((c) => !selected.includes(cardId(c)))
                            : myCards
                        }
                        isTrumpCard={isTrumpCard}
                        onSelect={canSelectCards ? toggleSelect : undefined}
                      />
                    </div>
                    {priv?.buriedKitty !== undefined && priv.buriedKitty.length > 0 && (
                      <BuriedKittyView
                        cards={priv.buriedKitty}
                        isTrumpCard={isTrumpCard}
                      />
                    )}
                  </>
                )}
              </div>
              <PointsPanel
                room={room}
                game={game!}
                isTrumpCard={isTrumpCard}
              />
            </div>

          </>
        )}
      </div>

      <ChatPanel room={room} send={send} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Table: a row of per-player boxes with each player's bubble above the box.
// Cards a player plays land inside their box.
// ---------------------------------------------------------------------------
function PokerTable({
  room,
  game,
  mySeat,
  cardsDealtToSeat,
  isTrumpCard,
  winningSeat,
  defaultBotType,
  setDefaultBotType,
  send,
}: {
  room: import('@tractor/protocol').RoomView;
  game: import('@tractor/protocol').GameStateView;
  mySeat: Seat | null;
  cardsDealtToSeat: (seat: Seat, real: number) => number;
  isTrumpCard: (c: Card) => boolean;
  winningSeat: Seat | null;
  defaultBotType: BotType;
  setDefaultBotType: (t: BotType) => void;
  send: (msg: import('@tractor/protocol').ClientMessage) => void;
}) {
  const ch = game.currentHand;
  const turnSeat = computeNextSeatToPlay(game);
  const inLobby =
    game.phase === 'between_hands' || game.phase === 'game_over';

  return (
    <div className="table-wrap">
      <CenterInfo game={game} />
      <div className="player-row">
        {room.seats.map((seat) => {
          const realCount = ch?.handSizes[seat.seat] ?? 0;
          const animatedCount = cardsDealtToSeat(seat.seat, realCount);
          // The play-box shows either this seat's play in the current
          // trick, or the cards backing the standing declaration if it's
          // theirs (during bidding phases).
          const trickPlay =
            ch?.currentTrick?.plays.find((p) => p.seat === seat.seat) ?? null;
          const decl = ch?.standingDeclaration ?? null;
          const declCards =
            decl !== null &&
            decl.seat === seat.seat &&
            (game.phase === 'declare_lockout' ||
              game.phase === 'declare_open')
              ? decl.cards
              : null;
          const displayedPlay = trickPlay ?? (declCards ? { cards: declCards } : null);
          return (
            <PlayerCell
              key={seat.seat}
              seat={seat}
              isSelf={mySeat === seat.seat}
              isLandlord={ch?.landlord === seat.seat}
              isTurn={turnSeat === seat.seat}
              isWinning={winningSeat === seat.seat}
              cardCount={animatedCount}
              displayedPlay={displayedPlay}
              inLobby={inLobby}
              isTrumpCard={isTrumpCard}
              landlordTeam={
                game.landlord !== null
                  ? game.config.seatToTeam[game.landlord] ?? null
                  : null
              }
              seatTeam={game.config.seatToTeam[seat.seat] ?? null}
              seatRank={
                game.teamLevels[game.config.seatToTeam[seat.seat] ?? ''] ??
                null
              }
              defaultBotType={defaultBotType}
              setDefaultBotType={setDefaultBotType}
              send={send}
            />
          );
        })}
      </div>
    </div>
  );
}

function CenterInfo({
  game,
}: {
  game: import('@tractor/protocol').GameStateView;
}) {
  const ch = game.currentHand;

  const SUIT: Record<string, string> = {
    spades: '♠',
    hearts: '♥',
    diamonds: '♦',
    clubs: '♣',
  };

  const trumpDisplay =
    ch === null || ch.trumpRank === null
      ? null
      : ch.trumpSuit === null
        ? '🃏'
        : (SUIT[ch.trumpSuit] ?? ch.trumpSuit);

  const isRedTrump = ch?.trumpSuit === 'hearts' || ch?.trumpSuit === 'diamonds';

  return (
    <div>
      {trumpDisplay !== null && (
        <div
          style={{
            fontSize: 36,
            lineHeight: 1,
            color: isRedTrump ? '#e88a8a' : '#e8efe9',
          }}
        >
          {trumpDisplay}
        </div>
      )}
      {ch !== null && ch.trumpRank !== null && (
        <div style={{ fontSize: 12, color: '#94a89a' }}>R = {ch.trumpRank}</div>
      )}
    </div>
  );
}

function PlayerCell({
  seat,
  isSelf,
  isLandlord,
  isTurn,
  isWinning,
  cardCount: _cardCount,
  displayedPlay,
  inLobby,
  isTrumpCard,
  landlordTeam,
  seatTeam,
  seatRank,
  defaultBotType,
  setDefaultBotType,
  send,
}: {
  seat: import('@tractor/protocol').SeatInfo;
  isSelf: boolean;
  isLandlord: boolean;
  isTurn: boolean;
  isWinning: boolean;
  cardCount: number;
  // The cards to display in this seat's play-box. Either the seat's
  // play in the current trick, or — during the bidding phase — the
  // cards backing the standing declaration if this seat made it.
  displayedPlay: { cards: ReadonlyArray<Card> } | null;
  inLobby: boolean;
  isTrumpCard: (c: Card) => boolean;
  // Team id (e.g. "A" or "B") of the landlord this hand. Drives the
  // red/blue color of each player's avatar. Null pre-game / between hands.
  landlordTeam: string | null;
  // This seat's team id, or null if unknown (lobby).
  seatTeam: string | null;
  // This seat's current rank (their team's level — e.g., '2', 'K', 'A').
  seatRank: NaturalRank | null;
  defaultBotType: BotType;
  setDefaultBotType: (t: BotType) => void;
  send: (msg: import('@tractor/protocol').ClientMessage) => void;
}) {
  const occupant = seat.occupant;
  const play = displayedPlay;

  // Avatar color: red for landlord's team, blue for the opposing team.
  // Before trump is finalized (no landlord set), fall back to a neutral
  // nickname-derived hue so seats still look distinct.
  let avatarBg: string;
  if (landlordTeam !== null && occupant !== null) {
    avatarBg = seatTeam === landlordTeam ? '#a83434' : '#345fa8';
  } else if (occupant !== null) {
    avatarBg = avatarColor(occupant.nickname);
  } else {
    avatarBg = '#243528';
  }

  const bubble =
    occupant === null ? (
      <div className="bubble empty">
        <div className="avatar empty">?</div>
        <div className="bubble-name">Seat {seat.seat}</div>
        {inLobby && (
          <div className="bubble-controls">
            <button onClick={() => send({ kind: 'take_seat', seat: seat.seat })}>
              Take seat
            </button>
            <select
              className="bot-type-select"
              value={defaultBotType}
              onChange={(e) => setDefaultBotType(e.target.value as BotType)}
              title="Bot strategy for this seat"
            >
              {BOT_TYPES.map((t) => (
                <option key={t} value={t}>
                  {botTypeLabel(t)}
                </option>
              ))}
            </select>
            <button
              onClick={() =>
                send({
                  kind: 'add_bot',
                  seat: seat.seat,
                  botType: defaultBotType,
                })
              }
            >
              🤖 Add bot
            </button>
          </div>
        )}
      </div>
    ) : (
      <div className="bubble">
        <div
          className={
            'avatar' +
            (isSelf ? ' self' : '') +
            (isTurn ? ' turn' : '') +
            (!occupant.connected ? ' disconnected' : '')
          }
          style={{ background: avatarBg }}
        >
          {avatarInitials(occupant.nickname)}
        </div>
        <div className="bubble-name-row">
          {seatRank !== null && (
            <span className="bubble-rank">{seatRank}</span>
          )}
          <span className="bubble-name">{occupant.nickname}</span>
          {seat.isBot && <span className="pod-badge bot">bot</span>}
          {isLandlord && <span className="pod-badge landlord">庄</span>}
        </div>
        {!occupant.connected && <div className="bubble-meta">offline</div>}
        {seat.isBot && inLobby && (
          <div className="bubble-controls">
            <button onClick={() => send({ kind: 'remove_bot', seat: seat.seat })}>
              Remove bot
            </button>
          </div>
        )}
      </div>
    );

  return (
    <div className={'player-cell' + (isTurn ? ' turn' : '')}>
      {bubble}
      <div className={'play-box' + (isWinning ? ' winning' : '')}>
        {play === null ? (
          <span className="play-box-empty">—</span>
        ) : (
          play.cards.map((c) => (
            <CardSvg key={cardId(c)} card={c} trump={isTrumpCard(c)} />
          ))
        )}
      </div>
    </div>
  );
}

// Short ~120ms tone via Web Audio. Created lazily so we don't construct
// the AudioContext before the user has interacted with the page (some
// browsers block that). Best-effort: silently swallows errors.
let audioCtx: AudioContext | null = null;
function playTurnBeep(): void {
  if (typeof window === 'undefined') return;
  try {
    if (audioCtx === null) {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      audioCtx = new Ctx();
    }
    const ctx = audioCtx;
    if (ctx.state === 'suspended') void ctx.resume();
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, now);
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
    osc.connect(gain).connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.2);
  } catch {
    // ignore — audio is a nice-to-have
  }
}

function avatarInitials(name: string): string {
  const trimmed = name.trim();
  if (trimmed === '') return '?';
  const words = trimmed.split(/\s+/);
  if (words.length === 1) return words[0]!.slice(0, 2).toUpperCase();
  return (words[0]![0]! + words[1]![0]!).toUpperCase();
}

function avatarColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) & 0x7fffffff;
  }
  const hue = h % 360;
  return `hsl(${hue}, 50%, 38%)`;
}

function computeNextSeatToPlay(
  game: import('@tractor/protocol').GameStateView,
): Seat | null {
  if (game.phase !== 'play') return null;
  const ch = game.currentHand;
  if (ch === null || ch.landlord === null) return null;
  const trick = ch.currentTrick;
  if (trick === null) {
    return ch.tricks.length === 0
      ? ch.landlord
      : ch.tricks[ch.tricks.length - 1]!.winner;
  }
  return (trick.leader + trick.plays.length) % game.config.numPlayers;
}

// ---------------------------------------------------------------------------
// Attacker (non-landlord team) points tracker
// ---------------------------------------------------------------------------
function PointsPanel({
  room,
  game,
  isTrumpCard,
}: {
  room: import('@tractor/protocol').RoomView;
  game: import('@tractor/protocol').GameStateView;
  isTrumpCard: (c: Card) => boolean;
}): JSX.Element | null {
  const ch = game.currentHand;
  // Use the hand's landlord (preserved through finalization) rather than
  // game.landlord, which may be cleared when attackers win — that would
  // otherwise hide the panel right when the user wants to see the result.
  const handLandlord = ch?.landlord ?? null;
  if (ch === null || handLandlord === null) return null;
  const landlordTeam = game.config.seatToTeam[handLandlord];
  if (landlordTeam === undefined) return null;

  const attackerSeats: Seat[] = [];
  for (let s = 0; s < game.config.numPlayers; s++) {
    if (game.config.seatToTeam[s] !== landlordTeam) attackerSeats.push(s);
  }

  const byseat: Record<Seat, Card[]> = {};
  for (const s of attackerSeats) byseat[s] = [];
  for (const trick of ch.tricks) {
    if (game.config.seatToTeam[trick.winner] === landlordTeam) continue;
    const pts = trick.plays
      .flatMap((p) => p.cards)
      .filter((c) => pointsForCard(c) > 0);
    const existing = byseat[trick.winner] ?? [];
    byseat[trick.winner] = [...existing, ...pts];
  }

  // Kitty bonus: only awarded if attackers won the last trick. Multiplier
  // is 2 × (cards in last winning combo).
  const lastTrick = ch.tricks.length > 0 ? ch.tricks[ch.tricks.length - 1]! : null;
  const lastWinnerTeam =
    lastTrick !== null ? game.config.seatToTeam[lastTrick.winner] : undefined;
  const attackersTookLast =
    lastTrick !== null && lastWinnerTeam !== landlordTeam;
  const kittyCards = ch.kittyCards ?? null;
  let kittyMult = 0;
  if (lastTrick !== null) {
    const lwc = lastTrick.plays.find((p) => p.seat === lastTrick.winner);
    kittyMult = 2 * (lwc?.cards.length ?? 1);
  }

  const totalPoints = game.config.numDecks * 100;
  const cur = ch.attackerPoints;
  const pct = Math.min(100, (cur / totalPoints) * 100);

  return (
    <div className="points-panel">
      <div className="points-header">Attacker points won</div>
      <div className="points-grid">
        {attackerSeats.map((s) => {
          const seatInfo = room.seats.find((x) => x.seat === s);
          const name = seatInfo?.occupant?.nickname ?? `Seat ${s}`;
          const cards = byseat[s] ?? [];
          const pts = cards.reduce((acc, c) => acc + pointsForCard(c), 0);
          // If attackers took the last trick, attribute the kitty cards to
          // that attacker's row (multiplied) so the user can see where the
          // bonus came from.
          const kittyForThisSeat =
            attackersTookLast &&
            kittyCards !== null &&
            lastTrick !== null &&
            lastTrick.winner === s
              ? kittyCards
              : null;
          return (
            <div className="points-row" key={s}>
              <div className="points-name">
                {name} <span className="muted">({pts} pts)</span>
              </div>
              <div className="points-cards">
                {cards.length === 0 && kittyForThisSeat === null ? (
                  <span className="empty-hint">no points yet</span>
                ) : (
                  cards.map((c, i) => (
                    <CardSvg key={i} card={c} trump={isTrumpCard(c)} />
                  ))
                )}
              </div>
              {kittyForThisSeat !== null && (
                <div className="kitty-bonus">
                  <div className="kitty-bonus-label">
                    Kitty × {kittyMult} (
                    {kittyMult * sumPointsOf(kittyForThisSeat)} pts)
                  </div>
                  <div className="points-cards">
                    {kittyForThisSeat.map((c, i) => (
                      <CardSvg key={i} card={c} trump={isTrumpCard(c)} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Two-stop gradient (red at 0% → blue at 100%) used by both the meter bar
// and the breakpoint nodes, so the node at position X is exactly the
// color the bar shows at X.
const METER_LO: readonly [number, number, number] = [196, 69, 69]; // red
const METER_HI: readonly [number, number, number] = [52, 95, 168]; // blue
function gradientColor(pct: number): string {
  const t = Math.max(0, Math.min(1, pct / 100));
  const r = Math.round(METER_LO[0] + t * (METER_HI[0] - METER_LO[0]));
  const g = Math.round(METER_LO[1] + t * (METER_HI[1] - METER_LO[1]));
  const b = Math.round(METER_LO[2] + t * (METER_HI[2] - METER_LO[2]));
  return `rgb(${r}, ${g}, ${b})`;
}

function PointsMeter({
  game,
}: {
  game: import('@tractor/protocol').GameStateView;
}): JSX.Element | null {
  const ch = game.currentHand;
  if (ch === null || ch.landlord === null) return null;
  const totalPoints = game.config.numDecks * 100;
  const cur = ch.attackerPoints;
  const pct = Math.min(100, (cur / totalPoints) * 100);
  // Six breakpoints — 0, 40, 80, 120, 160, 200 for the 2-deck case —
  // scaled to whatever totalPoints the variant uses.
  const step = totalPoints / 5;
  const breakpoints = [0, 1, 2, 3, 4, 5].map((i) => i * step);
  return (
    <div className="points-meter standalone">
      <div className="meter-bar">
        {/* The gradient lives on the bar itself, then a gray overlay
            covers the unfilled portion. That way a breakpoint's position
            on the gradient is fixed, and a node at position X shows the
            same color the bar shows at X. */}
        <div
          className="meter-unfilled"
          style={{ left: `${pct}%`, right: 0 }}
        />
        {breakpoints.map((bp, i) => {
          const bpPct = (bp / totalPoints) * 100;
          const reached = cur >= bp;
          return (
            <div
              key={i}
              className={'meter-node' + (reached ? ' reached' : '')}
              style={
                reached
                  ? { left: `${bpPct}%`, background: gradientColor(bpPct) }
                  : { left: `${bpPct}%` }
              }
            />
          );
        })}
        <span className="meter-fill-label" style={{ left: `${pct}%` }}>
          {cur}
        </span>
      </div>
    </div>
  );
}

function sumPointsOf(cards: ReadonlyArray<Card>): number {
  let n = 0;
  for (const c of cards) n += pointsForCard(c);
  return n;
}

// ---------------------------------------------------------------------------
// Chat / event log
// ---------------------------------------------------------------------------
function ChatPanel({
  room,
  send,
}: {
  room: import('@tractor/protocol').RoomView | null;
  send: (msg: import('@tractor/protocol').ClientMessage) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState('');
  const log = room?.log ?? [];
  useEffect(() => {
    if (ref.current === null) return;
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [log.length]);

  function submit() {
    const t = draft.trim();
    if (t === '') return;
    send({ kind: 'chat_message', text: t });
    setDraft('');
  }

  return (
    <div className="chat">
      <div className="chat-header">Chat & game log</div>
      <div className="chat-log" ref={ref}>
        {log.length === 0 ? (
          <div className="chat-empty">No events yet</div>
        ) : (
          log.map((e, i) => (
            <div
              className={'chat-entry ' + (e.from !== undefined ? 'chat' : 'system')}
              key={i}
            >
              <span className="ts">{formatTime(e.ts)}</span>
              <span>
                {e.from !== undefined && (
                  <strong className="chat-from">{e.from}: </strong>
                )}
                {e.text}
              </span>
            </div>
          ))
        )}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={draft}
          maxLength={300}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Type a message…"
          disabled={room === null}
        />
        <button onClick={submit} disabled={draft.trim() === '' || room === null}>
          Send
        </button>
      </div>
    </div>
  );
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const hh = d.getHours().toString().padStart(2, '0');
  const mm = d.getMinutes().toString().padStart(2, '0');
  return `${hh}:${mm}`;
}

// ---------------------------------------------------------------------------
// Hand + actions
// ---------------------------------------------------------------------------
function HandView({
  cards,
  isTrumpCard,
  onSelect,
}: {
  cards: ReadonlyArray<Card>;
  isTrumpCard: (c: Card) => boolean;
  onSelect?: (id: string) => void;
}) {
  return (
    <div className="hand-svg">
      {cards.length === 0 ? (
        <span className="muted">No cards in hand</span>
      ) : (
        cards.map((c) => {
          const id = cardId(c);
          return (
            <CardSvg
              key={id}
              card={c}
              trump={isTrumpCard(c)}
              onClick={onSelect ? () => onSelect(id) : undefined}
            />
          );
        })
      )}
    </div>
  );
}

function SpectatorHands({
  hands,
  room,
  isTrumpCard,
  trumpCtx,
  cardsDealtToSeat,
}: {
  hands: Readonly<Record<number, ReadonlyArray<Card>>>;
  room: import('@tractor/protocol').RoomView;
  isTrumpCard: (c: Card) => boolean;
  trumpCtx: TrumpContext | null;
  cardsDealtToSeat: (seat: Seat, real: number) => number;
}): JSX.Element {
  // Sort each seat's hand the same way we sort the active player's hand
  // so the spectator can read the cards consistently. Falls back to
  // unsorted when trump isn't finalized yet.
  const ctxForSort: TrumpContext =
    trumpCtx ?? { trumpRank: '2', trumpSuit: null };
  return (
    <div className="spectator-hands">
      {room.seats.map((seat) => {
        const full = hands[seat.seat] ?? [];
        // Apply the same deal-animation slice used for the local player
        // so the spectator sees cards arrive into each seat in real time.
        const dealt = cardsDealtToSeat(seat.seat, full.length);
        const sorted = [...full].sort((a, b) =>
          compareCardsForDisplay(a, b, ctxForSort),
        );
        const cards = sorted.slice(0, dealt);
        const name = seat.occupant?.nickname ?? `Seat ${seat.seat}`;
        return (
          <div className="spectator-row" key={seat.seat}>
            <div className="spectator-label">
              {name}{' '}
              <span className="muted">
                ({cards.length} card{cards.length === 1 ? '' : 's'})
              </span>
            </div>
            <HandView cards={cards} isTrumpCard={isTrumpCard} />
          </div>
        );
      })}
    </div>
  );
}

function BuriedKittyView({
  cards,
  isTrumpCard,
}: {
  cards: ReadonlyArray<Card>;
  isTrumpCard: (c: Card) => boolean;
}): JSX.Element {
  return (
    <div className="buried-kitty">
      <div className="buried-kitty-label">Your buried kitty</div>
      <div className="buried-kitty-cards">
        {cards.map((c, i) => (
          <CardSvg key={i} card={c} trump={isTrumpCard(c)} />
        ))}
      </div>
    </div>
  );
}

function PlayingBox({
  cards,
  isTrumpCard,
  onUnselect,
}: {
  cards: ReadonlyArray<Card>;
  isTrumpCard: (c: Card) => boolean;
  onUnselect: (id: string) => void;
}) {
  return (
    <div className="playing-box">
      {cards.map((c) => {
        const id = cardId(c);
        return (
          <CardSvg
            key={id}
            card={c}
            trump={isTrumpCard(c)}
            onClick={() => onUnselect(id)}
          />
        );
      })}
    </div>
  );
}

// Whether the current selection is a legal play given the phase/turn.
// Pure of any DOM state — used to gate the submit button.
function isPlayValid(
  game: import('@tractor/protocol').GameStateView,
  mySeat: Seat,
  myCards: ReadonlyArray<Card>,
  selected: ReadonlyArray<Card>,
): boolean {
  const ch = game.currentHand;
  const phase = game.phase;

  if (phase === 'kitty_replace') {
    if (ch?.landlord !== mySeat) return false;
    if (selected.length !== game.config.kittySize) return false;
    return true;
  }

  if (phase === 'play') {
    if (ch === null || ch.trumpRank === null) return false;
    if (computeNextSeatToPlay(game) !== mySeat) return false;
    if (selected.length === 0) return false;

    const ctx: TrumpContext = {
      trumpRank: ch.trumpRank as NaturalRank,
      trumpSuit: ch.trumpSuit,
    };
    const n = game.config.numDecks;
    const trick = ch.currentTrick;

    if (trick === null) {
      // Leader: single combo, or a throw (server applies forced-lead
      // penalty if unsafe — extra cards come back to hand).
      if (classifyCombo(selected, ctx, n).ok) return true;
      return decomposeThrow(selected, ctx, n).ok;
    }

    const lead = trick.plays[0]!.cards;
    if (selected.length !== lead.length) return false;
    const leadClass = classifyCombo(lead, ctx, n);
    if (leadClass.ok) {
      return validateFollow(leadClass.combo, selected, myCards, ctx, n).ok;
    }
    const leadThrow = decomposeThrow(lead, ctx, n);
    if (leadThrow.ok) {
      return validateFollowThrow(leadThrow.throw, selected, myCards, ctx).ok;
    }
    return false;
  }

  return false;
}

function ActionsBar({
  room,
  game,
  mySeat,
  myCards,
  selected,
  clearSelection,
  send,
  autoAdvance,
  setAutoAdvance,
  autoPlayArmed,
  armAutoPlay,
  disarmAutoPlay,
}: {
  room: import('@tractor/protocol').RoomView;
  game: import('@tractor/protocol').GameStateView;
  mySeat: Seat;
  myCards: ReadonlyArray<Card>;
  selected: ReadonlyArray<Card>;
  clearSelection: () => void;
  send: (msg: import('@tractor/protocol').ClientMessage) => void;
  autoAdvance: boolean;
  setAutoAdvance: (v: boolean) => void;
  autoPlayArmed: boolean;
  armAutoPlay: () => void;
  disarmAutoPlay: () => void;
}) {
  const ch = game.currentHand;
  const phase = game.phase;

  if (phase === 'between_hands' || phase === 'game_over') {
    return (
      <div className="actions">
        {phase === 'game_over' && game.winner !== null && (
          <span>🏆 Game over! Team {game.winner} wins.</span>
        )}
        {phase === 'between_hands' && (
          <button
            disabled={room.seats.some((s) => s.occupant === null)}
            onClick={() => send({ kind: 'start_hand' })}
            title={
              room.seats.some((s) => s.occupant === null)
                ? 'All seats must be filled before starting'
                : ''
            }
          >
            Start {game.handsCompleted === 0 ? 'first' : 'next'} hand
          </button>
        )}
        {phase === 'game_over' && (
          <button onClick={() => send({ kind: 'reset_game' })}>New game</button>
        )}
        <button onClick={() => send({ kind: 'leave_seat' })}>Leave seat</button>
      </div>
    );
  }

  if (phase === 'declare_lockout' || phase === 'declare_open') {
    const myR = trumpRankForSeat(mySeat, game);
    const declarations: AvailableDeclaration[] =
      myR === null
        ? []
        : availableDeclarations(myCards, myR, ch?.standingDeclaration ?? null);
    return (
      <div className="actions">
        {declarations.length === 0 ? (
          <span className="muted">No declaration available from your hand</span>
        ) : (
          declarations.map((d, idx) => (
            <button
              key={idx}
              className={`declare-btn ${declarationSuitClass(d)}`}
              onClick={() => {
                send({ kind: 'declare', cards: d.cards });
                clearSelection();
              }}
              title="Declare"
            >
              <span className="declare-cards">
                {d.cards.map((c, i) => (
                  <CardSvg key={i} card={c} size="small" />
                ))}
              </span>
            </button>
          ))
        )}
        {phase === 'declare_open' && ch?.landlord === mySeat && (
          <button onClick={() => send({ kind: 'pick_up_kitty' })}>
            Pick up kitty
          </button>
        )}
      </div>
    );
  }

  if (phase === 'kitty_replace') {
    if (ch?.landlord !== mySeat) {
      return <span className="muted">Waiting for landlord to bury 8 cards…</span>;
    }
    const valid = isPlayValid(game, mySeat, myCards, selected);
    return (
      <div className="actions">
        <span className="muted">
          Select 8 cards to bury ({selected.length}/8)
        </span>
        <button
          disabled={!valid}
          onClick={() => {
            send({ kind: 'replace_kitty', cards: selected });
            clearSelection();
          }}
        >
          Bury
        </button>
      </div>
    );
  }

  if (phase === 'trick_review') {
    const ct = ch?.currentTrick ?? null;
    const winnerHint =
      ct !== null && ct.plays.length > 0
        ? ` Lead: seat ${ct.leader}.`
        : '';
    return (
      <div className="actions">
        <span className="muted">Trick complete.{winnerHint}</span>
        <button onClick={() => send({ kind: 'advance_trick' })}>
          Next trick →
        </button>
        <AutoAdvanceToggle value={autoAdvance} onChange={setAutoAdvance} />
      </div>
    );
  }

  if (phase === 'play') {
    const myTurn = computeNextSeatToPlay(game) === mySeat;
    const valid = isPlayValid(game, mySeat, myCards, selected);
    if (myTurn) {
      return (
        <div className="actions">
          <button
            disabled={!valid}
            onClick={() => {
              send({ kind: 'play_cards', cards: selected });
              clearSelection();
            }}
            title={
              selected.length === 0
                ? 'Select cards to play'
                : !valid
                  ? 'Not a legal play'
                  : ''
            }
          >
            Play
          </button>
          <AutoAdvanceToggle value={autoAdvance} onChange={setAutoAdvance} />
        </div>
      );
    }
    // Not my turn — offer Auto-play arming if cards are pre-selected.
    return (
      <div className="actions">
        <button
          className={autoPlayArmed ? 'auto-play armed' : 'auto-play'}
          disabled={selected.length === 0 && !autoPlayArmed}
          onClick={() => (autoPlayArmed ? disarmAutoPlay() : armAutoPlay())}
        >
          Auto-play
        </button>
        <AutoAdvanceToggle value={autoAdvance} onChange={setAutoAdvance} />
        <span className="muted">
          Waiting for seat {computeNextSeatToPlay(game)}…
        </span>
      </div>
    );
  }

  return null;
}

// Unseated players still need the Start / New game / Advance trick
// controls so an all-bot game can be driven from the sidelines.
function SpectatorActionsBar({
  room,
  game,
  send,
}: {
  room: import('@tractor/protocol').RoomView;
  game: import('@tractor/protocol').GameStateView;
  send: (msg: import('@tractor/protocol').ClientMessage) => void;
}): JSX.Element | null {
  const phase = game.phase;
  if (phase === 'between_hands' || phase === 'game_over') {
    return (
      <div className="actions">
        {phase === 'game_over' && game.winner !== null && (
          <span>🏆 Game over! Team {game.winner} wins.</span>
        )}
        {phase === 'between_hands' && (
          <button
            disabled={room.seats.some((s) => s.occupant === null)}
            onClick={() => send({ kind: 'start_hand' })}
            title={
              room.seats.some((s) => s.occupant === null)
                ? 'All seats must be filled before starting'
                : ''
            }
          >
            Start {game.handsCompleted === 0 ? 'first' : 'next'} hand
          </button>
        )}
        {phase === 'game_over' && (
          <button onClick={() => send({ kind: 'reset_game' })}>New game</button>
        )}
      </div>
    );
  }
  if (phase === 'trick_review') {
    return (
      <div className="actions">
        <button onClick={() => send({ kind: 'advance_trick' })}>
          Next trick →
        </button>
      </div>
    );
  }
  return null;
}

function AutoAdvanceToggle({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
}): JSX.Element {
  return (
    <label className="auto-advance">
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>Auto-advance tricks</span>
    </label>
  );
}

