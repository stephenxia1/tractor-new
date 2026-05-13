'use client';
import type { Card, JokerRank, NaturalRank, Suit } from '@tractor/engine';

const SUIT_SYMBOL: Record<Suit, string> = {
  spades: '♠',
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
};

const RED_SUITS = new Set<Suit>(['hearts', 'diamonds']);

const CARD_W = 56;
const CARD_H = 80;

const FACE_RANKS = new Set<NaturalRank>(['J', 'Q', 'K']);

type CardProps = {
  card: Card;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  size?: 'normal' | 'small';
  // When true, mark the card visually as trump (crown in two corners).
  // Callers compute this from the active trump context.
  trump?: boolean;
};

export function CardSvg(props: CardProps): JSX.Element {
  const { card } = props;
  if (card.kind === 'joker') {
    return <JokerSvg {...props} rank={card.rank} />;
  }
  return <StandardCardSvg {...props} suit={card.suit} rank={card.rank} />;
}

// Rendered (display) dimensions. The internal viewBox stays at CARD_W ×
// CARD_H so all text/positions inside the SVG scale automatically.
function dims(size: 'normal' | 'small' | undefined): { w: number; h: number } {
  if (size === 'small') return { w: 50, h: 71 };
  return { w: 76, h: 109 };
}

// Unicode "BLACK CHESS QUEEN" — renders as a small crown glyph and
// inherits `fill` color, so we draw it in the same red/black as the rank.
const CROWN_CHAR = '♛';

function StandardCardSvg(
  props: CardProps & { suit: Suit; rank: NaturalRank },
): JSX.Element {
  const { suit, rank, selected, disabled, onClick, size, trump } = props;
  const isRed = RED_SUITS.has(suit);
  const color = isRed ? '#b1303a' : '#15201a';
  const suitChar = SUIT_SYMBOL[suit];
  const { w, h } = dims(size);
  const classes = [
    'card-svg',
    selected ? 'selected' : '',
    disabled ? 'disabled' : '',
    onClick ? 'clickable' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${CARD_W} ${CARD_H}`}
      className={classes}
      onClick={disabled ? undefined : onClick}
    >
      <rect
        x="0.5"
        y="0.5"
        width={CARD_W - 1}
        height={CARD_H - 1}
        rx="5"
        fill="#fdfbf3"
        stroke="#a89e83"
        strokeWidth="0.75"
      />
      {/* Top-left corner */}
      <text
        x="5"
        y="13"
        fontSize="11"
        fontWeight="700"
        fill={color}
        fontFamily="ui-sans-serif, system-ui"
      >
        {rank}
      </text>
      <text x="5" y="24" fontSize="11" fill={color}>
        {suitChar}
      </text>
      {/* Center */}
      {FACE_RANKS.has(rank) ? (
        <text
          x={CARD_W / 2}
          y="50"
          fontSize="28"
          textAnchor="middle"
          fontWeight="700"
          fill={color}
          fontStyle="italic"
          fontFamily="Georgia, 'Times New Roman', serif"
        >
          {rank}
        </text>
      ) : (
        <text
          x={CARD_W / 2}
          y="52"
          fontSize="28"
          textAnchor="middle"
          fill={color}
        >
          {suitChar}
        </text>
      )}
      {/* Bottom-right corner (rotated copy of top-left) */}
      <g transform={`rotate(180 ${CARD_W / 2} ${CARD_H / 2})`}>
        <text
          x="5"
          y="13"
          fontSize="11"
          fontWeight="700"
          fill={color}
          fontFamily="ui-sans-serif, system-ui"
        >
          {rank}
        </text>
        <text x="5" y="24" fontSize="11" fill={color}>
          {suitChar}
        </text>
      </g>
      {trump === true && (
        <>
          {/* Bottom-left and top-right crowns, same size as the rank text. */}
          <text
            x="5"
            y={CARD_H - 5}
            fontSize="11"
            fill={color}
            fontFamily="ui-sans-serif, system-ui"
          >
            {CROWN_CHAR}
          </text>
          <text
            x={CARD_W - 5}
            y="13"
            fontSize="11"
            textAnchor="end"
            fill={color}
            fontFamily="ui-sans-serif, system-ui"
          >
            {CROWN_CHAR}
          </text>
        </>
      )}
    </svg>
  );
}

function JokerSvg(props: CardProps & { rank: JokerRank }): JSX.Element {
  const { rank, selected, disabled, onClick, size } = props;
  const isBig = rank === 'big';
  const color = isBig ? '#b1303a' : '#15201a';
  const { w, h } = dims(size);
  const classes = [
    'card-svg',
    'joker',
    isBig ? 'big' : 'small',
    selected ? 'selected' : '',
    disabled ? 'disabled' : '',
    onClick ? 'clickable' : '',
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${CARD_W} ${CARD_H}`}
      className={classes}
      onClick={disabled ? undefined : onClick}
    >
      <rect
        x="0.5"
        y="0.5"
        width={CARD_W - 1}
        height={CARD_H - 1}
        rx="5"
        fill={isBig ? '#fff4e0' : '#fdfbf3'}
        stroke="#a89e83"
        strokeWidth="0.75"
      />
      <text
        x={CARD_W / 2}
        y="34"
        fontSize="22"
        textAnchor="middle"
        fill={color}
      >
        ★
      </text>
      <text
        x={CARD_W / 2}
        y="50"
        fontSize="8"
        fontWeight="700"
        textAnchor="middle"
        fill={color}
        letterSpacing="1"
      >
        JOKER
      </text>
      <text
        x={CARD_W / 2}
        y="62"
        fontSize="8"
        fontWeight="500"
        textAnchor="middle"
        fill={color}
        letterSpacing="0.5"
      >
        {isBig ? 'BIG' : 'sm'}
      </text>
    </svg>
  );
}

export function CardBackSvg({
  size,
}: {
  size?: 'normal' | 'small';
}): JSX.Element {
  const { w, h } = dims(size);
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${CARD_W} ${CARD_H}`}
      className="card-svg back"
    >
      <defs>
        <pattern
          id="cb-pat"
          width="6"
          height="6"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width="6" height="6" fill="#3a5240" />
          <rect width="3" height="6" fill="#4a6651" />
        </pattern>
      </defs>
      <rect
        x="0.5"
        y="0.5"
        width={CARD_W - 1}
        height={CARD_H - 1}
        rx="5"
        fill="url(#cb-pat)"
        stroke="#1f2d24"
        strokeWidth="0.75"
      />
      <rect
        x="6"
        y="6"
        width={CARD_W - 12}
        height={CARD_H - 12}
        rx="3"
        fill="none"
        stroke="#243528"
        strokeWidth="1"
        opacity="0.5"
      />
    </svg>
  );
}

export function CardFan({
  cards,
  selectedIds,
  onToggle,
}: {
  cards: ReadonlyArray<{ card: Card; id: string }>;
  selectedIds?: ReadonlyArray<string>;
  onToggle?: (id: string) => void;
}): JSX.Element {
  return (
    <div className="hand-svg">
      {cards.length === 0 ? (
        <span className="muted">No cards</span>
      ) : (
        cards.map(({ card, id }) => (
          <CardSvg
            key={id}
            card={card}
            selected={selectedIds?.includes(id)}
            onClick={onToggle ? () => onToggle(id) : undefined}
          />
        ))
      )}
    </div>
  );
}

export function PlayedCards({
  cards,
}: {
  cards: ReadonlyArray<{ card: Card; id: string }>;
}): JSX.Element | null {
  if (cards.length === 0) return null;
  return (
    <div className="played-svg">
      {cards.map(({ card, id }) => (
        <CardSvg key={id} card={card} size="small" />
      ))}
    </div>
  );
}

export function HandBacks({ count }: { count: number }): JSX.Element {
  // Show a compact stack of card backs representing the seat's hand size.
  const visible = Math.min(count, 8);
  return (
    <div className="hand-backs">
      {Array.from({ length: visible }).map((_, i) => (
        <CardBackSvg key={i} size="small" />
      ))}
      {count > visible && <span className="muted">+{count - visible}</span>}
    </div>
  );
}
