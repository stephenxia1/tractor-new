import type { Card } from './types';
import type { Seat } from './state';

// All events the reducer accepts. Events come from players (declare, play_cards,
// pick_up_kitty, replace_kitty) or from the server (start_hand, release_lockout).
export type GameEvent =
  | { readonly kind: 'start_hand'; readonly seed: number }
  | { readonly kind: 'declare'; readonly seat: Seat; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'release_lockout'; readonly seed?: number } // seed used if random landlord needed
  | { readonly kind: 'pick_up_kitty'; readonly seat: Seat }
  | { readonly kind: 'replace_kitty'; readonly seat: Seat; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'play_cards'; readonly seat: Seat; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'advance_trick' };

export type GameEventKind = GameEvent['kind'];
