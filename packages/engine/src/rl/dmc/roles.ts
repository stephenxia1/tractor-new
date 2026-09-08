// Role assignment for the 3-network DMC architecture.
//
// Three Q-networks: landlord, defender (landlord's partner), and attacker
// (shared across both attacking seats). Position relative to the landlord
// (left/right) is provided as a feature so the shared attacker net can
// specialize internally.

import { teamOf, type GameState, type Seat } from '../../state';

export type Role = 'landlord' | 'defender' | 'attacker';

// Where a non-landlord seat sits relative to the landlord (clockwise turn
// order). Position 1 plays immediately after landlord; position 3 plays
// just before. Defender (partner) is always position 2.
export type PositionFeature = {
  readonly role: Role;
  // 1, 2, or 3 — turn distance from landlord in clockwise order.
  readonly distanceFromLandlord: number;
  // For attackers: 'left' (plays right after landlord) or 'right' (plays
  // just before landlord). Defender always 'across'. Landlord: 'self'.
  readonly side: 'self' | 'left' | 'across' | 'right';
};

export function roleFor(state: GameState, seat: Seat): PositionFeature {
  const hand = state.currentHand;
  if (hand === null || hand.landlord === null) {
    // Landlord unset (between hands or pre-declare): treat seat as attacker
    // with neutral side. Training never queries roles in this state, but
    // defaulting protects callers.
    return { role: 'attacker', distanceFromLandlord: 0, side: 'self' };
  }
  const landlord = hand.landlord;
  if (seat === landlord) {
    return { role: 'landlord', distanceFromLandlord: 0, side: 'self' };
  }
  const numPlayers = state.config.numPlayers;
  // Clockwise distance: (seat - landlord) mod numPlayers.
  const dist = ((seat - landlord) + numPlayers) % numPlayers;
  if (teamOf(state, seat) === teamOf(state, landlord)) {
    // Same team => partner (defender). For 4-player, dist is always 2.
    return { role: 'defender', distanceFromLandlord: dist, side: 'across' };
  }
  // Attacker. For 4-player: dist=1 → left (plays next), dist=3 → right.
  const side: PositionFeature['side'] = dist === 1 ? 'left' : 'right';
  return { role: 'attacker', distanceFromLandlord: dist, side };
}
