"""Wire-layout constants, loaded from the spec.json that TypeScript emits.

Nothing here is hand-copied: `research/src/record.ts` is the single source
of truth and `npm run spec` regenerates spec.json. If the TS layout changes
without regenerating, LAYOUT_VERSION stops matching and this module raises
at import instead of silently misreading records.
"""

from __future__ import annotations

import json
from pathlib import Path

# The version this Python code was written against.
LAYOUT_VERSION = 2

RESEARCH_DIR = Path(__file__).resolve().parents[2]
SPEC_PATH = RESEARCH_DIR / "spec.json"

if not SPEC_PATH.exists():
    raise RuntimeError(
        f"{SPEC_PATH} is missing. Run `npm run spec` in {RESEARCH_DIR}."
    )

SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

if SPEC["version"] != LAYOUT_VERSION:
    raise RuntimeError(
        f"spec.json is version {SPEC['version']} but this module expects "
        f"{LAYOUT_VERSION}; regenerate spec.json or update layout.py."
    )

NUM_SLOTS: int = SPEC["numSlots"]
NUM_RANKS = 13
NUM_SUITS = 4

_H = SPEC["header"]
HEADER_BYTES: int = _H["bytes"]
H_VERSION: int = _H["version"]
H_SEAT: int = _H["seat"]
H_ROLE: int = _H["role"]
H_LANDLORD_REL: int = _H["landlordRel"]
H_TRUMP_RANK: int = _H["trumpRank"]
H_TRUMP_SUIT: int = _H["trumpSuit"]
H_TRICK_INDEX: int = _H["trickIndex"]
H_TRICK_POS: int = _H["trickPos"]
H_IS_LEAD: int = _H["isLead"]
H_HAND_SIZE: int = _H["handSize"]
H_WINNER_REL: int = _H["winnerRel"]
H_TABLE_POINTS: int = _H["tablePoints"]
H_ATTACKER_POINTS: int = _H["attackerPoints"]
H_CARDS_SEEN: int = _H["cardsSeen"]
H_NUM_CANDIDATES: int = _H["numCandidates"]
H_I_AM_ATTACKER: int = _H["iAmAttacker"]
H_VOID_MASK: int = _H["voidMask"]
H_GAME_ID: int = _H["gameId"]
H_HAND_ID: int = _H["handId"]

_P = SPEC["planes"]
NUM_PLANES: int = _P["count"]
PLANES_BYTES: int = _P["bytes"]
PLANE_MY_HAND: int = _P["myHand"]
PLANE_PLAYED_BASE: int = _P["playedBase"]
PLANE_TRICK_BASE: int = _P["trickBase"]
PLANE_MY_KITTY: int = _P["myKitty"]

STATE_BYTES: int = SPEC["stateBytes"]

_C = SPEC["candidate"]
CANDIDATE_HEADER_BYTES: int = _C["headerBytes"]
CANDIDATE_BYTES: int = _C["bytes"]
MAX_CANDIDATE_CARDS: int = _C["maxCards"]
NO_CARD: int = _C["noCard"]
NUM_SHAPES: int = _C["shapes"]["count"]
FLAG_ALL_TRUMP: int = _C["flags"]["allTrump"]
FLAG_ALL_LED_SUIT: int = _C["flags"]["allLedSuit"]
FLAG_HAS_POINTS: int = _C["flags"]["hasPoints"]
FLAG_BEATS_WINNER: int = _C["flags"]["beatsWinner"]

_E = SPEC["episode"]
EPISODE_BYTES: int = _E["bytes"]
E_HAND_ID: int = _E["handId"]
E_LANDLORD_SEAT: int = _E["landlordSeat"]
E_NUM_TRICKS: int = _E["numTricks"]
E_ATTACKER_POINTS: int = _E["attackerPoints"]
E_LEVEL_DELTA: int = _E["levelDelta"]
E_DEFENDERS_WON: int = _E["defendersWon"]
E_TRUMP_RANK: int = _E["trumpRank"]
E_TRUMP_SUIT: int = _E["trumpSuit"]
E_TRICKS_BASE: int = _E["tricksBase"]
MAX_TRICKS: int = _E["maxTricks"]
E_KITTY_POINTS: int = _E["kittyPoints"]
E_GAME_ID: int = _E["gameId"]

_F = SPEC["frame"]
FRAME_HEADER_BYTES: int = _F["headerBytes"]
BATCH_HEADER_BYTES: int = _F["batchHeaderBytes"]
ACTION_BYTES: int = _F["actionBytes"]

# Role ids (encode.ts).
ROLE_LANDLORD = 0
ROLE_DEFENDER = 1
ROLE_ATTACKER = 2
NUM_ROLES = 3

# Trump-suit id used when the hand is no-trump (joker declaration).
NO_TRUMP_SUIT = 4
