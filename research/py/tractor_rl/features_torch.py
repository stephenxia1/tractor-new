"""Featurisation in torch, so it can run on the device that trains.

The numpy featuriser is the twin of this file and remains the reference;
`test_torch_features.py` asserts the two agree bit for bit. Keeping both is
deliberate -- numpy is what the actor loop uses on CPU-only boxes, and this
is what a GPU learner uses so the accelerator is not left waiting on the
host.

Two things motivated the port, both measured:

* on a T4 a learner step is 12.1 ms while featurising the same batch in
  numpy costs 18-35 ms, so featurisation became two to three times the
  cost of the step it feeds
* raw wire records are ~980 bytes against ~10 KB of features, so moving
  the work to the device also cuts the host-to-device transfer ~10x

Everything here takes the *raw record bytes* as a uint8 tensor and derives
features on whatever device that tensor lives on.
"""

from __future__ import annotations

import numpy as np
import torch

from . import layout as L
from .encoding import CAPACITY, ENC_OF_RAW, NUM_ENC_SLOTS
from .features2 import ACTION_DIM, HISTORY_TOKEN_DIM, SCALAR_DIM, STATE_DIM

HISTORY_STEPS = L.HISTORY_STEPS
NUM_ROLES = L.NUM_ROLES
NUM_RANKS = 13

# Tables are small and constant; cache one copy per device.
_TABLES: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(enc_of_raw [13,5,54] int64, capacity [13,5,66] float32) on `device`."""
    cached = _TABLES.get(device)
    if cached is None:
        enc = torch.from_numpy(np.ascontiguousarray(ENC_OF_RAW)).to(device).long()
        cap = torch.from_numpy(np.ascontiguousarray(CAPACITY)).to(device).float()
        cached = (enc, cap)
        _TABLES[device] = cached
    return cached


def _u8(states: torch.Tensor, offset: int) -> torch.Tensor:
    return states[:, offset].long()


def _u16(states: torch.Tensor, offset: int) -> torch.Tensor:
    return states[:, offset].long() + states[:, offset + 1].long() * 256


def _trump(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _u8(states, L.H_TRUMP_RANK), _u8(states, L.H_TRUMP_SUIT)


def _encode_planes(raw: torch.Tensor, enc_idx: torch.Tensor) -> torch.Tensor:
    """[B, P, 54] raw counts -> [B, P, 66], scattering into chain positions.

    A scatter-add rather than a matmul, for the same reason as the numpy
    side: this is a permutation with collisions (the off-suit trump ranks
    share a slot), not a linear map worth materialising.
    """
    b, p, _ = raw.shape
    out = raw.new_zeros((b, p, NUM_ENC_SLOTS))
    idx = enc_idx.unsqueeze(1).expand(b, p, L.NUM_SLOTS)
    return out.scatter_add_(2, idx, raw)


def _one_hot_into(dest: torch.Tensor, at: int, values: torch.Tensor, width: int) -> int:
    rows = torch.arange(dest.shape[0], device=dest.device)
    dest[rows, at + values.clamp(0, width - 1)] = 1.0
    return at + width


def state_features(states: torch.Tensor) -> torch.Tensor:
    """[B, state_bytes] uint8 -> [B, STATE_DIM] float32. Policy-legal only."""
    b = states.shape[0]
    device = states.device
    if b == 0:
        return torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device)

    enc_tab, cap_tab = tables(device)
    rank, suit = _trump(states)
    enc_idx = enc_tab[rank, suit]                    # [B, 54]

    raw = states[:, L.HEADER_BYTES:L.HEADER_BYTES + L.PLANES_BYTES]
    raw = raw.reshape(b, L.NUM_PLANES, L.NUM_SLOTS).float()
    planes = _encode_planes(raw, enc_idx)            # [B, 10, 66]

    my_hand = planes[:, L.PLANE_MY_HAND]
    played = planes[:, L.PLANE_PLAYED_BASE:L.PLANE_PLAYED_BASE + 4]
    trick = planes[:, L.PLANE_TRICK_BASE:L.PLANE_TRICK_BASE + 4]
    kitty = planes[:, L.PLANE_MY_KITTY]

    capacity = cap_tab[rank, suit]                   # [B, 66]
    unseen = (capacity - my_hand - played.sum(dim=1) - kitty).clamp_min(0.0)

    parts = [
        (my_hand >= 1).float(),
        (my_hand >= 2).float(),
        (unseen >= 1).float(),
        (unseen >= 2).float(),
        played.reshape(b, -1) * 0.5,
        trick.reshape(b, -1) * 0.5,
        kitty * 0.5,
    ]
    plane_feats = torch.cat(parts, dim=1)

    scalars = torch.zeros((b, SCALAR_DIM), dtype=torch.float32, device=device)
    at = 0
    at = _one_hot_into(scalars, at, _u8(states, L.H_ROLE), NUM_ROLES)
    at = _one_hot_into(scalars, at, _u8(states, L.H_LANDLORD_REL), 4)
    at = _one_hot_into(scalars, at, _u8(states, L.H_TRICK_POS), 4)
    scalars[:, at] = states[:, L.H_IS_LEAD].float()
    at += 1
    winner = _u8(states, L.H_WINNER_REL)
    at = _one_hot_into(scalars, at, torch.where(winner > 3, 4, winner), 5)
    scalars[:, at] = states[:, L.H_I_AM_ATTACKER].float()
    at += 1

    scalars[:, at] = states[:, L.H_HAND_SIZE].float() / 25.0
    scalars[:, at + 1] = states[:, L.H_TRICK_INDEX].float() / 25.0
    scalars[:, at + 2] = _u16(states, L.H_ATTACKER_POINTS).float() / 200.0
    scalars[:, at + 3] = states[:, L.H_TABLE_POINTS].float() / 40.0
    scalars[:, at + 4] = _u16(states, L.H_CARDS_SEEN).float() / 108.0
    at += 5

    # Void mask, rotated so suit 0 is the trump suit (absolute under no-trump).
    void = states[:, L.H_VOID_MASK:L.H_VOID_MASK + 4].long()      # [B, 4]
    offs = torch.arange(4, device=device)
    rot = torch.where(suit.unsqueeze(1) < 4,
                      (suit.unsqueeze(1) + offs) % 4,
                      offs.expand(b, 4))                          # [B, 4]
    suit_bits = (void.unsqueeze(2) >> rot.unsqueeze(1)) & 1        # [B, 4, 4]
    trump_bits = ((void >> 4) & 1).unsqueeze(2)                    # [B, 4, 1]
    scalars[:, at:at + 20] = torch.cat([suit_bits, trump_bits], dim=2).reshape(b, 20).float()
    at += 20

    at = _one_hot_into(scalars, at, rank, NUM_RANKS)
    at = _one_hot_into(scalars, at, suit, 5)
    scalars[:, at] = 1.0
    at += 1
    assert at == SCALAR_DIM, (at, SCALAR_DIM)

    return torch.cat([plane_feats, scalars], dim=1)


def history_features(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, W, HISTORY_TOKEN_DIM] tokens and a [B, W] validity mask."""
    b = states.shape[0]
    device = states.device
    if b == 0:
        return (torch.zeros((0, HISTORY_STEPS, HISTORY_TOKEN_DIM), device=device),
                torch.zeros((0, HISTORY_STEPS), device=device))

    enc_tab, _ = tables(device)
    rank, suit = _trump(states)
    enc_idx = enc_tab[rank, suit]                    # [B, 54]

    base = L.HEADER_BYTES + L.PLANES_BYTES
    block = states[:, base:base + L.HISTORY_BYTES]
    block = block.reshape(b, HISTORY_STEPS, L.HISTORY_STEP_BYTES)
    n_cards = block[:, :, 0].float()
    seat_rel = block[:, :, 1].long()
    trick_idx = block[:, :, 2].float()
    pos = block[:, :, 3].long()
    slots = block[:, :, 4:4 + L.HISTORY_MAX_CARDS].long()          # [B, W, 8]

    valid = slots != L.NO_CARD
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    # Each card straight to its chain position, then counted there.
    enc_slots = torch.gather(enc_idx.unsqueeze(1).expand(b, HISTORY_STEPS, L.NUM_SLOTS),
                             2, safe)
    counts = torch.zeros((b, HISTORY_STEPS, NUM_ENC_SLOTS), dtype=torch.float32, device=device)
    counts.scatter_add_(2, enc_slots, valid.float())

    tokens = torch.zeros((b, HISTORY_STEPS, HISTORY_TOKEN_DIM),
                         dtype=torch.float32, device=device)
    tokens[:, :, :NUM_ENC_SLOTS] = counts * 0.5
    bb = torch.arange(b, device=device).unsqueeze(1).expand(b, HISTORY_STEPS)
    ww = torch.arange(HISTORY_STEPS, device=device).unsqueeze(0).expand(b, HISTORY_STEPS)
    tokens[bb, ww, NUM_ENC_SLOTS + seat_rel.clamp(0, 3)] = 1.0
    tokens[bb, ww, NUM_ENC_SLOTS + 4 + pos.clamp(0, 3)] = 1.0
    tokens[:, :, NUM_ENC_SLOTS + 8] = trick_idx / 25.0
    tokens[:, :, NUM_ENC_SLOTS + 9] = n_cards / 8.0

    lengths = _u8(states, L.H_HISTORY_LEN)
    mask = (torch.arange(HISTORY_STEPS, device=device).unsqueeze(0)
            < lengths.unsqueeze(1)).float()
    tokens *= mask.unsqueeze(2)
    return tokens, mask


def action_features(cands: torch.Tensor, states: torch.Tensor,
                    owner: torch.Tensor) -> torch.Tensor:
    """[N, CANDIDATE_BYTES] uint8 -> [N, ACTION_DIM] float32."""
    n = cands.shape[0]
    device = cands.device
    if n == 0:
        return torch.zeros((0, ACTION_DIM), dtype=torch.float32, device=device)

    enc_tab, _ = tables(device)
    rank, suit = _trump(states)
    enc_idx = enc_tab[rank[owner], suit[owner]]      # [N, 54]

    slots = cands[:, L.CANDIDATE_HEADER_BYTES:].long()
    valid = slots != L.NO_CARD
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    enc_slots = torch.gather(enc_idx, 1, safe)
    counts = torch.zeros((n, NUM_ENC_SLOTS), dtype=torch.float32, device=device)
    counts.scatter_add_(1, enc_slots, valid.float())

    out = torch.zeros((n, ACTION_DIM), dtype=torch.float32, device=device)
    out[:, :NUM_ENC_SLOTS] = (counts >= 1).float()
    out[:, NUM_ENC_SLOTS:2 * NUM_ENC_SLOTS] = (counts >= 2).float()
    at = 2 * NUM_ENC_SLOTS
    shape = cands[:, 1].long()
    at = _one_hot_into(out, at, shape, L.NUM_SHAPES)
    flags = cands[:, 2].long()
    out[:, at + 0] = ((flags & L.FLAG_ALL_TRUMP) != 0).float()
    out[:, at + 1] = ((flags & L.FLAG_ALL_LED_SUIT) != 0).float()
    out[:, at + 2] = ((flags & L.FLAG_HAS_POINTS) != 0).float()
    out[:, at + 3] = ((flags & L.FLAG_BEATS_WINNER) != 0).float()
    at += 4
    out[:, at] = cands[:, 0].float() / 8.0
    out[:, at + 1] = cands[:, 3].float() / 30.0
    return out


def privileged_features(states: torch.Tensor) -> torch.Tensor:
    """Training only: the other seats' hands and the kitty."""
    b = states.shape[0]
    device = states.device
    if states.shape[1] < L.STATE_BYTES_PRIVILEGED:
        raise ValueError("record carries no privileged block")
    enc_tab, _ = tables(device)
    rank, suit = _trump(states)
    base = L.HEADER_BYTES + L.PLANES_BYTES + L.HISTORY_BYTES
    raw = states[:, base:base + L.PRIVILEGED_BYTES]
    raw = raw.reshape(b, L.PRIVILEGED_PLANES, L.NUM_SLOTS).float()
    enc = _encode_planes(raw, enc_tab[rank, suit])
    # Halved to match the policy planes' scale -- the numpy twin does the
    # same, and the critic was trained against that scaling.
    return (enc * 0.5).reshape(b, -1)


def belief_targets(states: torch.Tensor, priv: torch.Tensor | None = None) -> torch.Tensor:
    """[B, 3 * 66] in {0,1}: does relative seat r hold a card at position p."""
    if priv is None:
        priv = privileged_features(states)
    b = states.shape[0]
    priv = priv.reshape(b, L.PRIVILEGED_PLANES, NUM_ENC_SLOTS)
    # `> 0` is scale-invariant, so this holds whether or not priv was halved.
    return (priv[:, :3] > 0).float().reshape(b, -1)
