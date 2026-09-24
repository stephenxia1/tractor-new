"""The burial policy: eight picks, one card each.

The landlord holds 33 cards and puts 8 back (rules.md §6). As one choice
that is C(33, 8), about fourteen million burials; as eight picks of one
card from what is still held it is a choice among at most 33, eight
times, which a policy can be trained on. The order of the picks carries
no meaning -- the burial is the set -- and nothing here pretends it does.

Each pick reads one token per card type: the card's static features in
this trump context (``features.static_card_features``) and how many
copies are held, already buried, and were in the 33. A context vector
carries the level, the trump suit, which pick this is, and the standing
declaration. A small set encoder scores every card type; cards not held
are masked out. The value head reads the pooled tokens and, because it
only ever reduces the variance of a gradient, the other three hands.

Trained in the pre-play phase with play frozen: the return of all eight
picks is the privileged critic's value at the play start they produce,
from the landlord's seat (``train/bury_loop.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..rules.cards import NUM_TYPES
from .features import D_STATIC
from .model import MASK_LOGIT

__all__ = [
    "BURY_CARD_DIM",
    "BURY_CTX_DIM",
    "BURY_PRIV_DIM",
    "BURY_STEPS",
    "BuryNetConfig",
    "BuryNet",
]

BURY_STEPS = 8
# Static features, then held / buried / in-the-33 counts over the decks.
BURY_CARD_DIM = D_STATIC + 3
# Level (13), trump suit or no-trump (5), pick (8), standing tier (5),
# declarer relative to the landlord (4, all zero for a kitty flip).
BURY_CTX_DIM = 13 + 5 + BURY_STEPS + 5 + 4
# The other three hands, in order after the landlord.
BURY_PRIV_DIM = 3 * NUM_TYPES


@dataclass(frozen=True)
class BuryNetConfig:
    d: int = 128
    layers: int = 2
    heads: int = 4
    value_width: int = 256


class BuryNet(nn.Module):
    def __init__(self, cfg: BuryNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or BuryNetConfig()
        self.card_in = nn.Linear(BURY_CARD_DIM, cfg.d)
        self.ctx_in = nn.Linear(BURY_CTX_DIM, cfg.d)
        self.card_embed = nn.Parameter(torch.zeros(NUM_TYPES, cfg.d))
        layer = nn.TransformerEncoderLayer(
            cfg.d, cfg.heads, dim_feedforward=2 * cfg.d, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d)
        self.score = nn.Linear(cfg.d, 1)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d + BURY_PRIV_DIM, cfg.value_width),
            nn.GELU(),
            nn.Linear(cfg.value_width, cfg.value_width),
            nn.GELU(),
            nn.Linear(cfg.value_width, 1),
        )
        nn.init.normal_(self.card_embed, std=0.02)
        # A uniform first policy over the held cards, so early training
        # explores rather than committing to whatever init favoured.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def encode(self, cards: Tensor, ctx: Tensor) -> Tensor:
        x = self.card_in(cards) + self.card_embed + self.ctx_in(ctx).unsqueeze(1)
        return self.norm(self.encoder(x))  # [B, 54, d]

    def forward(
        self, cards: Tensor, ctx: Tensor, mask: Tensor, privileged: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """-> (logits [B, 54], log_probs [B, 54], value [B])."""
        h = self.encode(cards, ctx)
        raw = self.score(h).squeeze(-1)
        logits = torch.where(mask, raw, torch.full_like(raw, MASK_LOGIT))
        pooled = h.mean(dim=1)
        value = self.value_head(torch.cat((pooled, privileged), dim=-1)).squeeze(-1)
        return logits, F.log_softmax(logits, dim=-1), value
