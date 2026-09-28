"""Named architectures, and rebuilding the pre-play networks a checkpoint holds.

``v5`` is what every run through v5-bury trained: GELU everywhere, the
flat critic, the 2 x 256 declaration MLP and the 2-layer, width-128
burial transformer. Every config class defaults to it, so a checkpoint
that predates saved configs rebuilds as ``v5``.

``v6`` is the widened set, chosen by the user on 2026-09-28:

* SwiGLU in place of every GELU (and the burial encoder's ReLU), at the
  same hidden widths, so each feed-forward holds half as many parameters
  again: actor 10.68M -> 13.27M;
* the critic reads the deal as card tokens (``model.CardCritic``): width
  256, 4 layers, 4 heads, feed-forward 1024; 0.48M -> 4.51M;
* declaration: a residual MLP, 3 blocks of width 512; 0.27M -> 5.0M;
* burial: 4 layers of width 256, value head 512; 0.42M -> 3.6M.

The play network's config has always been saved with a checkpoint
(``model_config``). The declaration and burial configs are saved beside
their weights from v6 on, as ``declare["config"]`` and ``bury["config"]``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .bury_net import BuryNet, BuryNetConfig
from .declare import DeclareConfig, DeclareNet

__all__ = ["ARCHES", "saved_config", "declare_net_from", "bury_net_from", "check_saved"]

ARCHES: dict[str, dict[str, dict[str, Any]]] = {
    "v5": {"model": {}, "declare_model": {}, "bury_model": {}},
    "v6": {
        "model": {
            "ffn": "swiglu",
            "critic_kind": "cards",
            "critic_d": 256,
            "critic_layers": 4,
            "critic_heads": 4,
            "critic_d_ff": 1024,
        },
        "declare_model": {"width": 512, "depth": 3, "residual": True, "ffn": "swiglu"},
        "bury_model": {"d": 256, "layers": 4, "heads": 4, "value_width": 512, "ffn": "swiglu"},
    },
}


def saved_config(part: dict | None) -> dict:
    """A checkpoint part's saved network config; ``{}`` (the v5 network)
    for one saved before configs were."""
    return {} if part is None else dict(part.get("config") or {})


def declare_net_from(part: dict) -> DeclareNet:
    """The declaration network a checkpoint's ``declare`` part holds."""
    net = DeclareNet(DeclareConfig(**saved_config(part)))
    net.load_state_dict(part["model"])
    return net


def bury_net_from(part: dict) -> BuryNet:
    """The burial network a checkpoint's ``bury`` part holds."""
    net = BuryNet(BuryNetConfig(**saved_config(part)))
    net.load_state_dict(part["model"])
    return net


def check_saved(name: str, part: dict | None, built: Any) -> None:
    """Refuse to resume a ``name`` network into a different architecture.

    ``built`` is the config the run constructed. Without this a mismatch
    surfaces as a wall of missing and unexpected keys, or -- where the
    shapes happen to agree -- not at all.
    """
    if part is None:
        return
    saved = type(built)(**saved_config(part))
    if saved != built:
        raise ValueError(
            f"the checkpoint's {name} network is {asdict(saved)}, "
            f"but this run builds {asdict(built)}"
        )
