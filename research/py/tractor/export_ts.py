"""Export a checkpoint's play, declaration and burial networks for the
TypeScript bot (``packages/engine/src/rl/zero``).

    python -m tractor.export_ts CHECKPOINT OUT_DIR
    python -m tractor.export_ts --random OUT.pt   # a random-weight v5 checkpoint

Writes ``OUT_DIR/weights-zero.bin`` (every tensor as little-endian
float16, concatenated) and ``OUT_DIR/weights-zero.json``: the three
networks' configs and, for each tensor, its shape and element offset
into the .bin. Names are the checkpoint's state-dict keys under the
prefixes ``actor.`` (the play network's history encoder, query encoder
and candidate scorer: everything the policy's logits need), ``declare.``
and ``bury.``. The critics and auxiliary heads are left out; the bot
never reads them.

Only v5 networks are supported (GELU, the plain declaration MLP, the
``nn.TransformerEncoder`` burial net), because that is what the TS port
implements; anything else is refused rather than exported into a file
the port would misread.

``--random`` builds the same three networks at random initialisation and
saves them in the checkpoint layout, for parity tests that should not
depend on a trained checkpoint being reachable. The declaration net's
last layer starts at zero (``DeclareNet``), which would make every
declaration logit equal; it is re-drawn so the tests can see an argmax.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

from .nn.arch import saved_config
from .nn.bury_net import BuryNet, BuryNetConfig
from .nn.declare import DeclareConfig, DeclareNet
from .nn.model import ActorCritic, ModelConfig

ACTOR_PARTS = ("actor.history.", "actor.query.", "actor.scorer.")
FORMAT = 1


def _check_v5(model_cfg: dict, declare_cfg: dict, bury_cfg: dict) -> None:
    mc = ModelConfig(**model_cfg)
    if mc.ffn != "gelu":
        raise ValueError(f"the TS port implements GELU play networks, not {mc.ffn!r}")
    dc = DeclareConfig(**declare_cfg)
    if dc.residual or dc.ffn != "gelu":
        raise ValueError(f"the TS port implements the plain declaration MLP, not {dc}")
    bc = BuryNetConfig(**bury_cfg)
    if bc.ffn != "gelu":
        raise ValueError(f"the TS port implements the v5 burial net, not {bc}")


def export(blob: dict, out_dir: str, source: str = "") -> dict:
    """Write the .bin and .json for ``blob`` (a loaded checkpoint)."""
    model_cfg = dict(blob["model_config"])
    declare_cfg = saved_config(blob.get("declare"))
    bury_cfg = saved_config(blob.get("bury"))
    if blob.get("declare") is None or blob.get("bury") is None:
        raise ValueError("the checkpoint has no declaration or burial network")
    _check_v5(model_cfg, declare_cfg, bury_cfg)

    tensors: dict[str, torch.Tensor] = {}
    for k, v in blob["model"].items():
        if k.startswith(ACTOR_PARTS):
            tensors[k] = v
    for k, v in blob["declare"]["model"].items():
        if k.startswith("actor."):
            tensors[f"declare.{k}"] = v
    for k, v in blob["bury"]["model"].items():
        tensors[f"bury.{k}"] = v

    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {
        "format": FORMAT,
        "source": source,
        "step": int(blob.get("step", 0)),
        "model_config": asdict(ModelConfig(**model_cfg)),
        "declare_config": asdict(DeclareConfig(**declare_cfg)),
        "bury_config": asdict(BuryNetConfig(**bury_cfg)),
        "tensors": {},
    }
    offset = 0
    with open(os.path.join(out_dir, "weights-zero.bin"), "wb") as f:
        for name in sorted(tensors):
            t = tensors[name]
            if not torch.is_floating_point(t):
                continue  # buffers such as step counters
            arr = t.detach().to(torch.float32).cpu().numpy().astype("<f2")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} does not fit in float16")
            f.write(arr.tobytes())
            manifest["tensors"][name] = {"shape": list(arr.shape), "offset": offset}
            offset += arr.size
    manifest["elements"] = offset
    with open(os.path.join(out_dir, "weights-zero.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def random_checkpoint(path: str, seed: int = 0) -> None:
    """A v5 checkpoint at random initialisation, in the trained layout."""
    torch.manual_seed(seed)
    cfg = ModelConfig()
    net = ActorCritic(cfg)
    declare = DeclareNet(DeclareConfig())
    torch.nn.init.normal_(declare.actor[-1].weight, std=0.05)
    torch.nn.init.normal_(declare.actor[-1].bias, std=0.05)
    bury = BuryNet(BuryNetConfig())
    torch.save({
        "step": 0,
        "model": net.state_dict(),
        "model_config": asdict(cfg),
        "declare": {"model": declare.state_dict(), "config": asdict(DeclareConfig())},
        "bury": {"model": bury.state_dict(), "config": asdict(BuryNetConfig())},
    }, path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--random", metavar="OUT_PT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.random:
        random_checkpoint(args.random, args.seed)
        print(f"wrote {args.random}")
        return 0
    if not (args.checkpoint and args.out_dir):
        ap.error("CHECKPOINT and OUT_DIR are required")
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m = export(blob, args.out_dir, source=args.checkpoint)
    size = os.path.getsize(os.path.join(args.out_dir, "weights-zero.bin"))
    print(f"exported {len(m['tensors'])} tensors, {m['elements']:,} values, "
          f"{size / 1e6:.1f} MB, step {m['step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
