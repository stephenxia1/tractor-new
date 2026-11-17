"""Reference positions and logits for the TypeScript port of the play
network (packages/engine/src/rl/zero/net.ts).

    python zero_net_parity.py --checkpoint CKPT --out-dir DIR --hands 40 --keep 0.08

Exports CKPT's weights to DIR/weights-zero.{bin,json} with
``tractor.export_ts``, then plays hands in the Python simulator (random
pre-play from ``gen.episode.generate``, uniformly random legal moves) and
writes a sample of the decisions to DIR/positions.jsonl: the packed
observation exactly as ``train/rollout._pack`` builds it for the network
(float16 already, so float32 holds it exactly), and the logits the
Python actor gives for it. The reference forward runs on the weights
rounded to float16, which is what the export stores, so the only
difference left for the TS side is the order of accumulation.

    npx tsx research/src/zeroNetParity.ts DIR

compares. ``--scale`` multiplies every weight matrix of a random
checkpoint, whose logits at initialisation are too flat to say much
about the argmax. ``--tiny`` writes a small-config fixture for the
engine's unit test instead (packages/engine/test/fixtures/zero-tiny).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tractor.export_ts import export  # noqa: E402
from tractor.gen.episode import generate  # noqa: E402
from tractor.nn.bury_net import BuryNet, BuryNetConfig  # noqa: E402
from tractor.nn.declare import DeclareConfig, DeclareNet  # noqa: E402
from tractor.nn.features import Cache  # noqa: E402
from tractor.nn.model import ActorCritic, ModelConfig, load_model  # noqa: E402
from tractor.train.buffer import obs_batch_from_arrays  # noqa: E402
from tractor.train.rollout import _InFlight, _pack  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def rounded(blob: dict) -> dict:
    """The checkpoint with every float tensor rounded through float16."""
    def r(sd: dict) -> dict:
        return {k: (v.to(torch.float16).to(torch.float32) if torch.is_floating_point(v) else v)
                for k, v in sd.items()}
    out = dict(blob)
    out["model"] = r(blob["model"])
    for part in ("declare", "bury"):
        if blob.get(part) is not None:
            out[part] = {**blob[part], "model": r(blob[part]["model"])}
    return out


def scaled(blob: dict, scale: float, seed: int) -> dict:
    """A random checkpoint made less flat: weight matrices times ``scale``,
    and the zero-initialised biases, norms and embeddings jittered."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for k, v in blob["model"].items():
        if not torch.is_floating_point(v):
            sd[k] = v
        elif k.endswith(".weight") and v.dim() == 2:
            sd[k] = v * scale
        elif k.endswith(".weight") and v.dim() == 1:  # LayerNorm gains
            sd[k] = 1.0 + 0.2 * torch.randn(v.shape, generator=g)
        else:
            sd[k] = v + 0.05 * torch.randn(v.shape, generator=g)
    return {**blob, "model": sd}


def tiny_blob(seed: int) -> dict:
    """A small-config checkpoint for the unit-test fixture."""
    torch.manual_seed(seed)
    cfg = ModelConfig(d_model=24, n_heads=3, n_layers_history=2, n_layers_query=2,
                      d_ff=40, scalar_d=8, critic_d=16, critic_heads=2, critic_d_ff=16,
                      critic_layers=1, belief_d=16, head_d=16)
    net = ActorCritic(cfg)
    dc = DeclareConfig(width=4, depth=1)
    bc = BuryNetConfig(d=8, layers=1, heads=2, value_width=8)
    return {
        "step": 0,
        "model": net.state_dict(),
        "model_config": asdict(cfg),
        "declare": {"model": DeclareNet(dc).state_dict(), "config": asdict(dc)},
        "bury": {"model": BuryNet(bc).state_dict(), "config": asdict(bc)},
    }


def positions(net: ActorCritic, cfg: ModelConfig, hands: int, keep: float,
              rng: np.random.Generator, every_first: int = 0):
    """Sampled decisions of ``hands`` random hands, with the actor's logits."""
    cache = Cache()
    for hand in range(hands):
        state, _info = generate(rng)
        step = 0
        while not state.done:
            packed = _pack(_InFlight(state=state, episode=0), cache, cfg)
            n = int(packed.candidate_mask.sum())
            if step < every_first or rng.random() < keep:
                obs = obs_batch_from_arrays(
                    packed.history[None], np.array([packed.history_len], dtype=np.int64),
                    packed.context[None], packed.unseen[None], packed.planes[None],
                    packed.cards[None], packed.candidates[None], packed.candidate_mask[None],
                    device="cpu")
                with torch.no_grad():
                    out = net.actor(obs)
                hl = int(packed.history_len)
                yield {
                    "hand": hand,
                    "step": step,
                    "history_len": hl,
                    "history": b64(packed.history[:hl].astype(np.float32)),
                    "context": b64(packed.context.astype(np.float32)),
                    "unseen": b64(packed.unseen.astype(np.float32)),
                    "planes": b64(packed.planes.astype(np.float32)),
                    "cards": b64(packed.cards.astype(np.float32)),
                    "num_candidates": n,
                    "candidates": b64(packed.candidates[:n].astype(np.float32)),
                    "logits": out.logits[0, :n].float().numpy().tolist(),
                }
            moves = packed.moves.moves
            state.step(moves[int(rng.integers(min(n, len(moves))))])
            step += 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hands", type=int, default=40)
    ap.add_argument("--keep", type=float, default=0.08)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    if args.tiny:
        blob = scaled(tiny_blob(args.seed), 8.0, args.seed)
    else:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.scale:
            blob = scaled(blob, args.scale, args.seed)
    blob = rounded(blob)
    os.makedirs(args.out_dir, exist_ok=True)
    export(blob, args.out_dir, source=args.checkpoint or "tiny")

    cfg = ModelConfig(**blob["model_config"])
    net = ActorCritic(cfg)
    load_model(net, blob["model"])
    net.eval()
    rng = np.random.default_rng(args.seed)
    name = "cases.jsonl" if args.tiny else "positions.jsonl"
    count = 0
    with open(os.path.join(args.out_dir, name), "w") as f:
        for rec in positions(net, cfg, args.hands, args.keep, rng):
            f.write(json.dumps(rec) + "\n")
            count += 1
    print(f"wrote {count} positions to {os.path.join(args.out_dir, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
