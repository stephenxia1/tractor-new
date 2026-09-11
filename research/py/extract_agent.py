"""Pull one agent out of a population checkpoint as a standalone v2 file.

A population checkpoint holds N agents in one payload; `arena.py` and
`ladder.py` expect the single-network shape `train2.py` writes. This
converts between them, so evaluating a population member needs no special
case anywhere downstream.

    PYTHONPATH=py python py/extract_agent.py checkpoints/pop8_latest.pt --agent 3
    PYTHONPATH=py python py/extract_agent.py checkpoints/pop8_latest.pt --all

By default the *acting* weights are written -- what the agent actually
played with, and so what its measured strength belongs to. `--live` takes
the trained weights instead, which are up to one iteration ahead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

RESEARCH_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RESEARCH_DIR / "checkpoints"


def extract(state: dict, idx: int, live: bool) -> dict:
    blob = state["agents"][idx]
    key = "model" if live else "acting"
    return {
        "arch": "v2",
        "model": blob.get(key, blob["model"]),
        "critic": blob["critic"],
        "net_config": state.get("net_config", {"history": "attn"}),
        "steps": blob.get("steps", 0),
        # The population counts decisions per agent rather than hands per run;
        # carry it through under the name the loader prints so the provenance
        # of an evaluated file is not silently blank.
        "hands": blob.get("decisions", 0),
        "snapshots": [],
        "population": {"agent": idx, "iteration": state.get("iteration", 0),
                       "weights": key},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--agent", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="take the training weights rather than the acting ones")
    ap.add_argument("--out-prefix", type=str, default="")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    state = torch.load(src, map_location="cpu", weights_only=False)
    if "agents" not in state:
        raise SystemExit(f"{src} is not a population checkpoint")
    n = len(state["agents"])
    if args.agent is None and not args.all:
        raise SystemExit(f"{src} holds {n} agents; pass --agent I or --all")

    prefix = args.out_prefix or src.stem.replace("_latest", "").replace("_final", "")
    targets = range(n) if args.all else [args.agent]
    for i in targets:
        if not 0 <= i < n:
            raise SystemExit(f"agent {i} is out of range (0..{n-1})")
        payload = extract(state, i, args.live)
        out = CHECKPOINT_DIR / f"{prefix}_a{i}.pt"
        torch.save(payload, out)
        print(f"{out.name}: agent {i}, {payload['population']['weights']} weights, "
              f"steps {payload['steps']}, {payload['hands']} decisions")


if __name__ == "__main__":
    main()
