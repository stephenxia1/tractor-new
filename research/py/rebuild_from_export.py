"""Rebuilds a v2 checkpoint from the shipped float16 export.

    PYTHONPATH=py python py/rebuild_from_export.py --out checkpoints/r0_export.pt

A fallback for when the trained checkpoint itself is not to hand. The export
in packages/engine/src/rl/v2 is the policy network rounded to float16; it
leaves out the privileged critic, the snapshot pool and the optimizer state,
so a run resumed from this file starts with a fresh critic and an empty
pool. Measured on a small machine, a fresh critic cost about 13 points in
the first few minutes (RESULTS.md, Phase 6).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tractor_rl.model2 import NetConfig, TractorNet

RESEARCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SRC = RESEARCH_DIR.parent / "packages" / "engine" / "src" / "rl" / "v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default="checkpoints/r0_export.pt")
    args = ap.parse_args()

    src = Path(args.src)
    manifest = json.loads((src / "weights-v2.json").read_text())
    blob = np.frombuffer((src / "weights-v2.bin").read_bytes(), dtype=np.float16)
    net = TractorNet(NetConfig(**manifest["net_config"]))
    state = net.state_dict()
    for name, spec in manifest["tensors"].items():
        n = int(np.prod(spec["shape"]))
        arr = blob[spec["offset"]: spec["offset"] + n].astype(np.float32).reshape(spec["shape"])
        if tuple(state[name].shape) != tuple(arr.shape):
            raise SystemExit(f"{name}: export {arr.shape} vs net {tuple(state[name].shape)}")
        state[name] = torch.from_numpy(arr.copy())
    missing = sorted(set(state) - set(manifest["tensors"]))
    if missing:
        raise SystemExit(f"export lacks {missing}")
    net.load_state_dict(state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "v2", "model": net.state_dict(), "net_config": manifest["net_config"],
                "steps": manifest["steps"], "hands": manifest["hands"], "snapshots": []}, out)
    print(f"wrote {out}: {len(manifest['tensors'])} tensors, steps {manifest['steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
