"""Exports a trained v2 checkpoint for the TypeScript inference path.

    python py/export_v2_weights.py --checkpoint checkpoints/v2_latest.pt

Writes a float16 blob plus a JSON manifest into
`packages/engine/src/rl/v2/`. Half precision because the file ships in the
Docker image and lives in a 256 MB VM; the weights are read once and
widened to float32 for the forward pass, so only storage is halved.

Only the tensors inference needs are exported: the belief head and the
privileged critic exist for training and would be dead weight here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

RESEARCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = RESEARCH_DIR.parent / "packages" / "engine" / "src" / "rl" / "v2"

# Everything the TS forward pass reads, in no particular order; the
# manifest carries the shapes so the loader never guesses. `belief_head` is
# included because the determinized sampler weights its worlds by it; it is
# the one "training-only" head that inference turned out to need.
WANTED_PREFIXES = (
    "stem.", "blocks.", "history.", "fuse.",
    "action_encoder.", "q_points.", "q_level.", "v_head.", "belief_head.",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v2_latest.pt")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("arch") != "v2":
        raise SystemExit(f"expected a v2 checkpoint, got arch={ck.get('arch')!r}")
    model = ck["model"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, offset, chunks = {}, 0, []
    skipped = []
    for name, tensor in model.items():
        if not name.startswith(WANTED_PREFIXES):
            skipped.append(name)
            continue
        arr = tensor.detach().cpu().float().numpy().astype(np.float16)
        tensors[name] = {"shape": list(arr.shape), "offset": offset}
        offset += arr.size
        chunks.append(arr.reshape(-1))

    blob = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float16)
    (out_dir / "weights-v2.bin").write_bytes(blob.tobytes())

    manifest = {
        "format": "float16",
        "count": int(blob.size),
        "steps": int(ck.get("steps", 0)),
        "hands": int(ck.get("hands", 0)),
        "net_config": ck.get("net_config", {}),
        "tensors": tensors,
    }
    (out_dir / "weights-v2.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"exported {len(tensors)} tensors, {blob.size:,} params "
          f"({blob.nbytes/1e6:.2f} MB fp16) from step {manifest['steps']:,}")
    print(f"  -> {out_dir / 'weights-v2.bin'}")
    if skipped:
        print(f"  skipped (training-only): {', '.join(sorted({s.split('.')[0] for s in skipped}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
