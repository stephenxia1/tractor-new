"""Checks the TypeScript inference path against the Python one.

    npx tsx src/dumpParity.ts --decisions 200 > /tmp/parity.jsonl
    python py/test_ts_parity.py /tmp/parity.jsonl

The weights were fitted to what `features2.py` produces. If the shipped
TypeScript featuriser disagrees by even one slot, nothing crashes -- the
bot simply plays a slightly different, worse game than the one the arena
measured. That failure is invisible without a test like this one.

Features are required to be *bit-identical*; both sides compute the same
arithmetic on the same integers, so there is no tolerance to spend. The Q
values are allowed a small tolerance: the exported weights are float16 and
the two forward passes accumulate in a different order.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tractor_rl import layout as L
from tractor_rl.features2 import (
    ACTION_DIM,
    HISTORY_STEPS,
    HISTORY_TOKEN_DIM,
    STATE_DIM,
    action_features,
    history_features,
    state_features,
)
from tractor_rl.model2 import NetConfig, TractorNet

# float16 weights plus a different summation order; 2e-3 is far below the
# gaps between candidate Q values that actually decide a move.
Q_TOLERANCE = 2e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="jsonl from src/dumpParity.ts")
    ap.add_argument("--checkpoint", default="checkpoints/v2_latest.pt")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.dump).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty dump")

    records = [base64.b64decode(r["record"]) for r in rows]
    states = np.stack([np.frombuffer(rec[: L.STATE_BYTES], dtype=np.uint8) for rec in records])

    py_state = state_features(states)
    py_tokens, py_mask = history_features(states)

    ts_state = np.array([r["state"] for r in rows], dtype=np.float32)
    ts_tokens = np.array([r["tokens"] for r in rows], dtype=np.float32).reshape(
        len(rows), HISTORY_STEPS, HISTORY_TOKEN_DIM)
    ts_mask = np.array([r["mask"] for r in rows], dtype=np.float32)

    failures = 0

    def check(name: str, a: np.ndarray, b: np.ndarray) -> None:
        nonlocal failures
        if a.shape != b.shape:
            print(f"FAIL {name}: shape {a.shape} vs {b.shape}")
            failures += 1
            return
        bad = np.argwhere(a != b)
        if len(bad):
            failures += 1
            print(f"FAIL {name}: {len(bad)} of {a.size} entries differ")
            for idx in bad[:5]:
                key = tuple(int(i) for i in idx)
                print(f"     at {key}: ts={b[key]!r} py={a[key]!r}")
        else:
            print(f"ok   {name}: {a.size:,} values bit-identical")

    check("state features", py_state, ts_state)
    check("history tokens", py_tokens, ts_tokens)
    check("history mask", py_mask, ts_mask)

    # Action features, decision by decision (candidate counts vary).
    cand_bytes, ok_actions, n_actions = L.CANDIDATE_BYTES, True, 0
    for i, (rec, row) in enumerate(zip(records, rows)):
        n = int(row["numCandidates"])
        if n == 0:
            continue
        block = np.frombuffer(rec[L.STATE_BYTES:], dtype=np.uint8).reshape(n, cand_bytes)
        py_act = action_features(block, states[i: i + 1], np.zeros(n, dtype=np.int64))
        ts_act = np.array(row["actions"], dtype=np.float32)
        n_actions += py_act.size
        if py_act.shape != ts_act.shape or not np.array_equal(py_act, ts_act):
            ok_actions = False
            print(f"FAIL action features at decision {i}")
            diff = np.argwhere(py_act != ts_act) if py_act.shape == ts_act.shape else []
            for idx in diff[:5]:
                key = tuple(int(x) for x in idx)
                print(f"     at {key}: ts={ts_act[key]!r} py={py_act[key]!r}")
            break
    if ok_actions:
        print(f"ok   action features: {n_actions:,} values bit-identical")
    else:
        failures += 1

    # Q values, if the dump carried them.
    if rows[0].get("scores") is not None:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net = TractorNet(NetConfig(**ck["net_config"]))
        net.load_state_dict(ck["model"])
        net.eval()

        worst, argmax_mismatch, total = 0.0, 0, 0
        with torch.no_grad():
            for i, (rec, row) in enumerate(zip(records, rows)):
                n = int(row["numCandidates"])
                if n == 0:
                    continue
                block = np.frombuffer(rec[L.STATE_BYTES:], dtype=np.uint8).reshape(n, cand_bytes)
                acts = action_features(block, states[i: i + 1], np.zeros(n, dtype=np.int64))
                out = net(
                    torch.from_numpy(py_state[i: i + 1]),
                    torch.from_numpy(acts),
                    torch.zeros(n, dtype=torch.long),
                    torch.tensor([int(row["role"])], dtype=torch.long),
                    torch.from_numpy(py_tokens[i: i + 1]),
                    torch.from_numpy(py_mask[i: i + 1]),
                )
                py_q = out["q"].numpy()
                ts_q = np.array(row["scores"], dtype=np.float32)
                worst = max(worst, float(np.abs(py_q - ts_q).max()))
                if int(py_q.argmax()) != int(ts_q.argmax()):
                    argmax_mismatch += 1
                total += 1

        status = "ok  " if worst <= Q_TOLERANCE else "FAIL"
        print(f"{status} Q values: max abs diff {worst:.2e} over {total} decisions "
              f"(tolerance {Q_TOLERANCE:.0e})")
        if worst > Q_TOLERANCE:
            failures += 1
        # The chosen move is what actually matters; report it separately.
        print(f"{'ok  ' if argmax_mismatch == 0 else 'WARN'} argmax: "
              f"{total - argmax_mismatch}/{total} decisions pick the same candidate")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("TypeScript inference matches the Python trainer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
