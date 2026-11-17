"""Checks the TypeScript zero bot's declarations and burials against the
Python networks they were exported from.

    python test_zero_preplay_parity.py make OUT_DIR      # checkpoints + exports
    npx tsx ../src/zeroPrePlayParity.ts --weights OUT_DIR/full > /tmp/preplay.jsonl
    python test_zero_preplay_parity.py check /tmp/preplay.jsonl OUT_DIR/full.pt

``make`` writes two random-weight checkpoints and their exports: ``full``
at the v5 sizes the trained bot uses, and ``tiny`` (small widths) for the
committed vitest fixture. The declaration net's last layer and the burial
net's score head start at zero in training (a uniform first policy), which
would make every argmax pick the first legal action; both are re-drawn
here so a greedy choice is a real test.

``check`` recomputes every dumped decision with the evaluation path the
measured bot used (``eval/bots.py``: ``_declare_choice``, ``_bury_choice``
with the burial network) on the checkpoint's weights rounded to float16,
as the export stores them. Features must be bit-identical -- both sides
compute the same float32 arithmetic on the same integers. Logits get a
small tolerance (the two sides accumulate in a different order and
precision). Choices must agree, except where the Python logits of the two
contenders are themselves within that tolerance.

``check ... --fixture OUT.json --limit N`` also writes N decisions of
each kind, spread through the dump, with Python's results, for the
vitest test.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

from tractor.eval.bots import _bury_choice, _declare_choice
from tractor.export_ts import export
from tractor.nn.arch import bury_net_from, declare_net_from
from tractor.nn.bury_net import BuryNet, BuryNetConfig
from tractor.nn.declare import DeclareConfig, DeclareNet
from tractor.nn.features import Cache
from tractor.nn.model import ActorCritic, ModelConfig
from tractor.rules.cards import Ctx
from tractor.sim.bury import _step_inputs
from tractor.sim.deal import declare_features

LOGIT_TOL = 2e-4

TINY_MODEL = dict(d_model=32, n_layers_history=1, n_layers_query=1, n_heads=2, d_ff=64,
                  critic_d=32, critic_layers=1, critic_heads=2, critic_d_ff=64,
                  belief_d=32, scalar_d=16, head_d=32)
TINY_DECLARE = dict(width=16, depth=2)
TINY_BURY = dict(d=16, layers=1, heads=2, value_width=16)


def _checkpoint(path: str, seed: int, model: dict, declare_cfg: dict, bury_cfg: dict) -> None:
    torch.manual_seed(seed)
    cfg = ModelConfig(**model)
    net = ActorCritic(cfg)
    dc = DeclareConfig(**declare_cfg)
    declare = DeclareNet(dc)
    torch.nn.init.normal_(declare.actor[-1].weight, std=0.05)
    torch.nn.init.normal_(declare.actor[-1].bias, std=0.05)
    bc = BuryNetConfig(**bury_cfg)
    bury = BuryNet(bc)
    torch.nn.init.normal_(bury.score.weight, std=0.05)
    torch.nn.init.normal_(bury.score.bias, std=0.05)
    torch.save({
        "step": 0,
        "model": net.state_dict(),
        "model_config": asdict(cfg),
        "declare": {"model": declare.state_dict(), "config": asdict(dc)},
        "bury": {"model": bury.state_dict(), "config": asdict(bc)},
    }, path)


def make(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, model, dcfg, bcfg in (("full", {}, {}, {}),
                                    ("tiny", TINY_MODEL, TINY_DECLARE, TINY_BURY)):
        pt = os.path.join(out_dir, f"{name}.pt")
        _checkpoint(pt, 7, model, dcfg, bcfg)
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        export(blob, os.path.join(out_dir, name), source=f"random:{name}")
        if name == "tiny":
            _strip_actor(os.path.join(out_dir, name))
        print(f"wrote {pt} and {os.path.join(out_dir, name)}/")


def _strip_actor(d: str) -> None:
    """Keep only the declaration and burial tensors of an export, so the
    committed fixture stays small."""
    with open(os.path.join(d, "weights-zero.json")) as fh:
        m = json.load(fh)
    flat = np.fromfile(os.path.join(d, "weights-zero.bin"), dtype="<f2")
    keep, parts, offset = {}, [], 0
    for name in sorted(m["tensors"]):
        if not name.startswith(("declare.", "bury.")):
            continue
        spec = m["tensors"][name]
        size = int(np.prod(spec["shape"]))
        parts.append(flat[spec["offset"]: spec["offset"] + size])
        keep[name] = {"shape": spec["shape"], "offset": offset}
        offset += size
    m["tensors"], m["elements"] = keep, offset
    np.concatenate(parts).astype("<f2").tofile(os.path.join(d, "weights-zero.bin"))
    with open(os.path.join(d, "weights-zero.json"), "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")


def _f32(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def _nets(path: str) -> tuple:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    dnet = declare_net_from(blob["declare"]).eval()
    bnet = bury_net_from(blob["bury"]).eval()
    with torch.no_grad():
        for p in list(dnet.parameters()) + list(bnet.parameters()):
            p.copy_(p.half().float())
    return dnet, bnet


def _tie(logits: np.ndarray, allowed: np.ndarray, a: int, b: int) -> bool:
    """Whether actions ``a`` and ``b`` are within tolerance of the best."""
    return bool(allowed[a] and allowed[b] and abs(logits[a] - logits[b]) <= LOGIT_TOL)


def check(dump: str, ckpt: str, fixture: str | None, limit: int) -> int:
    dnet, bnet = _nets(ckpt)
    cache = Cache()
    n_dec = n_bury = 0
    feat_bad = choice_bad = choice_tie = 0
    max_dlogit = max_blogit = 0.0
    bury_feat_bad = bury_bad = bury_tie_steps = 0
    picks_total = picks_same = 0
    fix_dec: list = []
    fix_bury: list = []
    with open(dump) as fh:
        for line in fh:
            r = json.loads(line)
            if r["t"] == "declare":
                n_dec += 1
                mask = np.asarray(r["mask"], dtype=bool)
                obs = declare_features(
                    revealed=np.asarray(r["hand_counts"], dtype=np.int8), num_decks=2,
                    level_rank=r["level_rank"], landlord=r["landlord"], seat=r["seat"],
                    dealt=r["dealt"], total=100, closing=bool(r["closing"]),
                    standing_tier=r["standing_tier"], standing_suit=r["standing_suit"],
                    standing_seat=r["standing_seat"], mask=mask,
                )
                ts = _f32(r["features"])
                if not np.array_equal(obs.view(np.uint32), ts.view(np.uint32)):
                    feat_bad += 1
                with torch.no_grad():
                    raw = dnet.actor(torch.from_numpy(obs[None]))[0].numpy().astype(np.float64)
                tsl = np.asarray(r["logits"], dtype=np.float64)
                max_dlogit = max(max_dlogit, float(np.abs(raw - tsl).max()))
                msg = {"hand": r["hand_counts"], "mask": r["mask"], "level_rank": r["level_rank"],
                       "landlord": r["landlord"], "seat": r["seat"], "dealt": r["dealt"],
                       "closing": r["closing"], "standing_tier": r["standing_tier"],
                       "standing_suit": r["standing_suit"], "standing_seat": r["standing_seat"]}
                py = _declare_choice(dnet, msg, "cpu")
                if py != r["choice"]:
                    if _tie(raw, mask, py, r["choice"]):
                        choice_tie += 1
                    else:
                        choice_bad += 1
                if fixture:
                    fix_dec.append({**{k: r[k] for k in (
                        "seat", "level_rank", "landlord", "dealt", "total", "closing",
                        "standing_tier", "standing_suit", "standing_seat", "hand_counts", "mask")},
                        "features": r["features"], "logits": raw.tolist(), "choice": py})
            elif r["t"] == "bury":
                n_bury += 1
                pool = np.asarray(r["pool"], dtype=np.int8)
                msg = {"pool": r["pool"], "level_rank": r["level_rank"],
                       "trump_suit": r["trump_suit"], "seat": r["seat"],
                       "standing_tier": r["standing_tier"], "standing_seat": r["standing_seat"]}
                py = np.asarray(_bury_choice(None, msg, "cpu", None, cache, bnet), dtype=np.int64)
                ts_counts = np.asarray(r["counts"], dtype=np.int64)
                ctx = Ctx(r["level_rank"], r["trump_suit"])
                static = cache.static_card_features(ctx, 2)[None]
                buried = np.zeros((1, 54), dtype=np.float32)
                pool_f = pool.astype(np.float32)[None]
                tied = False
                for step, s in enumerate(r["steps"]):
                    cards, ctxv, m = _step_inputs(static, pool_f, buried, [ctx],
                                                  [(r["standing_tier"], r["declarer_rel"])],
                                                  step, 2)
                    if not (np.array_equal(cards[0].reshape(-1).view(np.uint32),
                                           _f32(s["cards"]).view(np.uint32))
                            and np.array_equal(ctxv[0].view(np.uint32), _f32(s["ctx"]).view(np.uint32))
                            and np.array_equal(m[0], np.asarray(s["mask"], dtype=bool))):
                        bury_feat_bad += 1
                    with torch.no_grad():
                        logits, _lp, _v = bnet(torch.from_numpy(cards), torch.from_numpy(ctxv),
                                               torch.from_numpy(m),
                                               torch.zeros(1, 3 * 54))
                    lg = logits[0].numpy().astype(np.float64)
                    tsl = np.asarray(s["logits"], dtype=np.float64)
                    max_blogit = max(max_blogit, float(np.abs(lg - tsl)[m[0]].max()))
                    best = int(np.argmax(np.where(m[0], lg, -np.inf)))
                    picks_total += 1
                    picks_same += int(best == s["pick"])
                    if best != s["pick"]:
                        if _tie(lg, m[0], best, s["pick"]):
                            tied = True
                            bury_tie_steps += 1
                        # follow the TS picks so later steps compare like for like
                    buried[0, s["pick"]] += 1
                if not np.array_equal(py, ts_counts) and not tied:
                    bury_bad += 1
                if fixture:
                    fix_bury.append({k: r[k] for k in (
                        "level_rank", "trump_suit", "standing_tier", "declarer_rel", "pool")}
                        | {"counts": py.tolist(),
                           "step0_logits": _first_logits(bnet, static, pool_f, ctx, r).tolist()})
    print(f"declarations: {n_dec}, features differing {feat_bad}, max |logit diff| "
          f"{max_dlogit:.2e}, choices differing {choice_bad} (+{choice_tie} within tolerance)")
    print(f"burials: {n_bury}, step inputs differing {bury_feat_bad}, max |logit diff| "
          f"{max_blogit:.2e}, picks agreeing {picks_same}/{picks_total} "
          f"({bury_tie_steps} within tolerance), burials differing {bury_bad}")
    if fixture:
        fix_dec = _spread(fix_dec, limit)
        fix_bury = _spread(fix_bury, limit)
        with open(fixture, "w") as fh:
            json.dump({"declare": fix_dec, "bury": fix_bury}, fh, separators=(",", ":"))
        print(f"wrote {fixture}: {len(fix_dec)} declarations, {len(fix_bury)} burials")
    ok = feat_bad == 0 and choice_bad == 0 and bury_feat_bad == 0 and bury_bad == 0 \
        and max_dlogit <= LOGIT_TOL and max_blogit <= LOGIT_TOL and n_dec > 0 and n_bury > 0
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def _spread(items: list, n: int) -> list:
    """``n`` items evenly spaced through ``items``."""
    if len(items) <= n:
        return items
    return [items[(i * len(items)) // n] for i in range(n)]


def _first_logits(bnet, static, pool_f, ctx, r) -> np.ndarray:
    cards, ctxv, m = _step_inputs(static, pool_f, np.zeros((1, 54), dtype=np.float32), [ctx],
                                  [(r["standing_tier"], r["declarer_rel"])], 0, 2)
    with torch.no_grad():
        logits, _lp, _v = bnet(torch.from_numpy(cards), torch.from_numpy(ctxv),
                               torch.from_numpy(m), torch.zeros(1, 3 * 54))
    return np.where(m[0], logits[0].numpy().astype(np.float64), 0.0)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("out_dir")
    ck = sub.add_parser("check")
    ck.add_argument("dump")
    ck.add_argument("checkpoint")
    ck.add_argument("--fixture")
    ck.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make(args.out_dir)
        return 0
    return check(args.dump, args.checkpoint, args.fixture, args.limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
