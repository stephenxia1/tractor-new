"""Checks the TypeScript `rl-zero-search` bot against the Python search.

    cd research
    npx tsx src/zeroSearchParity.ts --decisions 300 --searches 40 --weights DIR > search.jsonl
    python py/test_zero_search_parity.py search.jsonl CHECKPOINT [--fixture OUT]

What it checks, against the PyTorch network with its weights rounded to
float16 as the export stores them:

1. **Critic.** The TS critic's value of every dumped position against
   ``ActorCritic.forward``'s.
2. **Worlds.** Every sampled world deals exactly the cards the seat has
   not seen, sizes each hand as it stands, keeps the landlord's own
   burial, and respects every void public play has shown unless the
   world says it had to relax them.
3. **Leaves.** Each (world, candidate) simulation's settled points
   (including the kitty bonus when the hand ended) and critic value,
   rebuilt from the dumped leaf position.
4. **Simulated moves.** For every move the TS simulation made for a
   seat with a choice, the move the Python simulator's own list
   (``legal_moves``) would have led to. The TS simulation runs on the
   engine's candidate list, so this measures that one designed
   difference rather than asserting it away.
5. **Whole searches.** ``search_move`` run in Python on the same worlds:
   the same candidates searched, the per-candidate means and the choice.
   Differences here should trace to (4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tractor.eval.bots import state_from_wire  # noqa: E402
from tractor.eval.search import SearchConfig, World, known_voids, search_move  # noqa: E402
from tractor.nn.features import Cache  # noqa: E402
from tractor.nn.model import ActorCritic, ModelConfig, load_model  # noqa: E402
from tractor.rules.cards import POINTS  # noqa: E402
from tractor.sim.actions import MoveSet  # noqa: E402
from tractor.sim.state import NUM_SEATS, team_of  # noqa: E402
from tractor.train.buffer import obs_batch_from_arrays  # noqa: E402
from tractor.train.rollout import _evaluate, _InFlight, _pack  # noqa: E402

VALUE_TOL = 1e-4
MEAN_TOL = 0.05  # points; the means are sums of values times 400


def _net(path: str) -> ActorCritic:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(ModelConfig(**blob["model_config"]))
    load_model(net, {k: v.to(torch.float16).to(torch.float32) if torch.is_floating_point(v) else v
                     for k, v in blob["model"].items()})
    return net.eval()


def _forward(net, state, cache, cfg, moves=None):
    p = _pack(_InFlight(state=state, episode=0), cache, cfg, moves=moves)
    obs = obs_batch_from_arrays(
        p.history[None], np.array([p.history_len], dtype=np.int64), p.context[None],
        p.unseen[None], p.planes[None], p.cards[None], p.candidates[None],
        p.candidate_mask[None], device="cpu")
    priv = torch.from_numpy(p.privileged[None].astype(np.float32))
    with torch.no_grad():
        out, value = _evaluate(net, obs, priv)
    return p, out, float(value[0])


def check_world(state, seat: int, world: dict) -> list[str]:
    errs = []
    hands = np.asarray(world["hands"], dtype=np.int64)
    kitty = np.asarray(world["kitty"], dtype=np.int64)
    if not np.array_equal(hands[seat], state.hands[seat]):
        errs.append("seat's own hand changed")
    for s in range(NUM_SEATS):
        if int(hands[s].sum()) != int(state.hands[s].sum()):
            errs.append(f"seat {s} holds {int(hands[s].sum())}, should {int(state.hands[s].sum())}")
    if int(kitty.sum()) != int(state.kitty.sum()):
        errs.append("kitty size")
    if seat == state.landlord and not np.array_equal(kitty, state.kitty):
        errs.append("landlord's own burial changed")
    unseen = state.unseen_from(seat).astype(np.int64)
    placed = sum(hands[s] for s in range(NUM_SEATS) if s != seat)
    if seat != state.landlord:
        placed = placed + kitty
    if not np.array_equal(placed, unseen):
        errs.append("cards dealt are not the unseen cards")
    if not world["relaxed"]:
        voids = known_voids(state)
        eff = state.ctx.eff
        for s in range(NUM_SEATS):
            if s == seat:
                continue
            for t in np.nonzero(hands[s])[0]:
                if voids[s, eff[t]]:
                    errs.append(f"seat {s} dealt a card of a suit it is void in")
                    break
    return errs


def settled_of(leaf_state, start: int, team: int, done: bool) -> float:
    settled = 0.0
    for t in leaf_state.tricks[start:]:
        settled += t.points if team_of(t.winner) == team else -t.points
    if done:
        last = leaf_state.tricks[-1]
        defenders = team_of(leaf_state.landlord)
        if team_of(last.winner) != defenders:
            cards = int(last.plays[(last.winner - last.leader) % NUM_SEATS].sum())
            bonus = int((leaf_state.kitty.astype(np.int64) * POINTS).sum()) * 2 * cards
            settled += bonus if team != defenders else -bonus
    return settled


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("checkpoint")
    ap.add_argument("--fixture", default="")
    ap.add_argument("--fixture-critic", type=int, default=12)
    ap.add_argument("--fixture-search", type=int, default=3)
    args = ap.parse_args(argv)
    net = _net(args.checkpoint)
    cfg = net.cfg
    cache = Cache()

    n = 0
    crit_err = 0.0
    bad: dict[str, int] = {}
    worlds_checked = relaxed = 0
    leaf_err = 0.0
    leaves = 0
    settled_bad = 0
    sim_moves = sim_same = 0
    searches = choice_same = top_same = 0
    mean_err = 0.0
    fixture_critic, fixture_search = [], []
    for line in open(args.dump):
        rec = json.loads(line)
        msg = rec["msg"]
        state = state_from_wire(msg)
        seat = int(msg["seat"])
        cands = [np.asarray(c, dtype=np.int8) for c in msg["candidates"]]
        p, out, value = _forward(net, state, cache, cfg, MoveSet(moves=cands))
        err = abs(value - rec["critic"])
        crit_err = max(crit_err, err)
        if err > VALUE_TOL * max(1.0, abs(value)):
            bad["critic"] = bad.get("critic", 0) + 1
        if (args.fixture and "state" in rec and len(fixture_critic) < args.fixture_critic
                and n % 7 == 0):
            fixture_critic.append({"state": rec["state"], "seat": seat, "value": value})
        n += 1
        srch = rec.get("search")
        if srch is None:
            continue
        searches += 1
        start = len(state.tricks)
        team = team_of(seat)
        for wd in srch["worlds"]:
            worlds_checked += 1
            relaxed += int(wd["relaxed"])
            errs = check_world(state, seat, wd)
            if errs:
                bad["world"] = bad.get("world", 0) + 1
                if bad["world"] == 1:
                    print("first bad world:", errs[:3], file=sys.stderr)
        for leaf in srch["leaves"]:
            leaves += 1
            ls = state_from_wire(leaf["msg"])
            want = settled_of(ls, start, team, leaf["done"])
            if abs(want - leaf["settled"]) > 1e-9:
                settled_bad += 1
            if not leaf["done"]:
                _p, _o, v = _forward(net, ls, cache, cfg)
                leaf_err = max(leaf_err, abs(v - leaf["value"]))
                if abs(v - leaf["value"]) > VALUE_TOL * max(1.0, abs(v)):
                    bad["leaf_value"] = bad.get("leaf_value", 0) + 1
        for pl in srch["playouts"]:
            ps = state_from_wire(pl["msg"])
            ts_move = np.asarray(pl["candidates"][pl["chosen"]], dtype=np.int8)
            opts = ps.legal_moves(cap=cfg.max_candidates)
            if len(opts.moves) == 1:
                py_move = np.asarray(opts.moves[0], dtype=np.int8)
            else:
                pp, po, _ = _forward(net, ps, cache, cfg)
                py_move = np.asarray(pp.moves.moves[int(po.logits[0].argmax())], dtype=np.int8)
            sim_moves += 1
            sim_same += int(np.array_equal(py_move, ts_move))
        # The whole search in Python, on the same worlds.
        worlds = [World(hands=np.asarray(w["hands"], dtype=np.int8),
                        kitty=np.asarray(w["kitty"], dtype=np.int8), relaxed=w["relaxed"])
                  for w in srch["worlds"]]
        log_prior = out.log_probs[0, : len(cands)].float().numpy()
        details: dict = {}
        conf = SearchConfig(worlds=len(worlds), top_k=len(srch["top"]), extra_tricks=0, confident=1.01)
        index, did = search_move(net, state, cands, log_prior, conf, np.random.default_rng(0),
                                 cache, cfg, "cpu", worlds=worlds, details=details)
        top_same += int(list(details.get("top", [])) == list(srch["top"]))
        choice_same += int(index == srch["choice"])
        if "mean" in details:
            mean_err = max(mean_err, float(np.max(np.abs(details["mean"] - np.asarray(srch["means"])))))
        # Fixture searches: ones whose simulations the two sides played
        # identically, so the TS means must reproduce the Python ones.
        exact = "mean" in details and float(np.max(np.abs(
            details["mean"] - np.asarray(srch["means"])))) < 1e-4
        if args.fixture and "state" in rec and exact and len(fixture_search) < args.fixture_search:
            fixture_search.append({"state": rec["state"], "seat": seat, "worlds": srch["worlds"],
                                   "top": [int(x) for x in details["top"]],
                                   "means": [float(x) for x in details["mean"]],
                                   "choice": int(index)})

    print(f"{n} positions: critic max abs error {crit_err:.2e}")
    print(f"{searches} searches, {worlds_checked} worlds ({relaxed} relaxed), {leaves} leaves: "
          f"leaf value max abs error {leaf_err:.2e}, settled mismatches {settled_bad}")
    if sim_moves:
        print(f"simulated moves: {sim_moves}, the Python simulator's list leads to the same move "
              f"on {sim_same} ({100 * sim_same / sim_moves:.1f}%)")
    if searches:
        print(f"whole searches on the same worlds: same candidates {top_same}/{searches}, "
              f"same choice {choice_same}/{searches}, max |mean difference| {mean_err:.3f} points")
    if settled_bad:
        bad["settled"] = settled_bad
    if args.fixture:
        with open(args.fixture, "w") as f:
            json.dump({"note": "generated by research/py/test_zero_search_parity.py --fixture",
                       "critic": fixture_critic, "search": fixture_search}, f, separators=(",", ":"))
        print(f"wrote {args.fixture}")
    if bad:
        print(f"MISMATCHES: {bad}")
        return 1
    print("critic, worlds and leaves match")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
