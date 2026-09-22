"""Playing strength against the heuristic bots, over duplicate deals.

``duplicate.py`` scores one Python policy against another. The strongest
opponents this project has are not Python: ``smart-v34`` and its lineage
live in the TypeScript engine, are the bots the shipped game actually
uses, and are the bar a learned agent has to clear to be worth
deploying. Nothing measures the network against them, so "is it any
good" has so far been answered against uniform random.

This is the bridge. ``research/src/evalhost.ts`` owns the game -- the
engine's rules, the engine's bots, the engine's duplicate-deal seeding
-- and hands over each learner decision as a position plus the legal
moves at it, over NDJSON. Here the position is rebuilt as a
``PlayState``, encoded with the training encoder, and the supplied
candidates are scored; the chosen index goes back down the pipe.

Two things this deliberately does not do.

It does not re-enumerate the legal moves. The host's candidate list is
the authority, because the host is what will actually play the chosen
move: a move this side considered legal and the engine rejects is a
crash at best and a silently different game at worst. The list arrives
as card counts and is scored as given.

It does not build its own observation. Packing goes through
``rollout._pack``, the same call the collector makes, so the policy
being measured is the policy that was trained. An evaluation that
encoded even slightly differently would still run, still return
plausible numbers, and be wrong in a way only a comparison between the
two paths could reveal.

The statistics are ``duplicate.py``'s: one score per deal, taken as the
mean of the network's signed result in the two halves, with the error
bar clustered by deal rather than by hand. See that module for why the
deal is the independent unit.

    python -m tractor.eval.bots --deals 4 --bot smart-v34
"""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..nn.features import Cache
from ..nn.model import ModelConfig
from ..rules.cards import Ctx, NUM_TYPES
from ..sim.actions import MoveSet
from ..sim.state import NUM_SEATS, PlayState, TrickRecord, team_of
from ..train.buffer import obs_batch_from_arrays
from ..train.rollout import _InFlight, _choose, _evaluate, _pack
from .duplicate import HandOutcome, _interval, signed_levels, signed_points

__all__ = ["play_against_bot", "state_from_wire"]

# research/py/tractor/eval/bots.py -> the repo root, which is where the
# npm workspace lives and therefore where the host has to be launched.
REPO = Path(__file__).resolve().parents[4]
HOST = REPO / "research" / "src" / "evalhost.ts"

# The opponents a checkpoint is measured against, cheapest first.
#
# One rung cannot tell saturation from a plateau: a policy that has
# stopped improving and a policy whose opponent is too weak to separate
# look identical against that opponent, and different across a ladder --
# the plateau is flat everywhere, the saturation only flat at the easy
# end. This run showed exactly that, reading as a plateau against random
# play while sitting within noise of the strongest bot.
#
# The order is load bearing. Per decision, measured: the first eight
# rungs cost 12 to 18 milliseconds, smart-v9 costs 93, and the rungs
# that search cost 560 to 670 -- a factor of fifty. A measurement cut
# short by its budget therefore still returns the cheap end, which is
# the half that runs every time.
LADDER: tuple[str, ...] = (
    "random",
    "lowball",
    "point-hoarder",
    "aggressive",
    "smart",
    "smart-v2",
    "smart-v4",
    "smart-v9",
    "smart-v18",
    "smart-v28",
    "smart-v34",
)


def resolve_bots(names: Sequence[str]) -> tuple[str, ...]:
    """Expand the name ``ladder`` to the rungs; pass anything else through."""
    out: list[str] = []
    for name in names:
        if name == "ladder":
            out.extend(r for r in LADDER if r not in out)
        elif name not in out:
            out.append(name)
    return tuple(out)

# Which team the network holds in each half. Half 0 seats it on 0 and 2,
# half 1 on 1 and 3; ``team_of`` is the authority on which team that is,
# and each half is one whole team so "the network's result" is a single
# number. The host builds the same two seatings.
NET_TEAM = (team_of(0), team_of(1))

# Lines of the child's stderr kept for an error message. A failure is
# usually a stack trace from the host, which is a few dozen lines; the
# cap is there so a child stuck in a logging loop cannot exhaust memory.
STDERR_LINES = 400


# --- the wire format ------------------------------------------------------


def state_from_wire(msg: dict) -> PlayState:
    """Rebuild the position the host is asking about.

    The fields are the ones ``tests/test_bots.py`` pins: the trump
    context, all four hands as they stand now, the burial, the completed
    tricks, the trick in progress and who leads it. A state assembled
    this way has a log that does not describe it -- see
    ``PlayState.rebuild_log`` -- so it is rebuilt before anything reads
    it, or the encoder would see an empty history and a hand of phantom
    unseen cards.

    All four hands cross the wire because the critic reads all four --
    that is what ``privileged_view`` is -- and because the seat to act
    changes from one message to the next. The actor's own observation
    may not see three of them, and that is enforced on this side rather
    than by withholding fields: ``tests/test_features.py`` re-deals
    everything the acting seat cannot see and demands a bit-identical
    observation, so a leak fails there rather than depending on what a
    host remembered to omit.
    """
    ctx = Ctx(int(msg["level_rank"]), int(msg["trump_suit"]))
    hands = np.asarray(msg["hands"], dtype=np.int8)
    if hands.shape != (NUM_SEATS, NUM_TYPES):
        raise ValueError(f"expected {NUM_SEATS} hands of {NUM_TYPES}, got {hands.shape}")
    state = PlayState.new(
        ctx,
        hands,
        np.asarray(msg["kitty"], dtype=np.int8),
        int(msg["landlord"]),
        int(msg["num_decks"]),
    )
    state.tricks = [
        TrickRecord(
            leader=int(t["leader"]),
            plays=[np.asarray(p, dtype=np.int8) for p in t["plays"]],
            winner=int(t["winner"]),
            points=int(t["points"]),
        )
        for t in msg["tricks"]
    ]
    state.trick_plays = [np.asarray(p, dtype=np.int8) for p in msg["trick_plays"]]
    state.trick_leader = int(msg["trick_leader"])
    state.to_act = int(msg["to_act"])
    state.rebuild_log()
    return state


# --- the child process ----------------------------------------------------


class HostError(RuntimeError):
    """The host failed, exited, or said something unintelligible."""


class _Host:
    """The NDJSON host as a context manager.

    Three failure modes matter and all three are the same bug -- a
    blocked pipe -- wearing different clothes. The child's stderr is
    drained by a thread, because a host that filled the 64 KB stderr
    buffer would block forever writing a stack trace nobody was reading.
    A closed stdout is read as end of stream and raised, rather than
    waited on. And the child is killed in a ``finally``, so an exception
    on this side does not leave a Node process holding the terminal.
    """

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        self._errors: deque[str] = deque(maxlen=STDERR_LINES)
        self.proc = subprocess.Popen(
            self.argv,
            cwd=str(REPO),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._drain = threading.Thread(target=self._pump_stderr, daemon=True)
        self._drain.start()

    def _pump_stderr(self) -> None:
        stream = self.proc.stderr
        assert stream is not None
        for line in stream:
            self._errors.append(line.rstrip("\n"))

    def stderr_text(self) -> str:
        # Joined on demand rather than accumulated, so the thread does no
        # string building on the hot path.
        return "\n".join(self._errors)

    def fail(self, what: str) -> HostError:
        code = self.proc.poll()
        tail = self.stderr_text()
        return HostError(
            f"{what} (exit {code})\n"
            f"  command: {' '.join(self.argv)}\n"
            f"  stderr:\n{tail if tail else '    (empty)'}"
        )

    def read(self) -> dict:
        stream = self.proc.stdout
        assert stream is not None
        line = stream.readline()
        if line == "":
            # A moment for the exit code and the last of stderr to land,
            # so the error carries the reason rather than only the fact.
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
                pass
            self._drain.join(timeout=2)
            raise self.fail("the host closed its output before saying done")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise self.fail(f"unparseable line from the host: {line[:200]!r}") from exc

    def write(self, obj: dict) -> None:
        stream = self.proc.stdin
        assert stream is not None
        try:
            stream.write(json.dumps(obj) + "\n")
            stream.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self.fail("the host stopped reading its input") from exc

    def close(self) -> None:
        """Shut the child down, whatever state it is in."""
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._drain.join(timeout=5)
        if self.proc.stderr is not None:
            self.proc.stderr.close()


def _launcher(node: str | Sequence[str] | None) -> list[str]:
    """The argv prefix that runs a TypeScript file.

    ``None`` resolves ``npx`` on PATH, which is what picks up ``npx.cmd``
    on Windows. A string is taken as the path to ``npx``; a sequence is
    used verbatim, for an environment that runs tsx some other way.
    """
    if node is None:
        found = shutil.which("npx")
        if found is None:
            raise HostError(
                "npx is not on PATH; the bot bridge needs Node to run the "
                "TypeScript host. Pass node= to point at it."
            )
        return [found, "tsx"]
    if isinstance(node, str):
        return [node, "tsx"]
    return list(node)


# --- scoring one decision -------------------------------------------------


def _greedy_index(
    net: Any,
    msg: dict,
    cache: Cache,
    cfg: ModelConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    """The network's move, as an index into the host's candidate list.

    Greedy, for the same reason ``train/evaluate.py`` is: a measurement
    wants the policy's considered move, and sampling would add variance
    the duplicate pairing cannot cancel because it is not a property of
    the deal.
    """
    candidates = [np.asarray(c, dtype=np.int8) for c in msg["candidates"]]
    if len(candidates) > cfg.max_candidates:
        # The host is capped by --max-candidates and this should be
        # unreachable, but silently keeping a prefix would hide a
        # move the network was never shown and never told about.
        raise HostError(
            f"the host offered {len(candidates)} candidates, more than the "
            f"model's {cfg.max_candidates}"
        )
    state = state_from_wire(msg)
    seat = int(msg["seat"])
    if int(state.to_act) != seat:
        raise HostError(f"asked seat {seat} to act, but {int(state.to_act)} is to act")

    packed = _pack(
        _InFlight(state=state, episode=0),
        cache,
        cfg,
        moves=MoveSet(moves=candidates),
    )
    obs = obs_batch_from_arrays(
        packed.history[None],
        np.array([packed.history_len], dtype=np.int64),
        packed.context[None],
        packed.unseen[None],
        packed.planes[None],
        packed.cards[None],
        packed.candidates[None],
        packed.candidate_mask[None],
        device=device,
    )
    privileged = torch.from_numpy(packed.privileged[None].astype(np.float32)).to(device)
    with torch.no_grad():
        out, _value = _evaluate(net, obs, privileged)
    slots, _logp = _choose(out.logits, obs.candidate_mask, 1.0, True, rng)
    return int(slots[0])


# --- pairing and statistics -----------------------------------------------


@dataclass(frozen=True, slots=True)
class _Half:
    """One played hand, as the host reported it."""

    landlord: int
    attacker_points: int
    defender_gain: int
    attacker_gain: int
    digest: int
    decisions: int


def _paired(games: dict[int, dict[int, _Half]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-deal paired scores, in levels and in attacker points.

    The two halves of a deal are a matched pair rather than two draws:
    whatever the cards were worth, both policies were handed them, and
    the mean of the two signed results is what is left once that
    cancels.
    """
    levels: list[float] = []
    points: list[float] = []
    for game in sorted(games):
        halves = games[game]
        if set(halves) != {0, 1}:
            raise HostError(
                f"deal {game} reported halves {sorted(halves)}, expected both"
            )
        # Checked here as well as in the host. The host compares the
        # cards it dealt; this compares what it *said* it dealt, which
        # is the only thing the statistics below are computed from.
        if halves[0].digest != halves[1].digest:
            raise HostError(
                f"deal {game}: the two halves were dealt different cards "
                f"(digest {halves[0].digest} vs {halves[1].digest})"
            )
        signed_l: list[int] = []
        signed_p: list[int] = []
        for half in (0, 1):
            got = halves[half]
            # ``tricks`` is not on the wire and nothing here reads it;
            # the outcome is built so that the sign convention comes
            # from duplicate.py rather than being restated.
            outcome = HandOutcome(
                attacker_points=got.attacker_points,
                defender_levels=got.defender_gain,
                attacker_levels=got.attacker_gain,
                defenders=team_of(got.landlord),
                tricks=0,
                decisions=got.decisions,
            )
            signed_l.append(signed_levels(outcome, NET_TEAM[half]))
            signed_p.append(signed_points(outcome, NET_TEAM[half]))
        levels.append(0.5 * (signed_l[0] + signed_l[1]))
        points.append(0.5 * (signed_p[0] + signed_p[1]))
    return np.array(levels, dtype=np.float64), np.array(points, dtype=np.float64)


def summarise(games: dict[int, dict[int, _Half]]) -> dict[str, float]:
    """The flat statistics dict, in ``train/evaluate.py``'s shape."""
    levels, points = _paired(games)
    mean, stderr, lo, hi = _interval(levels)
    pmean, pstderr, _plo, _phi = _interval(points)
    decisions = sum(h.decisions for halves in games.values() for h in halves.values())
    return {
        "deals": float(levels.size),
        "levels": mean,
        "levels_stderr": stderr,
        "levels_ci_low": lo,
        "levels_ci_high": hi,
        "points": pmean,
        "points_stderr": pstderr,
        # The network's own decisions, not every decision in the hand:
        # the bot's moves are made inside the host and never seen here.
        "decisions": float(decisions),
        # Positive only when the whole interval is, which is the
        # question a training chart is actually being asked.
        "ahead": float(lo > 0.0),
    }


# --- the match ------------------------------------------------------------


def play_against_bot(
    net: Any,
    bot: str,
    *,
    device: torch.device | str = "cpu",
    cfg: ModelConfig | None = None,
    num_deals: int = 64,
    seed: int = 0,
    max_candidates: int = 128,
    node: str | Sequence[str] | None = None,
    workers: int = 1,
) -> dict[str, float]:
    """Duplicate-score ``net`` against the engine's ``bot``.

    ``num_deals`` deals, each played twice with the seats swapped, so
    ``2 * num_deals`` hands. ``seed`` is the deal seed base: fixed
    across training iterations, successive measurements then differ by
    the policy and not by which cards came up.

    ``workers`` hosts run at once, each playing the deals congruent to
    its own index. The heuristic bots search: measured here, a decision
    costs 5.5 ms from ``lowball`` and 668 ms from ``smart-v34``, which
    puts a single host at about a minute a deal and makes a hundred-deal
    measurement impossible to take often. The deals are independent
    given the seed base, so the shards partition the work exactly and
    the result does not depend on how many there are.

    Returns the same flat dict ``train/evaluate.py`` does, positive when
    the network is ahead.
    """
    if num_deals < 1:
        raise ValueError(f"num_deals must be at least 1, got {num_deals}")
    if not HOST.exists():  # pragma: no cover - a broken checkout
        raise HostError(f"the evaluation host is missing: {HOST}")
    cfg = cfg or getattr(net, "cfg", None) or ModelConfig()
    device = torch.device(device)
    # The host must never offer a move the padded candidate axis cannot
    # carry: the surplus would be dropped here, and a candidate the
    # policy cannot see is worse than one that was never offered.
    cap = min(int(max_candidates), int(cfg.max_candidates))

    shards = max(1, min(int(workers), int(num_deals)))

    was_training = getattr(net, "training", False)
    if hasattr(net, "eval"):
        net.eval()

    def run_shard(shard: int) -> dict[int, dict[int, _Half]]:
        argv = [
            *_launcher(node),
            str(HOST),
            "--deals",
            str(int(num_deals)),
            "--seed-base",
            str(int(seed)),
            "--bot",
            str(bot),
            "--max-candidates",
            str(cap),
            "--shards",
            str(shards),
            "--shard",
            str(shard),
        ]
        # One cache per shard, as in the rollout: it holds the static
        # per-trump-context block, which is shared across hands. Per
        # shard rather than shared, since a shard runs on its own thread
        # and the cache is not synchronised.
        cache = Cache()
        # Greedy play never draws from this, but ``_choose`` takes a
        # generator and a policy that silently became stochastic should
        # not also become irreproducible.
        rng = np.random.default_rng(seed + shard)

        games: dict[int, dict[int, _Half]] = {}
        pending_decisions: dict[tuple[int, int], int] = {}
        host = _Host(argv)
        try:
            while True:
                msg = host.read()
                kind = msg.get("t")
                if kind == "pending":
                    key = (int(msg["game"]), int(msg["half"]))
                    pending_decisions[key] = pending_decisions.get(key, 0) + 1
                    index = _greedy_index(net, msg, cache, cfg, device, rng)
                    host.write({"t": "choose", "index": index})
                elif kind == "hand":
                    game = int(msg["game"])
                    half = int(msg["half"])
                    if half not in (0, 1):
                        raise HostError(f"deal {game} reported half {half}")
                    games.setdefault(game, {})[half] = _Half(
                        landlord=int(msg["landlord"]),
                        attacker_points=int(msg["attacker_points"]),
                        defender_gain=int(msg["defender_gain"]),
                        attacker_gain=int(msg["attacker_gain"]),
                        digest=int(msg["digest"]),
                        decisions=pending_decisions.pop((game, half), 0),
                    )
                elif kind == "done":
                    got = int(msg["deals"])
                    # This shard plays the game ids congruent to its own
                    # index, so it owns a known share of the total.
                    mine = len(range(shard, num_deals, shards))
                    if got != num_deals or len(games) != mine:
                        raise HostError(
                            f"shard {shard} of {shards} owed {mine} of "
                            f"{num_deals} deals, reported {len(games)} "
                            f"(host finished {got})"
                        )
                    break
                else:
                    raise HostError(f"unknown message from the host: {msg!r}")
        finally:
            host.close()
        return games

    merged: dict[int, dict[int, _Half]] = {}
    try:
        if shards == 1:
            merged = run_shard(0)
        else:
            with ThreadPoolExecutor(max_workers=shards) as pool:
                for part in pool.map(run_shard, range(shards)):
                    merged.update(part)
    finally:
        if was_training and hasattr(net, "train"):
            net.train()

    if len(merged) != num_deals:
        raise HostError(
            f"asked for {num_deals} deals, {shards} shard(s) reported "
            f"{len(merged)}"
        )
    return summarise(merged)


# --- cli ------------------------------------------------------------------


def main() -> int:
    import argparse

    from ..nn.model import ActorCritic

    ap = argparse.ArgumentParser(description="Network against a TypeScript bot.")
    ap.add_argument("--deals", type=int, default=4)
    ap.add_argument("--bot", default="smart-v34")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-candidates", type=int, default=128)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = ModelConfig()
    net = ActorCritic(cfg)
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**blob["model_config"])
        net = ActorCritic(cfg)
        net.load_state_dict(blob["model"])
    net.to(args.device)

    started = time.perf_counter()
    stats = play_against_bot(
        net,
        args.bot,
        device=args.device,
        cfg=cfg,
        num_deals=args.deals,
        seed=args.seed,
        max_candidates=args.max_candidates,
    )
    elapsed = time.perf_counter() - started

    label = "checkpoint" if args.checkpoint else "untrained net"
    print(f"{label} vs {args.bot}")
    print(
        f"  {stats['levels']:+.4f} levels/deal  se {stats['levels_stderr']:.4f}  "
        f"95% [{stats['levels_ci_low']:+.4f}, {stats['levels_ci_high']:+.4f}]"
    )
    print(
        f"  {stats['points']:+.2f} points/deal  se {stats['points_stderr']:.2f}  "
        f"n={int(stats['deals'])} deals, {int(stats['decisions'])} decisions"
    )
    print(
        f"  {elapsed:.1f}s total, {elapsed / max(args.deals, 1):.2f}s/deal, "
        f"ahead {stats['ahead']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
