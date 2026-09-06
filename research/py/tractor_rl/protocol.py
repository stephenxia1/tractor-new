"""Spawns TypeScript self-play workers and speaks the batched frame protocol.

Each worker holds many games at once and only talks to Python when every
one of them needs a policy decision, so a round trip carries hundreds of
decisions instead of one. That is what keeps the process boundary cheap:
the network never leaves Python, the rules never leave TypeScript, and
neither side needs a copy of the other.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from . import layout as L
from .features2 import Batch, parse_frame

RESEARCH_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RESEARCH_DIR.parent
TSX_CLI = REPO_ROOT / "node_modules" / "tsx" / "dist" / "cli.mjs"
WORKER_SCRIPT = "src/worker.ts"


@dataclass(slots=True)
class WorkerConfig:
    games: int = 64
    seed: int = 1
    # Bit i set means seat i is learner-controlled. 15 = full self-play.
    learner_seats: int = 15
    # Bot filling non-learner seats during play.
    baseline: str = "smart-v34"
    # Bot handling declaration and kitty burial at every seat.
    preplay: str = "smart-v34"
    max_candidates: int = 48
    randomize: bool = True
    # Duplicate mode: when set, this worker's deals are a pure function of
    # (deal_seed_base, game slot, hand index), so a second worker given the
    # same base plays the identical cards while controlling other seats.
    deal_seed_base: int | None = None
    fixed_landlord: int | None = None
    fixed_level: str | None = None
    # A second bot, used on the seats in `alt_seats` (bit i = seat i).
    # Lets one team be filled by a different heuristic from the other.
    baseline_alt: str | None = None
    alt_seats: int = 0
    # Training only: emit the other seats' hands for the centralised critic.
    privileged: bool = False
    # Value-guided search at decision time; 0 worlds disables it.
    search_worlds: int = 0
    search_top: int = 6
    search_opponent: str = "smart-v2"


class Worker:
    def __init__(self, index: int, config: WorkerConfig, node: str = "node") -> None:
        if not TSX_CLI.exists():
            raise RuntimeError(f"tsx not found at {TSX_CLI}; run npm install at the repo root")
        args = [
            node,
            str(TSX_CLI),
            WORKER_SCRIPT,
            "--games",
            str(config.games),
            "--seed",
            str(config.seed),
            "--learner-seats",
            str(config.learner_seats),
            "--baseline",
            config.baseline,
            "--preplay",
            config.preplay,
            "--max-cands",
            str(config.max_candidates),
            "--randomize",
            "1" if config.randomize else "0",
        ]
        if config.deal_seed_base is not None:
            args += ["--deal-seed-base", str(config.deal_seed_base)]
        if config.fixed_landlord is not None:
            args += ["--fixed-landlord", str(config.fixed_landlord)]
        if config.fixed_level is not None:
            args += ["--fixed-level", str(config.fixed_level)]
        if config.privileged:
            args += ["--privileged", "1"]
        if config.search_worlds > 0:
            args += [
                "--search-worlds", str(config.search_worlds),
                "--search-top", str(config.search_top),
                "--search-opponent", config.search_opponent,
            ]
        if config.baseline_alt is not None:
            args += ["--baseline-alt", config.baseline_alt, "--alt-seats", str(config.alt_seats)]
        self.index = index
        self.proc = subprocess.Popen(
            args,
            cwd=str(RESEARCH_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # worker errors go straight to our stderr
            env={**os.environ, "NODE_OPTIONS": os.environ.get("NODE_OPTIONS", "")},
        )

    def read_frame(self) -> bytes:
        assert self.proc.stdout is not None
        head = self._read_exact(L.FRAME_HEADER_BYTES)
        payload_bytes = int.from_bytes(head, "little")
        return self._read_exact(payload_bytes)

    def send_actions(self, game_ids: np.ndarray, choices: np.ndarray) -> None:
        assert self.proc.stdin is not None
        n = int(game_ids.shape[0])
        payload = bytearray(2 + n * L.ACTION_BYTES)
        payload[0:2] = n.to_bytes(2, "little")
        off = 2
        gids = game_ids.astype(np.int64)
        picks = choices.astype(np.int64)
        for i in range(n):
            payload[off : off + 2] = int(gids[i]).to_bytes(2, "little")
            payload[off + 2] = int(picks[i])
            off += L.ACTION_BYTES
        frame = len(payload).to_bytes(4, "little") + bytes(payload)
        self.proc.stdin.write(frame)
        self.proc.stdin.flush()

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def _read_exact(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                code = self.proc.poll()
                raise RuntimeError(
                    f"worker {self.index} closed the pipe (exit code {code})"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class WorkerPool:
    """A set of workers stepped in lockstep: read every worker's pending
    decisions, decide them all in one forward pass, write the answers back.

    With `num_groups > 1` the workers are split into groups stepped in
    rotation instead. That matters because a worker computes only between
    receiving its actions and emitting its next frame: with one group, the
    TypeScript side sits idle for the whole featurise-and-forward window and
    Python sits idle for the whole worker window, and the two are almost the
    same size (measured 50.6% / 47.2% of loop time). Handing group *g* its
    actions and then immediately reading group *g+1* -- which has been
    computing since its own actions went out one iteration ago -- overlaps
    the two instead of alternating them.

    Each group is stepped at 1/num_groups the rate, so double the workers to
    keep the forward-pass batch the same size.
    """

    def __init__(
        self,
        num_workers: int,
        config: WorkerConfig,
        node: str = "node",
        num_groups: int = 1,
        configs: list[WorkerConfig] | None = None,
    ) -> None:
        # `configs` lets a caller give individual workers different settings
        # (the arena runs half its workers on each side of a duplicate pair).
        if configs is None:
            configs = [
                replace(config, seed=config.seed + i * 101) for i in range(num_workers)
            ]
        self.workers = [Worker(i, cfg, node) for i, cfg in enumerate(configs)]
        n = max(1, min(num_groups, len(self.workers)))
        # Round-robin so a group spans workers evenly even if the caller
        # ordered them by role.
        self.groups: list[list[Worker]] = [self.workers[i::n] for i in range(n)]
        self._cursor = 0

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def next_group(self) -> int:
        g = self._cursor
        self._cursor = (self._cursor + 1) % len(self.groups)
        return g

    def workers_of(self, group: int) -> list[Worker]:
        return self.groups[group]

    def read_group(self, group: int) -> list[Batch]:
        return [parse_frame(w.read_frame()) for w in self.groups[group]]

    def write_group(
        self, group: int, batches: Iterable[Batch], choices: Iterable[np.ndarray]
    ) -> None:
        for worker, batch, picks in zip(self.groups[group], batches, choices):
            worker.send_actions(game_ids_of(batch), picks)

    def read_round(self) -> list[Batch]:
        return [parse_frame(w.read_frame()) for w in self.workers]

    def write_round(self, batches: Iterable[Batch], choices: Iterable[np.ndarray]) -> None:
        for worker, batch, picks in zip(self.workers, batches, choices):
            game_ids = game_ids_of(batch)
            worker.send_actions(game_ids, picks)

    def write_actions(self, batches, game_ids_list, picks_list) -> None:
        """Reply with an explicit game-id list per worker.

        Search frames carry more records than games -- the leaves -- so the
        reply cannot be positional: it names the games it answers.
        """
        for worker, gids, picks in zip(self.workers, game_ids_list, picks_list):
            worker.send_actions(gids, picks)

    def close(self) -> None:
        for w in self.workers:
            w.close()


def game_ids_of(batch: Batch) -> np.ndarray:
    if batch.num_decisions == 0:
        return np.zeros(0, dtype=np.int64)
    return (
        batch.states[:, L.H_GAME_ID].astype(np.int64)
        + batch.states[:, L.H_GAME_ID + 1].astype(np.int64) * 256
    )


def hand_ids_of(batch: Batch) -> np.ndarray:
    if batch.num_decisions == 0:
        return np.zeros(0, dtype=np.int64)
    b = batch.states[:, L.H_HAND_ID : L.H_HAND_ID + 4].astype(np.int64)
    return b[:, 0] + (b[:, 1] << 8) + (b[:, 2] << 16) + (b[:, 3] << 24)


def node_executable() -> str:
    return os.environ.get("NODE", "node")


def check_node() -> None:
    try:
        subprocess.run(
            [node_executable(), "--version"],
            check=True,
            capture_output=True,
        )
    except Exception as exc:  # pragma: no cover - environment problem
        print(f"node not runnable: {exc}", file=sys.stderr)
        raise
