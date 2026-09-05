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
from .features import Batch, parse_frame

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
    decisions, decide them all in one forward pass, write the answers back."""

    def __init__(
        self,
        num_workers: int,
        config: WorkerConfig,
        node: str = "node",
    ) -> None:
        self.workers = [
            Worker(i, replace(config, seed=config.seed + i * 101), node)
            for i in range(num_workers)
        ]

    def read_round(self) -> list[Batch]:
        return [parse_frame(w.read_frame()) for w in self.workers]

    def write_round(self, batches: Iterable[Batch], choices: Iterable[np.ndarray]) -> None:
        for worker, batch, picks in zip(self.workers, batches, choices):
            game_ids = game_ids_of(batch)
            worker.send_actions(game_ids, picks)

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
