"""Drives the TS subprocess (`trainHost.ts`) over NDJSON stdin/stdout.

One-process-per-actor design: each TrainHost instance owns a long-lived
TS child. Episodes are issued sequentially via run_episode().
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional


@dataclass
class Decision:
    """One Q-network query inside an episode."""
    trick_idx: int
    seat: int
    role: str
    side: str
    lead_chosen_count: int
    lead_pattern_count: int
    observation: dict  # raw observation dict from TS
    candidates: list   # list of candidate dicts (kind, cards, tags, encoding, effectiveSuit)
    chosen_index: int = -1  # filled in by the actor when it chooses


@dataclass
class EpisodeResult:
    decisions: List[Decision]
    per_trick_attacker_captures: List[int]
    per_trick_attacker_won: List[int]
    last_trick_kitty_bonus: int
    attacker_points: int
    landlord: int


class TrainHost:
    """Long-lived TS subprocess wrapper."""

    def __init__(self, host_script: str, cwd: str, node_cmd: Optional[List[str]] = None) -> None:
        if node_cmd is None:
            node_cmd = ['npx', 'tsx', host_script]
        self.proc = subprocess.Popen(
            node_cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            shell=(os.name == 'nt'),  # required on Windows for npx.cmd
        )
        # Wait for the ready event.
        ready = self._read_message()
        if ready.get('event') != 'ready':
            raise RuntimeError(f'expected ready, got {ready}')

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg) + '\n'
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _read_message(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            stderr = ''
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read()
            raise RuntimeError(f'subprocess closed stdout. stderr: {stderr}')
        return json.loads(line)

    def run_episode(
        self,
        seed: int,
        choose: Callable[[Decision], int],
    ) -> EpisodeResult:
        """Run one hand. `choose(decision)` returns the index of the chosen
        candidate; this driver applies it and continues until 'done'."""
        self._send({'cmd': 'new_episode', 'seed': seed})
        decisions: List[Decision] = []
        while True:
            msg = self._read_message()
            event = msg.get('event')
            if event == 'decision':
                d = Decision(
                    trick_idx=msg['trickIdx'],
                    seat=msg['seat'],
                    role=msg['role'],
                    side=msg['side'],
                    lead_chosen_count=msg['leadInProgress']['chosen'],
                    lead_pattern_count=msg['leadInProgress']['patternCount'],
                    observation=msg['observation'],
                    candidates=msg['candidates'],
                )
                idx = choose(d)
                d.chosen_index = idx
                decisions.append(d)
                self._send({'cmd': 'choose', 'index': idx})
            elif event == 'done':
                return EpisodeResult(
                    decisions=decisions,
                    per_trick_attacker_captures=list(msg['perTrickAttackerCaptures']),
                    per_trick_attacker_won=list(msg['perTrickAttackerWon']),
                    last_trick_kitty_bonus=int(msg['lastTrickKittyBonus']),
                    attacker_points=int(msg['attackerPoints']),
                    landlord=int(msg['landlord']),
                )
            elif event == 'error':
                raise RuntimeError(f'subprocess error: {msg.get("message")}')
            else:
                raise RuntimeError(f'unexpected event: {msg}')

    def close(self) -> None:
        try:
            self._send({'cmd': 'exit'})
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
