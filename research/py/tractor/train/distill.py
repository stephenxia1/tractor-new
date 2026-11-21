"""Expert iteration: train the policy toward its own search's choices.

Search (``eval/search.py``) adds +0.03 to +0.05 win against the fair bots
at every checkpoint measured (README, *A zero bot that beats the fair
search bots*): sampled deals consistent with the seat's view, each of
the policy's top candidates played forward, the critic's value at the
leaves. That is a policy-improvement operator built only from the
policy, the critic and the rules -- zero, as AlphaZero's search is.
This module turns it into training signal.

``generate`` plays hands from training-distribution starts with the
policy, sampling each move as training does (so the positions are the
learner's own), and wherever search would act -- more than one move and
no move above ``search.confident`` -- runs it and records the position
with search's choice. ``Distiller`` then fits the actor to those
choices with a cross-entropy step of its own, beside the PPO update
(its own optimiser, so the PPO optimiser's state and any CUDA graphs
captured around it are left alone).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from ..eval.search import SearchConfig, search_move
from ..nn.features import Cache
from ..nn.model import ModelConfig
from .buffer import obs_batch_from_arrays
from .rollout import _InFlight, _evaluate, _pack

__all__ = ["DistillRows", "generate", "load_rows", "Distiller"]


@dataclass
class DistillRows:
    """Positions where search acted, as the rollout packs them, and the
    index of the candidate search chose."""

    history: np.ndarray  # float16[R, H, D]
    history_len: np.ndarray  # int64[R]
    context: np.ndarray  # float16[R, ...]
    unseen: np.ndarray
    planes: np.ndarray
    cards: np.ndarray
    candidates: np.ndarray  # float16[R, C, 54]
    candidate_mask: np.ndarray  # bool[R, C]
    privileged: np.ndarray  # float16[R, ...]
    target: np.ndarray  # int64[R]
    prior_top: np.ndarray  # int64[R], the policy's own argmax there

    def __len__(self) -> int:
        return len(self.target)

    def save(self, path: str) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})


def load_rows(pattern: str) -> DistillRows:
    """Every shard matching ``pattern``, concatenated."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no distillation shards match {pattern}")
    parts = [np.load(p) for p in paths]
    return DistillRows(**{f.name: np.concatenate([p[f.name] for p in parts])
                          for f in fields(DistillRows)})


def _actor(net: Any) -> Any:
    return getattr(net, "actor", net)


def generate(
    net: Any,
    starts: list,
    search: SearchConfig,
    rng: np.random.Generator,
    device: Any,
    cfg: ModelConfig,
    cache: Cache | None = None,
    max_rows: int | None = None,
    agree: bool = False,
) -> tuple[DistillRows, dict[str, float]]:
    """Play ``starts`` to the end with the policy; label the positions
    where search acts with search's choice. -> (rows, stats).

    ``agree``: search each such position a second time, on worlds of its
    own, and keep the label only where both searches choose the same
    move. Two searches of one position agree on ~72% of them (README,
    "Distillation flattened the policy"); the rest are search choosing
    between moves its few worlds cannot tell apart. Costs a second search
    per label."""
    from ..eval.search import CandidatePolicy

    cache = cache or Cache()
    keep: dict[str, list] = {f.name: [] for f in fields(DistillRows)}
    decisions = searched = changed = agreed = 0
    net.eval()
    # Search's forward (eval/fastnet.py): the policy's logits computed as
    # the training forward computes them, without the work not read.
    policy = CandidatePolicy(net, cache, cfg, device)
    fast = policy.fast
    for start in starts:
        state = start.state()
        while not state.done:
            moves = state.legal_moves(cap=cfg.max_candidates)
            packed = _pack(_InFlight(state=state, episode=0), cache, cfg, moves=moves)
            n = int(packed.candidate_mask.sum())
            obs = obs_batch_from_arrays(
                packed.history[None], np.array([packed.history_len], dtype=np.int64),
                packed.context[None], packed.unseen[None], packed.planes[None],
                packed.cards[None], packed.candidates[None], packed.candidate_mask[None],
                device=device,
            )
            priv = torch.from_numpy(packed.privileged[None].astype(np.float32)).to(device)
            with torch.no_grad():
                if fast is not None:
                    logits = fast.logits(obs)
                else:
                    logits = _evaluate(net, obs, priv)[0].logits
                logp_all = torch.log_softmax(logits.float(), dim=-1)
            logp = logp_all[0, :n].cpu().numpy()
            decisions += 1
            top = int(np.argmax(logp))
            if n > 1 and float(np.exp(logp[top])) < search.confident:
                index, did = search_move(net, state, list(moves.moves[:n]), logp, search,
                                         rng, cache, cfg, device, policy=policy)
                if did:
                    searched += 1
                    changed += int(index != top)
                if did and agree:
                    again, _ = search_move(net, state, list(moves.moves[:n]), logp, search,
                                           rng, cache, cfg, device, policy=policy)
                    did = again == index
                    agreed += int(did)
                if did:
                    for name in ("history", "context", "unseen", "planes", "cards",
                                 "candidates", "candidate_mask", "privileged"):
                        keep[name].append(getattr(packed, name))
                    keep["history_len"].append(packed.history_len)
                    keep["target"].append(index)
                    keep["prior_top"].append(top)
            # The learner's own move, sampled as the rollout samples it.
            p = np.exp(logp - logp.max())
            state.step(moves.moves[int(rng.choice(n, p=p / p.sum()))])
            if max_rows is not None and len(keep["target"]) >= max_rows:
                break
        if max_rows is not None and len(keep["target"]) >= max_rows:
            break
    if not keep["target"]:
        raise RuntimeError("search acted on no decision")
    rows = DistillRows(**{
        name: np.asarray(v, dtype=np.int64) if name in ("history_len", "target", "prior_top")
        else np.stack(v) for name, v in keep.items()})
    stats = {"decisions": float(decisions), "searched": float(searched),
             "search_changed": float(changed),
             "searched_frac": searched / max(decisions, 1),
             "changed_frac": changed / max(searched, 1)}
    if agree:
        stats.update(search_agreed=float(agreed), agree_frac=agreed / max(searched, 1))
    return rows, stats


class Distiller:
    """Cross-entropy toward search's choices on the actor alone."""

    def __init__(self, net: Any, rows: DistillRows | None, device: Any, lr: float = 1e-5,
                 minibatch: int = 512, coef: float = 1.0, max_grad_norm: float = 0.5,
                 bf16: bool = True, source: str | None = None,
                 window_rows: int = 0, max_age: int = 0) -> None:
        """``rows`` fixed for the run, or ``source`` (a shard glob) read by
        ``refresh`` -- the newest shards up to ``window_rows`` rows, none
        older than ``max_age`` steps (0: any age) -- for labels made while
        the run trains (``train/labels.py``). Until a refresh finds any,
        ``step`` does nothing."""
        self.net = net
        self.rows = rows
        self.fresh = None
        if source is not None:
            from .labels import FreshRows

            self.fresh = FreshRows(source, window_rows, max_age)
        self.device = torch.device(device)
        self.minibatch = int(minibatch)
        self.coef = float(coef)
        self.max_grad_norm = float(max_grad_norm)
        self.params = [p for p in _actor(net).parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.amp = bool(bf16) and self.device.type == "cuda"

    def refresh(self, step: int | None = None) -> dict[str, float]:
        """Reload the freshest shards from ``source``; ``step``, the
        trainer's, ages them. -> metrics."""
        if self.fresh is None:
            return {}
        rows, metrics = self.fresh.load(step)
        if rows is not None:
            self.rows = rows
        elif self.fresh.max_age > 0:
            # Every shard is past its age: stop distilling rather than
            # keep fitting the last window, until fresh labels arrive.
            self.rows = None
        return metrics

    def step(self, n_minibatches: int, rng: np.random.Generator) -> dict[str, float]:
        r = self.rows
        if r is None or len(r) == 0:
            return {"distill/rows": 0.0}
        actor = _actor(self.net)
        actor.train()
        ce_sum = agree_sum = 0.0
        # The PPO update may run as captured CUDA graphs that own the
        # parameters' gradient buffers; this step's backward must not
        # replace them, so they are set aside and put back.
        stash = [p.grad for p in self.params]
        for p in self.params:
            p.grad = None
        for _ in range(int(n_minibatches)):
            idx = rng.integers(0, len(r), size=min(self.minibatch, len(r)))
            obs = obs_batch_from_arrays(
                r.history[idx], r.history_len[idx], r.context[idx], r.unseen[idx],
                r.planes[idx], r.cards[idx], r.candidates[idx], r.candidate_mask[idx],
                device=self.device,
            )
            target = torch.from_numpy(r.target[idx]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.amp):
                out = actor(obs)
            logp = out.log_probs.float()
            ce = -logp.gather(1, target[:, None]).squeeze(1).mean()
            loss = self.coef * ce
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.opt.step()
            ce_sum += float(ce.detach())
            agree_sum += float((logp.argmax(dim=1) == target).float().mean())
        for p, g in zip(self.params, stash):
            p.grad = g
        actor.eval()
        n = max(int(n_minibatches), 1)
        return {"distill/ce": ce_sum / n, "distill/agree": agree_sum / n,
                "distill/rows": float(len(r))}
