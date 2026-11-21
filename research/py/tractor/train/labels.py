"""Expert-iteration labels on CPUs (2026-11-19).

Search labels used to be a GPU job run once per round
(``cluster gen``, ``modal_perf.py::_distill_gen``). Search is cheap on a CPU
now (``eval/fastnet.py``: ~0.2 s a searched decision at 12 worlds x top 4
on one core, against ~1.2 s for the training forward), so labels can be
made beside training instead of between rounds:

* ``label_hands`` -- one batch of labelled hands from one checkpoint, on
  one core: the agent's own declarations and burials make the starts,
  the policy plays them, search labels the decisions it would act on
  (``distill.generate``).
* ``generate_pool`` -- the same, split across one process per core, for
  a fixed shard (``cluster gen --workers``).
* ``follow`` / ``follow_pool`` -- one process per core following a
  training run: each reloads the run's newest checkpoint whenever there
  is one, writes small shards named by that checkpoint's step into a
  rolling directory, and keeps only its own newest ``keep`` shards
  (``cluster follow``; ``research/ycrc/follow.sbatch``). Before the run
  has written a checkpoint, ``initial`` names one to label from instead
  of waiting (``auto``: the one the run resumed from).
* ``FreshRows`` -- the trainer's side: the newest shards up to a row
  budget, and none older than ``max_age`` steps, reloading only what is
  new (``distill.Distiller`` with ``reload_every``).

Zero throughout: the labels are the agent's own search over its own
policy and critic, under the rules.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "load_agent", "label_hands", "generate_pool", "newest_checkpoint",
    "initial_checkpoint", "shard_name", "shard_step", "follow", "follow_pool",
    "FreshRows",
]

_SHARD = re.compile(r"s(\d{7})-t(\d{3})-w(\d{2})-(\d{5})\.npz$")
_STEP = re.compile(r"step_(\d+)\.pt$")


def load_agent(checkpoint: str | Path, device: Any = "cpu"):
    """``(net, declare_net, bury_net, cfg, step)`` from a full-hand checkpoint."""
    import torch

    from ..nn.arch import bury_net_from, declare_net_from
    from ..nn.model import ActorCritic, ModelConfig, load_model

    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    net = ActorCritic(cfg).to(device)
    load_model(net, blob["model"])
    net.eval()
    if blob.get("declare") is None or blob.get("bury") is None:
        raise ValueError(f"{checkpoint} carries no declaration or burial policy")
    declare_net = declare_net_from(blob["declare"]).to(device).eval()
    bury_net = bury_net_from(blob["bury"]).to(device).eval()
    return net, declare_net, bury_net, cfg, int(blob.get("step", 0))


def label_hands(agent: tuple, hands: int, search: Any, rng: np.random.Generator,
                device: Any = "cpu", cache: Any = None, agree: bool = False):
    """``(DistillRows, stats)`` for ``hands`` hands from ``agent`` (as
    ``load_agent`` returns it); ``agree`` keeps only labels two searches
    agree on (``distill.generate``)."""
    from ..nn.features import Cache
    from . import starts as S
    from .distill import generate

    net, declare_net, bury_net, cfg, _step = agent
    cache = cache or Cache()
    pre = S.PrePlayConfig(random_burial_frac=0.0, burial="policy")
    starts, _rows, _st = S.make_starts(rng, hands, net, declare_net, device, pre=pre,
                                       cache=cache, bury_net=bury_net)
    return generate(net, starts, search, rng, device, cfg, cache, agree=agree)


def _save(rows: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    rows.save(str(tmp))
    os.replace(tmp, out)


def _single_thread() -> None:
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# --- a fixed shard, one process per core ------------------------------------


def _pool_worker(args: tuple) -> tuple[str, dict]:
    checkpoint, hands, search, seed, out = args
    _single_thread()
    agent = load_agent(checkpoint)
    rows, stats = label_hands(agent, hands, search, np.random.default_rng(seed))
    _save(rows, Path(out))
    return out, stats


def generate_pool(checkpoint: str, hands: int, search: Any, seed: list[int],
                  workers: int, out: Path) -> dict:
    """``hands`` hands split over ``workers`` processes, merged into
    ``out``. -> summed stats."""
    import multiprocessing as mp

    from .distill import DistillRows, load_rows

    workers = max(1, min(int(workers), int(hands)))
    each = [hands // workers + (1 if w < hands % workers else 0) for w in range(workers)]
    parts = [out.with_name(f"{out.stem}.part{w:02d}.npz") for w in range(workers)]
    jobs = [(checkpoint, each[w], search, [*seed, w], str(parts[w])) for w in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_pool_worker, jobs)
    rows = load_rows(str(out.with_name(f"{out.stem}.part*.npz")))
    _save(rows, out)
    for p in parts:
        p.unlink(missing_ok=True)
    total: dict[str, float] = {}
    for _path, stats in results:
        for k in ("decisions", "searched", "search_changed"):
            total[k] = total.get(k, 0.0) + float(stats[k])
    total["rows"] = float(len(rows))
    assert isinstance(rows, DistillRows)
    return total


# --- following a training run -----------------------------------------------


def newest_checkpoint(run_dir: str | Path) -> tuple[Path, int] | None:
    """The run's newest ``checkpoints/step_*.pt`` and its step."""
    best: tuple[Path, int] | None = None
    for p in Path(run_dir, "checkpoints").glob("step_*.pt"):
        m = _STEP.search(p.name)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (p, int(m.group(1)))
    return best


def initial_checkpoint(run_dir: str | Path, spec: str) -> tuple[Path, int] | None:
    """What a follower labels from before the run has written a checkpoint
    of its own: ``spec`` itself, or with ``auto`` the checkpoint the run
    resumed from, which ``relaunch.resolve_resume`` records in
    ``launch.json`` under Slurm. With its step; None until there is one.

    A run resumed from step 5000 writes its first checkpoint a checkpoint
    interval later, and until then a follower that only watched
    ``checkpoints/`` would sit idle and the run would train undistilled.
    """
    import torch

    if spec == "auto":
        marker = Path(run_dir) / "launch.json"
        try:
            spec = json.loads(marker.read_text()).get("resume", "")
        except (FileNotFoundError, ValueError):
            return None
    if not spec or not Path(spec).is_file():
        return None
    blob = torch.load(str(spec), map_location="cpu", weights_only=False, mmap=True)
    return Path(spec), int(blob.get("step", 0))


def shard_name(step: int, task: int, worker: int, n: int) -> str:
    return f"s{step:07d}-t{task:03d}-w{worker:02d}-{n % 100000:05d}.npz"


def shard_step(path: str | Path) -> int:
    m = _SHARD.search(str(path))
    if m is None:
        raise ValueError(f"not a label shard: {path}")
    return int(m.group(1))


def _run_finished(run_dir: Path, step: int) -> bool:
    """The run wrote ``final.pt`` and nothing newer is left to label."""
    final = run_dir / "checkpoints" / "final.pt"
    newest = newest_checkpoint(run_dir)
    return final.exists() and (newest is None or newest[1] <= step)


def follow(run_dir: str | Path, out_dir: str | Path, hands: int, search: Any,
           seed: int, task: int = 0, worker: int = 0, keep: int = 50,
           poll_s: float = 60.0, max_shards: int | None = None,
           log: Any = print, initial: str = "", agree: bool = False) -> int:
    """Label shards from ``run_dir``'s newest checkpoint until the run
    finishes (or ``max_shards``). -> shards written.

    Each shard is ``hands`` hands from the newest checkpoint at the time
    it starts. This worker's own shards beyond the newest ``keep`` are
    deleted, so several workers and array tasks share one directory
    without racing. Until the run has a checkpoint, ``initial``
    (``initial_checkpoint``) stands in for it; "" waits. ``agree`` keeps
    only labels two searches agree on.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mine = sorted(out_dir.glob(f"s*-t{task:03d}-w{worker:02d}-*.npz"))
    n = max((int(_SHARD.search(p.name).group(4)) + 1 for p in mine), default=0)
    agent = None
    loaded_step = -1
    first: tuple[Path, int] | None = None
    written = 0
    while max_shards is None or written < max_shards:
        found = newest_checkpoint(run_dir)
        if found is None:
            if (run_dir / "checkpoints" / "final.pt").exists():
                return written
            if initial and first is None:
                first = initial_checkpoint(run_dir, initial)
            if first is None:
                time.sleep(poll_s)
                continue
            found = first
        path, step = found
        if step != loaded_step:
            try:
                agent = load_agent(path)
            except FileNotFoundError:  # pruned between the glob and the load
                continue
            loaded_step = step
        t0 = time.perf_counter()
        rng = np.random.default_rng([seed, task, worker, n])
        try:
            rows, stats = label_hands(agent, hands, search, rng, agree=agree)
        except RuntimeError:
            if not agree:
                raise
            # Agreement left no label in these few hands: the next shard.
            n += 1
            if _run_finished(run_dir, loaded_step):
                return written
            continue
        out = out_dir / shard_name(loaded_step, task, worker, n)
        _save(rows, out)
        n += 1
        written += 1
        kept = f", {stats['search_agreed']:.0f} agreed" if agree else ""
        log(f"[t{task} w{worker}] {out.name}: {len(rows)} rows from step {loaded_step} "
            f"in {time.perf_counter() - t0:.0f} s "
            f"({stats['searched']:.0f} searched of {stats['decisions']:.0f}{kept})")
        mine = sorted(out_dir.glob(f"s*-t{task:03d}-w{worker:02d}-*.npz"),
                      key=lambda p: (shard_step(p), p.name))
        for old in mine[: max(0, len(mine) - keep)]:
            old.unlink(missing_ok=True)
        if _run_finished(run_dir, loaded_step):
            return written
    return written


def _follow_worker(kwargs: dict) -> int:
    _single_thread()
    return follow(**kwargs)


def follow_pool(workers: int, **kwargs: Any) -> int:
    """``follow`` in ``workers`` processes, one per core."""
    import multiprocessing as mp

    jobs = [{**kwargs, "worker": w} for w in range(int(workers))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return sum(pool.map(_follow_worker, jobs))


# --- the trainer's side -------------------------------------------------------


class FreshRows:
    """The newest label shards matching ``pattern``, up to ``window_rows``
    rows (0: every shard), reloading only shards not already held.

    ``max_age`` > 0 also leaves out shards labelled more than that many
    steps before the step ``load`` is given, so a follower that falls
    behind (queued, requeued) does not leave the trainer fitting old
    labels for as long as nothing newer arrives.
    """

    def __init__(self, pattern: str, window_rows: int = 0, max_age: int = 0) -> None:
        self.pattern = pattern
        self.window_rows = int(window_rows)
        self.max_age = int(max_age)
        self._held: dict[str, Any] = {}

    def load(self, step: int | None = None) -> tuple[Any, dict[str, float]]:
        """``(DistillRows or None, metrics)``; ``step``, the trainer's,
        measures the labels' age."""
        from .distill import DistillRows

        paths = [p for p in glob.glob(self.pattern) if not p.endswith(".tmp.npz")]

        def key(p: str) -> tuple:
            try:
                return (shard_step(p), os.path.getmtime(p))
            except (ValueError, FileNotFoundError):
                return (-1, 0.0)

        too_old = 0
        if self.max_age > 0 and step is not None:
            fresh = [p for p in paths if key(p)[0] >= step - self.max_age]
            too_old = len(paths) - len(fresh)
            paths = fresh
        paths.sort(key=key, reverse=True)
        chosen: list[str] = []
        total = 0
        for p in paths:
            if p not in self._held:
                try:
                    with np.load(p) as z:
                        self._held[p] = DistillRows(**{f.name: z[f.name] for f in fields(DistillRows)})
                except (FileNotFoundError, OSError, ValueError, KeyError):
                    continue  # pruned or still being replaced; next time
            chosen.append(p)
            total += len(self._held[p])
            if self.window_rows and total >= self.window_rows:
                break
        for p in list(self._held):
            if p not in chosen:
                del self._held[p]
        aged = {"distill/shards_too_old": float(too_old)} if self.max_age > 0 else {}
        if not chosen:
            return None, {"distill/rows": 0.0, "distill/shards": 0.0, **aged}
        parts = [self._held[p] for p in chosen]
        rows = DistillRows(**{f.name: np.concatenate([getattr(r, f.name) for r in parts])
                              for f in fields(DistillRows)})
        steps = [s for s in (key(p)[0] for p in chosen) if s >= 0]
        metrics = {"distill/rows": float(len(rows)), "distill/shards": float(len(chosen)),
                   **aged}
        if steps:
            metrics["distill/label_step_min"] = float(min(steps))
            metrics["distill/label_step_max"] = float(max(steps))
        if step is not None:
            # Steps since each row's checkpoint, weighted by rows.
            known = [(step - key(p)[0], len(self._held[p])) for p in chosen if key(p)[0] >= 0]
            if known:
                n = sum(w for _, w in known)
                metrics["distill/label_age_mean"] = float(sum(a * w for a, w in known) / n)
                metrics["distill/label_age_max"] = float(max(a for a, _ in known))
        return rows, metrics
