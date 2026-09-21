"""Optional Weights & Biases mirror of the metrics the trainers already log.

Both trainers build a `row` dict every logging interval and append it to
`runs/<run>.jsonl`. This mirrors that same row to wandb, so there is one
source of truth for what a metric means and the JSONL stays the thing you
can diff, grep and replay offline.

Three properties it has to have, in order of importance:

1. **A logging failure never kills a training run.** An expired key, a
   network partition or a wandb outage at hour eleven of a twelve-hour
   segment must cost the charts, not the run. Everything below is wrapped,
   and the first failure disables the mirror for the rest of the process.
2. **A resumed run continues the same chart.** The run id is derived from
   `--run-name`, with `resume="allow"`, so the Modal segment chain and any
   restart after a preemption append to one wandb run instead of starting
   a new one each time. Without this, a run that is preempted six times
   appears as seven unrelated runs.
3. **It costs nothing when it is off.** wandb is not imported at all unless
   `--wandb` is passed, so it stays out of the default dependency set.

Population rows carry per-agent lists (`steps`, `loss_q`, ...). Those are
expanded into `agent3/loss_q` series plus `loss_q/mean|min|max`, because a
list is not a chartable quantity.
"""

from __future__ import annotations

import math
import re
from typing import Any


def add_arguments(ap) -> None:
    """Register the shared flags on a trainer's ArgumentParser."""
    ap.add_argument("--wandb", action="store_true",
                    help="mirror the JSONL metrics to Weights & Biases")
    ap.add_argument("--wandb-project", type=str, default="tractor-rl")
    ap.add_argument("--wandb-entity", type=str, default="",
                    help="team or user; defaults to the key's own account")
    ap.add_argument("--wandb-group", type=str, default="",
                    help="groups the arms of one sweep together")
    ap.add_argument("--wandb-tags", type=str, default="",
                    help="comma-separated")
    ap.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def flatten(row: dict) -> dict:
    """Metrics row -> flat {name: number}, dropping what cannot be charted."""
    out: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif _finite(value):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if _finite(item):
                    out[f"agent{i}/{key}"] = item
            nums = [v for v in value if _finite(v)]
            if nums:
                out[f"{key}/mean"] = sum(nums) / len(nums)
                out[f"{key}/min"] = min(nums)
                out[f"{key}/max"] = max(nums)
        elif isinstance(value, dict):
            for sub, item in value.items():
                if _finite(item):
                    out[f"{key}/{sub}"] = item
        # None, strings and anything else belong in config, not metrics.
    return out


class Mirror:
    """A wandb run, or a no-op that quacks like one."""

    def __init__(self) -> None:
        self.run = None
        self._dead = False

    @property
    def enabled(self) -> bool:
        return self.run is not None and not self._dead

    def _disable(self, what: str, exc: Exception) -> None:
        self._dead = True
        print(f"wandb {what} failed ({type(exc).__name__}: {exc}); "
              f"continuing without it", flush=True)

    def log(self, row: dict) -> None:
        """Log one metrics row.

        No explicit `step=`. The learner does not step at all until the
        replay buffer passes `min_buffer`, so an explicit
        `step=learner.steps` sends every row of the fill phase to step 0,
        and wandb overwrites a repeated step -- the first minutes of every
        run collapse to a single point. The x-axis is instead declared at
        init as a *metric* (`learner_steps`), which wandb is happy to have
        stand still, and which is also the axis these experiments are
        compared on.
        """
        if not self.enabled:
            return
        try:
            self.run.log(flatten(row))
        except Exception as exc:  # never take the training run down
            self._disable("log", exc)

    def summary(self, **values) -> None:
        if not self.enabled:
            return
        try:
            self.run.summary.update(values)
        except Exception as exc:
            self._disable("summary", exc)

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            self._disable("finish", exc)


def start(args, config: dict | None = None,
          step_metric: str = "learner_steps") -> Mirror:
    """Open the mirror if `--wandb` was passed, else return a no-op."""
    mirror = Mirror()
    if not getattr(args, "wandb", False):
        return mirror
    try:
        import wandb
    except ImportError:
        print("wandb not installed (pip install wandb); continuing without it",
              flush=True)
        return mirror

    # wandb ids allow [A-Za-z0-9_-]; run names are ours to choose but a
    # stray character would otherwise fail init at startup.
    run_id = re.sub(r"[^A-Za-z0-9_-]", "-", args.run_name)[:120]
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    try:
        mirror.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=args.run_name,
            # id + resume is what stitches a segment chain, or a restart
            # after preemption, into one run rather than N.
            id=run_id,
            resume="allow",
            tags=tags or None,
            mode=args.wandb_mode,
            config=dict(config or {}),
        )
        # Charts default to the axis the experiments are compared on,
        # rather than to wall-clock log order.
        wandb.define_metric(step_metric)
        wandb.define_metric("*", step_metric=step_metric)
    except Exception as exc:
        print(f"wandb init failed ({type(exc).__name__}: {exc}); "
              f"continuing without it", flush=True)
        mirror.run = None
    return mirror
