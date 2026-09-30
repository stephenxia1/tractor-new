"""Where a training job starts: fresh, resumed, or picked up again after
Modal rescheduled it.

Modal re-runs an input whose worker disappears, with the arguments it was
first called with. For a job launched fresh that is a second run from step
0 into the same directory: `pop8b`'s worker vanished at step 934, and the
rerun spent eleven hours retracing it, its checkpoints overwriting the
first run's by step number. So a job records the call that started it in
``<run>/launch.json``; a start that finds its own call there resumes from
the run's newest checkpoint, and a fresh start that finds another job's
checkpoints refuses rather than training over them.
"""

from __future__ import annotations

import json
from pathlib import Path

MARKER = "launch.json"


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """The newest ``*.pt`` written in ``ckpt_dir``: ``final.pt`` after a run
    that finished, the last ``step_*.pt`` after one that was cut off."""
    found = sorted(Path(ckpt_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def resolve_resume(run_dir: str | Path, resume: str, call_id: str | None,
                   overwrite: bool = False) -> str:
    """The checkpoint a job should start from ("" for a fresh start), and
    the job recorded as the run's.

    * The call already recorded -- Modal rescheduled it: the run's newest
      checkpoint, or what it was launched from if it wrote none.
    * ``resume == "latest"``: the run's newest checkpoint.
    * A fresh start into a run that has checkpoints: refused, unless
      ``overwrite`` (a smoke test's scratch run).
    * Otherwise ``resume`` as given.

    ``call_id`` None (not on Modal) records nothing.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    marker = run_dir / MARKER
    newest = latest_checkpoint(ckpt_dir) if ckpt_dir.exists() else None
    record = json.loads(marker.read_text()) if marker.exists() else {}

    if call_id is not None and record.get("call_id") == call_id:
        return str(newest) if newest is not None else record.get("resume", "")
    if resume == "latest":
        if newest is None:
            raise FileNotFoundError(f"--resume latest: no checkpoint in {ckpt_dir}")
        chosen = str(newest)
    elif not resume and newest is not None and not overwrite:
        raise FileExistsError(
            f"{run_dir} already holds checkpoints ({newest.name}); a fresh run would "
            "train over them. --resume latest continues it; a new --name starts another."
        )
    else:
        chosen = resume
    if call_id is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"call_id": call_id, "resume": chosen}))
    return chosen
