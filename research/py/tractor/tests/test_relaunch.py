"""Where a training job starts, including after Modal reschedules it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..train.relaunch import resolve_resume


def _ckpt(run: Path, name: str, mtime: float) -> Path:
    path = run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


def test_a_fresh_start_into_an_empty_run_starts_fresh(tmp_path: Path) -> None:
    assert resolve_resume(tmp_path / "run", "", "fc-1") == ""
    assert (tmp_path / "run" / "launch.json").exists()


def test_a_rescheduled_job_resumes_from_its_newest_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert resolve_resume(run, "", "fc-1") == ""
    _ckpt(run, "step_0000050.pt", 100)
    newest = _ckpt(run, "step_0000100.pt", 200)
    # The same call again: Modal re-running the input after losing its worker.
    assert resolve_resume(run, "", "fc-1") == str(newest)


def test_a_rescheduled_job_with_no_checkpoint_yet_keeps_its_launch_resume(
        tmp_path: Path) -> None:
    run = tmp_path / "run"
    source = "/vol/pop8b/milestones/step_0000900.pt"
    assert resolve_resume(run, source, "fc-2") == source
    assert resolve_resume(run, source, "fc-2") == source
    newest = _ckpt(run, "step_0000950.pt", 300)
    assert resolve_resume(run, source, "fc-2") == str(newest)


def test_a_fresh_start_over_another_jobs_checkpoints_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "run"
    resolve_resume(run, "", "fc-1")
    _ckpt(run, "step_0000050.pt", 100)
    with pytest.raises(FileExistsError, match="already holds checkpoints"):
        resolve_resume(run, "", "fc-9")
    # Off Modal too: a local fresh start is refused the same way.
    with pytest.raises(FileExistsError):
        resolve_resume(run, "", None)
    # A scratch run may say so.
    assert resolve_resume(run, "", "fc-9", overwrite=True) == ""


def test_latest_is_the_newest_checkpoint_and_needs_one(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        resolve_resume(run, "latest", "fc-1")
    _ckpt(run, "step_0000100.pt", 100)
    final = _ckpt(run, "final.pt", 200)
    assert resolve_resume(run, "latest", "fc-3") == str(final)


def test_off_modal_nothing_is_recorded(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert resolve_resume(run, "", None) == ""
    assert not (run / "launch.json").exists()
