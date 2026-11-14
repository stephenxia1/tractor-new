"""scripts/modal_apps.py: what it will and will not stop."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "modal_apps", Path(__file__).resolve().parents[1] / "scripts" / "modal_apps.py")
modal_apps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(modal_apps)


def test_only_project_apps_and_never_the_bridge() -> None:
    modal_apps.check_stoppable("tractor-train-zero1")
    for bad in ("tractor-research", "other-app", "tractor-", "tractor-x; rm", "TRACTOR-A"):
        with pytest.raises(SystemExit):
            modal_apps.check_stoppable(bad)


def test_running_ids_matches_name_and_skips_stopped() -> None:
    rows = [
        {"App ID": "ap-1", "Description": "tractor-train-zero1", "State": "ephemeral (detached)"},
        {"App ID": "ap-2", "Description": "tractor-train-zero1", "State": "stopped"},
        {"App ID": "ap-3", "Description": "tractor-train-zero10", "State": "ephemeral"},
        {"App ID": "ap-4", "Description": "tractor-eval", "State": "ephemeral"},
    ]
    assert modal_apps.running_ids(rows, "tractor-train-zero1") == ["ap-1"]
