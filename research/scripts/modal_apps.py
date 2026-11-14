"""List or stop this project's Modal apps, from a bridge job.

    python scripts/modal_apps.py list
    python scripts/modal_apps.py stop tractor-train-zero1

The bridge (``modal_bridge.py``) launches apps with ``modal run --detach``
but has no way to stop one, and changing a running training's flags means
stopping it and resuming from its newest checkpoint. Its job containers
carry the Modal CLI and the workspace token, so this runs the CLI. It
only stops apps named ``tractor-*`` and never the bridge itself.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

BRIDGE_APP = "tractor-research"
STOPPABLE = re.compile(r"^tractor-[a-z0-9][a-z0-9-]{0,60}$")


def check_stoppable(name: str) -> None:
    if not STOPPABLE.match(name):
        raise SystemExit(f"refusing to stop {name!r}: only tractor-* apps")
    if name == BRIDGE_APP:
        raise SystemExit("refusing to stop the bridge itself")


def running_ids(rows: list[dict], name: str) -> list[str]:
    """The IDs of the apps called ``name`` that have not stopped, from
    ``modal app list --json``. A detached ``modal run`` makes an
    ephemeral app, which ``modal app stop <name>`` does not find by name."""
    out = []
    for row in rows:
        desc = row.get("Description") or row.get("description") or row.get("name")
        state = str(row.get("State") or row.get("state") or "").lower()
        app_id = row.get("App ID") or row.get("app_id") or row.get("id")
        if desc == name and app_id and "stopped" not in state:
            out.append(str(app_id))
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["list"]:
        return subprocess.run(["modal", "app", "list", "--json"]).returncode
    if len(argv) == 2 and argv[0] == "stop":
        name = argv[1]
        check_stoppable(name)
        listing = subprocess.run(["modal", "app", "list", "--json"],
                                 capture_output=True, text=True)
        if listing.returncode != 0:
            print(listing.stderr)
            return listing.returncode
        ids = running_ids(json.loads(listing.stdout), name)
        if not ids:
            print(f"no running app called {name}")
            return 1
        code = 0
        for app_id in ids:
            print(f"stopping {name} ({app_id})", flush=True)
            code |= subprocess.run(["modal", "app", "stop", app_id]).returncode
        return code
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
