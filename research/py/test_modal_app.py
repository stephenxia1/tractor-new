"""Checks the Modal launcher's control API and resume logic, without Modal.

    PYTHONPATH=py python py/test_modal_app.py      (from research/)

Needs the `modal` and `fastapi` packages importable (no Modal account or
network): the API is built with fake spawn/status functions over a temporary
directory standing in for the Volume.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import modal_app as M
    from fastapi.testclient import TestClient
except ImportError as err:  # pragma: no cover
    print(f"skip: {err}")
    sys.exit(0)


def check_api() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vol = Path(tmp)
        spawned: list[tuple] = []
        cancelled: list[str] = []

        def spawn(job_id, ref, script, args, cpu, mem):
            spawned.append((job_id, ref, script, args, cpu, mem))
            return f"fc-{len(spawned)}"

        api = M.make_api(vol, "s3cret", spawn=spawn,
                         call_status=lambda cid: {"state": "running"},
                         cancel=cancelled.append)
        c = TestClient(api)
        auth = {"Authorization": "Bearer s3cret"}

        assert c.get("/health").json() == {"ok": True}
        assert c.get("/jobs").status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401

        body = {"ref": "claude/adoring-sagan-s6vwwp", "script": "py/train2.py",
                "args": ["--minutes", "1", "--run-name", "e0"], "job_id": "e0-test"}
        r = c.post("/jobs", json=body, headers=auth)
        assert r.status_code == 200, r.text
        assert spawned[0][:4] == ("e0-test", body["ref"], "py/train2.py", body["args"])
        assert c.post("/jobs", json=body, headers=auth).status_code == 409

        for bad in ({"ref": "evil; rm -rf /"}, {"ref": "feature/x"},
                    {"script": "../../etc/passwd"}, {"script": "py/../../x.py"},
                    {"script": "scripts/bootstrap_vm.sh"}, {"cpu": 1000}):
            r = c.post("/jobs", json={**body, "job_id": "x", **bad}, headers=auth)
            assert r.status_code == 400, (bad, r.status_code, r.text)

        listing = c.get("/jobs", headers=auth).json()
        assert [j["job_id"] for j in listing] == ["e0-test"]
        assert listing[0]["status"] == {"state": "running"}

        d = vol / "jobs" / "e0-test"
        (d / "log.txt").write_text("line1\nline2\nline3\n")
        (d / "exit_code").write_text("0")
        got = c.get("/jobs/e0-test?lines=2", headers=auth).json()
        assert got["status"] == {"state": "finished", "exit_code": 0}
        assert got["log_tail"] == ["line2", "line3"]
        r = c.get("/jobs/e0-test/log?offset=6", headers=auth)
        assert r.text == "line2\nline3\n" and r.headers["X-Log-Size"] == "18"

        assert c.post("/jobs/e0-test/cancel", headers=auth).status_code == 200
        assert cancelled == ["fc-1"]
        assert c.get("/jobs/nope", headers=auth).status_code == 404

        (vol / "checkpoints").mkdir()
        (vol / "checkpoints" / "e0_latest.pt").write_bytes(b"x" * 10)
        files = c.get("/files?prefix=checkpoints", headers=auth).json()
        assert files[0]["path"] == "checkpoints/e0_latest.pt" and files[0]["bytes"] == 10
        assert c.get("/file?path=checkpoints/e0_latest.pt", headers=auth).content == b"x" * 10
        assert c.get("/file?path=../../etc/passwd", headers=auth).status_code == 400
    print("ok  control API: auth, validation, jobs, logs, files")


def check_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vol = Path(tmp)
        args = ["--resume", "checkpoints/v2_latest.pt", "--run-name", "e1"]
        assert M._resume_own_checkpoint(args, vol) == (args, None)
        (vol / "checkpoints").mkdir()
        (vol / "checkpoints" / "e1_latest.pt").write_bytes(b"")
        out, rel = M._resume_own_checkpoint(args, vol)
        assert rel == "checkpoints/e1_latest.pt"
        assert out == ["--resume", "checkpoints/e1_latest.pt", "--run-name", "e1"]
        out, _ = M._resume_own_checkpoint(["--run-name", "e1"], vol)
        assert out == ["--run-name", "e1", "--resume", "checkpoints/e1_latest.pt"]
    print("ok  a restarted train2 job resumes its own latest checkpoint")


if __name__ == "__main__":
    check_api()
    check_resume()
