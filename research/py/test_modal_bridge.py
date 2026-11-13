"""Checks the Modal bridge's control API, commands and resume logic.

    PYTHONPATH=py python py/test_modal_bridge.py      (from research/)

Needs the `modal` and `fastapi` packages importable (no Modal account or
network): the API is built with fake spawn/status functions over
temporary directories standing in for the two volumes.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import modal_bridge as B
    from fastapi.testclient import TestClient
except ImportError as err:  # pragma: no cover
    print(f"skip: {err}")
    sys.exit(0)


def check_commands() -> None:
    py = "/usr/bin/python3"
    assert B.build_command("script", "py/train2.py", ["--x"], py) == [py, "-u", "py/train2.py", "--x"]
    assert B.build_command("script", "src/selftest.ts", [], py) == ["npx", "tsx", "src/selftest.ts"]
    assert B.build_command("module", "tractor.eval.bots", ["--deals", "4"], py) == [
        py, "-u", "-m", "tractor.eval.bots", "--deals", "4"]
    assert B.build_command("modal", "modal_eval.py::smart", ["--checkpoints", "c"], py) == [
        "modal", "run", "--detach", "modal_eval.py::smart", "--checkpoints", "c"]
    ok = [("script", "py/tractor/eval/audit.py"), ("script", "scripts/tab_evals.py"),
          ("module", "tractor.eval.bots"), ("modal", "modal_train.py::launch")]
    for kind, target in ok:
        B.validate(kind, target, [], {"TRACTOR_EVAL_APP": "x"})
    bad = [("script", "../etc/passwd"), ("script", "py/../../x.py"), ("script", "scripts/run.sh"),
           ("module", "os.system"), ("modal", "modal_train.py"), ("modal", "other.py::launch"),
           ("modal", "modal_x.py::a;b"), ("shell", "ls")]
    for kind, target in bad:
        try:
            B.validate(kind, target, [], {})
        except ValueError:
            continue
        raise AssertionError(f"accepted {kind} {target}")
    for env in ({"PATH": "/x"}, {"MODAL_TOKEN_ID": "x"}, {"tractor_x": "y"}):
        try:
            B.validate("script", "py/train2.py", [], env)
        except ValueError:
            continue
        raise AssertionError(f"accepted env {env}")
    print("ok  commands: three kinds, never a shell, targets and env names restricted")


def check_api() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        runs, rl = Path(a), Path(b)
        spawned: list[tuple] = []
        cancelled: list[str] = []

        def spawn(*args):
            spawned.append(args)
            return f"fc-{len(spawned)}"

        api = B.make_api({"runs": runs, "rl": rl}, "s3cret", spawn=spawn,
                         call_status=lambda cid: {"state": "running"}, cancel=cancelled.append)
        c = TestClient(api)
        auth = {"Authorization": "Bearer s3cret"}

        assert c.get("/health").json() == {"ok": True}
        assert c.get("/jobs").status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401

        body = {"ref": "rl-research", "kind": "modal", "target": "modal_eval.py::smart",
                "args": ["--checkpoints", "/vol/v5-bury/checkpoints/final.pt"],
                "env": {"TRACTOR_EVAL_APP": "tractor-eval-x"}, "job_id": "eval-x"}
        r = c.post("/jobs", json=body, headers=auth)
        assert r.status_code == 200, r.text
        assert spawned[0] == ("eval-x", "rl-research", "modal", "modal_eval.py::smart",
                              body["args"], body["env"], 1.0, 4096)
        assert c.post("/jobs", json=body, headers=auth).status_code == 409

        # The old request shape (script, no kind) still works.
        old = {"ref": "rl-research", "script": "py/test_returns.py", "job_id": "old"}
        assert c.post("/jobs", json=old, headers=auth).status_code == 200
        assert spawned[1][2:4] == ("script", "py/test_returns.py") and spawned[1][6:] == (16.0, 32768)

        for bad in ({"ref": "evil; rm -rf /"}, {"ref": "feature/x"}, {"target": "../../x"},
                    {"kind": "shell"}, {"env": {"PATH": "/tmp"}}, {"cpu": 1000}):
            r = c.post("/jobs", json={**body, "job_id": "x", **bad}, headers=auth)
            assert r.status_code == 400, (bad, r.status_code, r.text)

        listing = c.get("/jobs", headers=auth).json()
        assert sorted(j["job_id"] for j in listing) == ["eval-x", "old"]
        d = runs / "jobs" / "eval-x"
        (d / "log.txt").write_text("line1\nline2\nline3\n")
        (d / "exit_code").write_text("0")
        got = c.get("/jobs/eval-x?lines=2", headers=auth).json()
        assert got["status"] == {"state": "finished", "exit_code": 0}
        assert got["log_tail"] == ["line2", "line3"]
        r = c.get("/jobs/eval-x/log?offset=6", headers=auth)
        assert r.text == "line2\nline3\n" and r.headers["X-Log-Size"] == "18"
        assert c.post("/jobs/eval-x/cancel", headers=auth).status_code == 200
        assert cancelled == ["fc-1"]
        assert c.get("/jobs/nope", headers=auth).status_code == 404

        # Files on either volume, whole or by tail.
        (rl / "pop8c").mkdir()
        (rl / "pop8c" / "metrics.jsonl").write_text("a\nb\nc\n")
        files = c.get("/files?volume=rl&prefix=pop8c", headers=auth).json()
        assert files[0]["path"] == "pop8c/metrics.jsonl" and files[0]["bytes"] == 6
        assert c.get("/file?volume=rl&path=pop8c/metrics.jsonl", headers=auth).text == "a\nb\nc\n"
        r = c.get("/file?volume=rl&path=pop8c/metrics.jsonl&tail_bytes=2", headers=auth)
        assert r.text == "c\n" and r.headers["X-File-Size"] == "6"
        assert c.get("/file?volume=rl&path=../../etc/passwd", headers=auth).status_code == 400
        assert c.get("/files?volume=other", headers=auth).status_code == 400
    print("ok  control API: auth, three kinds, validation, jobs, logs, both volumes")


def check_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vol = Path(tmp)
        args = ["--resume", "checkpoints/v2_latest.pt", "--run-name", "e1"]
        assert B._resume_own_checkpoint(args, vol) == (args, None)
        (vol / "checkpoints").mkdir()
        (vol / "checkpoints" / "e1_latest.pt").write_bytes(b"")
        out, rel = B._resume_own_checkpoint(args, vol)
        assert rel == "checkpoints/e1_latest.pt"
        assert out == ["--resume", "checkpoints/e1_latest.pt", "--run-name", "e1"]
    print("ok  a restarted train2 job resumes its own latest checkpoint")


if __name__ == "__main__":
    check_commands()
    check_api()
    check_resume()
