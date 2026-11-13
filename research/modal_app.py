"""Runs the research scripts on Modal, started and watched over plain HTTPS.

Deploy once, from any machine with the Modal CLI logged in:

    modal deploy research/modal_app.py

Two pieces:

* ``job`` -- a large CPU container that clones one branch of this (public)
  repository, installs it, runs one research script, and keeps
  ``research/checkpoints``, ``research/runs`` and the job's own log on the
  ``tractor-runs`` Volume. Code is fetched when a job starts, so pushing to a
  branch never needs a redeploy.
* ``control`` -- a small HTTPS API to start, list, watch, cancel and fetch
  jobs. Every request needs ``Authorization: Bearer <token>``, where the token
  is ``TRACTOR_CONTROL_TOKEN`` in the ``tractor-control`` Modal Secret.

Why HTTPS rather than the Modal CLI: the client that drives these runs may
sit behind a proxy that carries HTTPS but not gRPC, which is what the Modal
CLI speaks. A web endpoint is reachable from anywhere a browser is.

What a token holder can do is deliberately narrow: run a script that is
already in this repository (``research/py/*.py`` or ``research/src/*.ts``),
from a branch of it (``main``, ``rl-research`` or ``claude/...``), with
arguments passed as a list -- never through a shell. Only collaborators can
create those branches.

Restarts: the job function retries after an infrastructure failure. A
``train2.py`` job that restarts resumes from its own latest checkpoint
rather than from the checkpoint it was started with, so a crash costs at
most the time since the last checkpoint (10 minutes by default).
"""


import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import modal

APP_NAME = "tractor-research"
VOLUME_NAME = "tractor-runs"
SECRET_NAME = "tractor-control"
REPO_URL = "https://github.com/stephenxia1/tractor-new.git"
VOLUME_ROOT = "/vol"
WORK_DIR = "/root/work"

ALLOWED_REF = re.compile(r"^(main|rl-research|claude/[A-Za-z0-9._/-]+)$")
ALLOWED_SCRIPT = re.compile(r"^(py/[a-z0-9_]+\.py|src/[A-Za-z0-9_]+\.ts)$")
ALLOWED_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
# Directories the research scripts write to, kept on the Volume.
PERSISTED = ("checkpoints", "runs")
COMMIT_EVERY_S = 300

DEFAULT_CPU = 16.0
DEFAULT_MEMORY_MB = 32768
MAX_CPU = 64.0
MAX_MEMORY_MB = 262144

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    .pip_install("numpy", "fastapi[standard]")
    # CPU build: the training loop has no GPU path today, and the CUDA wheel
    # is several gigabytes of libraries it would never load.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
)


# ---------------------------------------------------------------------------
# The job runner. Plain Python, so it can be exercised without Modal.
# ---------------------------------------------------------------------------


def _log(logf, msg: str) -> None:
    logf.write(f"[runner {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}\n")
    logf.flush()


def _cpu_quota() -> str:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return "unlimited" if quota == "max" else f"{int(quota) / int(period):.1f}"
    except (OSError, ValueError):
        return "unknown"


def _resume_own_checkpoint(args: list[str], vol: Path) -> tuple[list[str], str | None]:
    """For a restarted train2 job, resume from the run's own latest checkpoint."""
    if "--run-name" not in args:
        return args, None
    i = args.index("--run-name")
    if i + 1 >= len(args):
        return args, None
    latest = vol / "checkpoints" / f"{args[i + 1]}_latest.pt"
    if not latest.exists():
        return args, None
    rel = f"checkpoints/{latest.name}"
    out = list(args)
    if "--resume" in out:
        j = out.index("--resume")
        if j + 1 < len(out):
            out[j + 1] = rel
        else:
            out.append(rel)
    else:
        out += ["--resume", rel]
    return out, rel


def run_job(
    vol: Path,
    job_id: str,
    ref: str,
    script: str,
    args: list[str],
    *,
    work_dir: Path = Path(WORK_DIR),
    repo_url: str = REPO_URL,
    commit: Callable[[], None] = lambda: None,
    python: str = sys.executable,
) -> int:
    """Clones `ref`, runs `script` from research/, returns its exit code."""
    if not ALLOWED_REF.match(ref) or not ALLOWED_SCRIPT.match(script):
        raise ValueError(f"refusing ref={ref!r} script={script!r}")
    job_dir = vol / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "log.txt").open("a", encoding="utf-8") as logf:
        _log(logf, f"attempt starting: ref={ref} script={script} args={args}")
        _log(logf, f"cpu quota {_cpu_quota()}, os.cpu_count {os.cpu_count()}")

        if work_dir.exists():
            shutil.rmtree(work_dir)
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", ref, repo_url, str(work_dir)],
            stdout=logf, stderr=subprocess.STDOUT)
        if clone.returncode != 0:
            _log(logf, f"git clone failed ({clone.returncode})")
            return clone.returncode
        sha = subprocess.run(["git", "-C", str(work_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        _log(logf, f"checked out {sha}")
        npm = subprocess.run(["npm", "ci", "--include=dev", "--no-audit", "--no-fund"],
                             cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
        if npm.returncode != 0:
            _log(logf, f"npm ci failed ({npm.returncode})")
            return npm.returncode

        research = work_dir / "research"
        for name in PERSISTED:
            target = vol / name
            target.mkdir(parents=True, exist_ok=True)
            link = research / name
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.exists():
                shutil.rmtree(link)
            link.symlink_to(target, target_is_directory=True)

        if script == "py/train2.py":
            args, resumed = _resume_own_checkpoint(args, vol)
            if resumed is not None:
                _log(logf, f"run already has a checkpoint; resuming from {resumed}")

        cmd = ([python, "-u", script] if script.endswith(".py") else ["npx", "tsx", script]) + args
        env = {**os.environ, "PYTHONPATH": str(research / "py")}
        with (job_dir / "attempts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"started": time.time(), "commit": sha, "cmd": cmd[1:]}) + "\n")
        commit()

        stop = threading.Event()

        def committer() -> None:
            while not stop.wait(COMMIT_EVERY_S):
                try:
                    commit()
                except Exception as err:  # a failed commit must not kill the run
                    _log(logf, f"volume commit failed: {err!r}")

        threading.Thread(target=committer, daemon=True).start()
        _log(logf, f"running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=research, env=env, stdout=logf, stderr=subprocess.STDOUT)
        stop.set()
        _log(logf, f"exit code {proc.returncode}")
    (job_dir / "exit_code").write_text(str(proc.returncode))
    commit()
    return proc.returncode


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=24 * 3600,
    volumes={VOLUME_ROOT: volume},
    # Infrastructure failures only: a script's own nonzero exit is returned,
    # not raised, so a failing evaluation is not run three times.
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
)
def job(job_id: str, ref: str, script: str, args: list[str]) -> int:
    return run_job(Path(VOLUME_ROOT), job_id, ref, script, args, commit=volume.commit)


# ---------------------------------------------------------------------------
# The control API. Built by a factory so it can be tested without Modal.
# ---------------------------------------------------------------------------


def make_api(
    vol: Path,
    token: str,
    spawn: Callable[[str, str, str, list[str], float, int], str],
    call_status: Callable[[str], dict],
    cancel: Callable[[str], None],
    reload: Callable[[], None] = lambda: None,
    commit: Callable[[], None] = lambda: None,
):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, PlainTextResponse
    from pydantic import BaseModel, Field

    api = FastAPI(title="tractor research control")

    def authorised(authorization: str = Header(default="")) -> None:
        if not token:
            raise HTTPException(500, "TRACTOR_CONTROL_TOKEN is not set")
        given = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "bad or missing bearer token")

    class JobRequest(BaseModel):
        ref: str
        script: str
        args: list[str] = Field(default_factory=list)
        cpu: float = DEFAULT_CPU
        memory_mb: int = DEFAULT_MEMORY_MB
        job_id: str | None = None

    def job_dir(job_id: str) -> Path:
        if not ALLOWED_JOB_ID.match(job_id):
            raise HTTPException(400, "bad job id")
        d = vol / "jobs" / job_id
        if not (d / "job.json").exists():
            raise HTTPException(404, f"no job {job_id}")
        return d

    def describe(d: Path, log_lines: int = 0) -> dict:
        meta = json.loads((d / "job.json").read_text())
        exit_file = d / "exit_code"
        if exit_file.exists():
            meta["status"] = {"state": "finished", "exit_code": int(exit_file.read_text())}
        else:
            meta["status"] = call_status(meta["call_id"])
        attempts = d / "attempts.jsonl"
        meta["attempts"] = (
            [json.loads(x) for x in attempts.read_text().splitlines() if x.strip()]
            if attempts.exists() else [])
        if log_lines > 0 and (d / "log.txt").exists():
            meta["log_tail"] = (d / "log.txt").read_text(errors="replace").splitlines()[-log_lines:]
        return meta

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    @api.post("/jobs", dependencies=[Depends(authorised)])
    def start(req: JobRequest) -> dict:
        if not ALLOWED_REF.match(req.ref):
            raise HTTPException(400, f"ref {req.ref!r} is not an allowed branch")
        if not ALLOWED_SCRIPT.match(req.script):
            raise HTTPException(400, f"script {req.script!r} is not under py/ or src/")
        if not (0 < req.cpu <= MAX_CPU) or not (1024 <= req.memory_mb <= MAX_MEMORY_MB):
            raise HTTPException(400, "cpu or memory out of range")
        reload()
        job_id = req.job_id or time.strftime("%Y%m%d-%H%M%S") + "-" + Path(req.script).stem
        if not ALLOWED_JOB_ID.match(job_id):
            raise HTTPException(400, "bad job id")
        d = vol / "jobs" / job_id
        if (d / "job.json").exists():
            raise HTTPException(409, f"job {job_id} exists")
        call_id = spawn(job_id, req.ref, req.script, req.args, req.cpu, req.memory_mb)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"job_id": job_id, "call_id": call_id, "ref": req.ref, "script": req.script,
                "args": req.args, "cpu": req.cpu, "memory_mb": req.memory_mb,
                "created": time.time()}
        (d / "job.json").write_text(json.dumps(meta, indent=2))
        commit()
        return meta

    @api.get("/jobs", dependencies=[Depends(authorised)])
    def list_jobs() -> list[dict]:
        reload()
        root = vol / "jobs"
        if not root.exists():
            return []
        dirs = sorted((p for p in root.iterdir() if (p / "job.json").exists()), reverse=True)
        return [describe(p) for p in dirs[:50]]

    @api.get("/jobs/{job_id}", dependencies=[Depends(authorised)])
    def get_job(job_id: str, lines: int = 60) -> dict:
        reload()
        return describe(job_dir(job_id), log_lines=max(0, min(lines, 2000)))

    @api.get("/jobs/{job_id}/log", dependencies=[Depends(authorised)])
    def get_log(job_id: str, offset: int = 0) -> PlainTextResponse:
        reload()
        log = job_dir(job_id) / "log.txt"
        if not log.exists():
            return PlainTextResponse("", headers={"X-Log-Size": "0"})
        data = log.read_bytes()
        return PlainTextResponse(data[max(0, offset):].decode(errors="replace"),
                                 headers={"X-Log-Size": str(len(data))})

    @api.post("/jobs/{job_id}/cancel", dependencies=[Depends(authorised)])
    def cancel_job(job_id: str) -> dict:
        reload()
        meta = json.loads((job_dir(job_id) / "job.json").read_text())
        cancel(meta["call_id"])
        return {"cancelled": job_id}

    def inside(rel: str) -> Path:
        p = (vol / rel).resolve()
        if p != vol.resolve() and vol.resolve() not in p.parents:
            raise HTTPException(400, "path escapes the volume")
        return p

    @api.get("/files", dependencies=[Depends(authorised)])
    def list_files(prefix: str = "") -> list[dict]:
        reload()
        base = inside(prefix)
        if not base.exists():
            return []
        items = [base] if base.is_file() else sorted(base.rglob("*"))
        root = vol.resolve()
        return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                 "modified": p.stat().st_mtime} for p in items if p.is_file()][:2000]

    @api.get("/file", dependencies=[Depends(authorised)])
    def get_file(path: str) -> FileResponse:
        reload()
        p = inside(path)
        if not p.is_file():
            raise HTTPException(404, f"no file {path}")
        return FileResponse(p)

    return api


def _call_status(call_id: str) -> dict:
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return {"state": "running"}
    except modal.exception.OutputExpiredError:
        return {"state": "finished", "exit_code": None}
    except Exception as err:  # the job raised, or Modal gave up retrying it
        return {"state": "failed", "error": repr(err)[:500]}
    return {"state": "finished", "exit_code": result}


def _spawn(job_id: str, ref: str, script: str, args: list[str], cpu: float, memory_mb: int) -> str:
    fn = job.with_options(cpu=cpu, memory=memory_mb)
    return fn.spawn(job_id, ref, script, args).object_id


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["TRACTOR_CONTROL_TOKEN"])],
)
@modal.asgi_app(label="tractor-control")
def control():
    return make_api(
        Path(VOLUME_ROOT),
        os.environ.get("TRACTOR_CONTROL_TOKEN", ""),
        spawn=_spawn,
        call_status=_call_status,
        cancel=lambda call_id: modal.FunctionCall.from_id(call_id).cancel(),
        reload=volume.reload,
        commit=volume.commit,
    )
