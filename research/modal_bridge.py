"""Drive this repository's Modal work over plain HTTPS.

Deployed by `.github/workflows/modal-ops.yml` on a push that changes this
file, or once by hand:

    modal deploy research/modal_bridge.py

It is for a client that cannot run the Modal CLI itself -- a cloud agent
session whose network carries HTTPS but not gRPC, which is what the CLI
speaks. Such a client calls the ``control`` web endpoint, and a ``job``
container does what a laptop would: clone a branch of this (public)
repository, install it, and run one command from ``research/``. Code is
fetched when a job starts, so a push needs no redeploy.

A job is one of three kinds:

* ``script`` -- ``python research/<path>`` for ``py/**.py`` or
  ``scripts/*.py``, or ``npx tsx`` for ``src/*.ts``;
* ``module`` -- ``python -m tractor.<...>`` with ``PYTHONPATH=py``;
* ``modal`` -- ``modal run --detach <modal_*.py>::<function> <args>``,
  i.e. any of ``modal_train.py``/``modal_eval.py``/``modal_perf.py``'s
  entrypoints, launched exactly as from a checkout (they build their images
  from the checkout). Their work then runs in their own apps and writes to
  ``tractor-rl`` as usual; the job's log holds what the entrypoint prints.

Storage: the bridge's own job logs live on ``tractor-runs`` (``/vol``), and
``tractor-rl`` (``/rl``, the training runs, milestones and evaluations) is
mounted too, so ``/files`` and ``/file`` can read either.

Every request except ``/health`` needs ``Authorization: Bearer <token>``,
the ``TRACTOR_CONTROL_TOKEN`` in the ``tractor-control`` Modal Secret. The
``modal`` kind runs the CLI with the Modal API token in the
``tractor-modal-cli`` secret. What a token holder can run is narrow: code
already in this repository, from ``main``, ``rl-research`` or
``claude/...`` branches (only collaborators can create those), with
arguments passed as a list and never through a shell. Environment
variables may be set only if their names start with ``TRACTOR_``.
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
RUNS_VOLUME = "tractor-runs"
RL_VOLUME = "tractor-rl"
CONTROL_SECRET = "tractor-control"
CLI_SECRET = "tractor-modal-cli"
REPO_URL = "https://github.com/stephenxia1/tractor-new.git"
VOL = "/vol"
RL = "/rl"
WORK_DIR = "/root/work"

ALLOWED_REF = re.compile(r"^(main|rl-research|claude/[A-Za-z0-9._/-]+)$")
ALLOWED_SCRIPT = re.compile(
    r"^(py/[a-z0-9_]+(/[a-z0-9_]+)*\.py|scripts/[a-z0-9_]+\.py|src/[A-Za-z0-9_]+\.ts)$")
ALLOWED_MODULE = re.compile(r"^tractor(\.[a-z0-9_]+)+$")
ALLOWED_ENTRYPOINT = re.compile(r"^modal_[a-z0-9_]+\.py::[a-z_][a-z0-9_]*$")
ALLOWED_ENV = re.compile(r"^TRACTOR_[A-Z0-9_]{1,40}$")
ALLOWED_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
KINDS = ("script", "module", "modal")
# Directories the research scripts write to, kept on the bridge's volume.
PERSISTED = ("checkpoints", "runs")
COMMIT_EVERY_S = 300

DEFAULT_CPU = {"script": 16.0, "module": 16.0, "modal": 1.0}
DEFAULT_MEMORY_MB = {"script": 32768, "module": 32768, "modal": 4096}
MAX_CPU = 64.0
MAX_MEMORY_MB = 262144

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
rl_volume = modal.Volume.from_name(RL_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    .pip_install("numpy", "fastapi[standard]", "modal", "pytest")
    # CPU build: GPU work goes through the repository's own Modal apps (the
    # `modal` kind), not through these containers.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
)


# ---------------------------------------------------------------------------
# Commands and the job runner. Plain Python, so they can be tested without
# Modal.
# ---------------------------------------------------------------------------


def validate(kind: str, target: str, args: list[str], env: dict[str, str]) -> None:
    """Raises ValueError unless (kind, target, args, env) is allowed."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    pattern = {"script": ALLOWED_SCRIPT, "module": ALLOWED_MODULE, "modal": ALLOWED_ENTRYPOINT}[kind]
    if not pattern.match(target) or ".." in target:
        raise ValueError(f"{kind} target {target!r} is not allowed")
    if not all(isinstance(a, str) for a in args):
        raise ValueError("args must be strings")
    for name in env:
        if not ALLOWED_ENV.match(name):
            raise ValueError(f"environment variable {name!r} is not allowed (TRACTOR_* only)")


def build_command(kind: str, target: str, args: list[str], python: str = sys.executable) -> list[str]:
    """The argv a job runs from research/. Never a shell string."""
    if kind == "script":
        if target.endswith(".ts"):
            return ["npx", "tsx", target, *args]
        return [python, "-u", target, *args]
    if kind == "module":
        return [python, "-u", "-m", target, *args]
    return ["modal", "run", "--detach", target, *args]


def _log(logf, msg: str) -> None:
    logf.write(f"[bridge {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}\n")
    logf.flush()


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
    kind: str,
    target: str,
    args: list[str],
    env: dict[str, str] | None = None,
    *,
    work_dir: Path = Path(WORK_DIR),
    repo_url: str = REPO_URL,
    commit: Callable[[], None] = lambda: None,
    python: str = sys.executable,
) -> int:
    """Clones `ref`, runs one command from research/, returns its exit code."""
    env = env or {}
    if not ALLOWED_REF.match(ref):
        raise ValueError(f"ref {ref!r} is not allowed")
    validate(kind, target, args, env)
    job_dir = vol / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "log.txt").open("a", encoding="utf-8") as logf:
        _log(logf, f"attempt starting: ref={ref} {kind} {target} args={args} env={sorted(env)}")
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
        # The modal kind only launches; the entrypoint's own image installs
        # the workspace. Everything else runs here and needs it.
        if kind != "modal":
            npm = subprocess.run(["npm", "ci", "--include=dev", "--no-audit", "--no-fund"],
                                 cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
            if npm.returncode != 0:
                _log(logf, f"npm ci failed ({npm.returncode})")
                return npm.returncode

        research = work_dir / "research"
        for name in PERSISTED:
            target_dir = vol / name
            target_dir.mkdir(parents=True, exist_ok=True)
            link = research / name
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.exists():
                shutil.rmtree(link)
            link.symlink_to(target_dir, target_is_directory=True)

        if kind == "script" and target == "py/train2.py":
            args, resumed = _resume_own_checkpoint(args, vol)
            if resumed is not None:
                _log(logf, f"run already has a checkpoint; resuming from {resumed}")

        cmd = build_command(kind, target, args, python)
        full_env = {**os.environ, **env, "PYTHONPATH": str(research / "py")}
        with (job_dir / "attempts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"started": time.time(), "commit": sha, "cmd": cmd}) + "\n")
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
        proc = subprocess.run(cmd, cwd=research, env=full_env, stdout=logf, stderr=subprocess.STDOUT)
        stop.set()
        _log(logf, f"exit code {proc.returncode}")
    (job_dir / "exit_code").write_text(str(proc.returncode))
    commit()
    return proc.returncode


@app.function(
    image=image,
    cpu=DEFAULT_CPU["script"],
    memory=DEFAULT_MEMORY_MB["script"],
    timeout=24 * 3600,
    volumes={VOL: runs_volume, RL: rl_volume},
    secrets=[modal.Secret.from_name(CLI_SECRET, required_keys=["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])],
    # Infrastructure failures only: a command's own nonzero exit is
    # returned, not raised, so a failing evaluation is not run three times.
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
)
def job(job_id: str, ref: str, kind: str, target: str, args: list[str], env: dict[str, str]) -> int:
    return run_job(Path(VOL), job_id, ref, kind, target, args, env, commit=runs_volume.commit)


# ---------------------------------------------------------------------------
# The control API. Built by a factory so it can be tested without Modal.
# ---------------------------------------------------------------------------


def make_api(
    roots: dict[str, Path],
    token: str,
    spawn: Callable[[str, str, str, str, list[str], dict[str, str], float, int], str],
    call_status: Callable[[str], dict],
    cancel: Callable[[str], None],
    reload: Callable[[], None] = lambda: None,
    commit: Callable[[], None] = lambda: None,
):
    """`roots` maps a volume name ("runs", "rl") to where it is mounted; job
    records live under roots["runs"]."""
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, PlainTextResponse
    from pydantic import BaseModel, Field

    vol = roots["runs"]
    api = FastAPI(title="tractor research bridge")

    def authorised(authorization: str = Header(default="")) -> None:
        if not token:
            raise HTTPException(500, "TRACTOR_CONTROL_TOKEN is not set")
        given = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "bad or missing bearer token")

    class JobRequest(BaseModel):
        ref: str
        kind: str = "script"
        # For kind "script", the old field name still works.
        target: str | None = None
        script: str | None = None
        args: list[str] = Field(default_factory=list)
        env: dict[str, str] = Field(default_factory=dict)
        cpu: float | None = None
        memory_mb: int | None = None
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
        target = req.target or req.script
        if target is None:
            raise HTTPException(400, "target is required")
        if not ALLOWED_REF.match(req.ref):
            raise HTTPException(400, f"ref {req.ref!r} is not an allowed branch")
        try:
            validate(req.kind, target, req.args, req.env)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        cpu = req.cpu if req.cpu is not None else DEFAULT_CPU[req.kind]
        mem = req.memory_mb if req.memory_mb is not None else DEFAULT_MEMORY_MB[req.kind]
        if not (0 < cpu <= MAX_CPU) or not (1024 <= mem <= MAX_MEMORY_MB):
            raise HTTPException(400, "cpu or memory out of range")
        reload()
        stem = re.sub(r"[^A-Za-z0-9]+", "-", target.split("::")[-1].split("/")[-1].removesuffix(".py"))
        job_id = req.job_id or time.strftime("%Y%m%d-%H%M%S") + "-" + stem.strip("-")
        if not ALLOWED_JOB_ID.match(job_id):
            raise HTTPException(400, "bad job id")
        d = vol / "jobs" / job_id
        if (d / "job.json").exists():
            raise HTTPException(409, f"job {job_id} exists")
        call_id = spawn(job_id, req.ref, req.kind, target, req.args, req.env, cpu, mem)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"job_id": job_id, "call_id": call_id, "ref": req.ref, "kind": req.kind,
                "target": target, "args": req.args, "env": req.env, "cpu": cpu,
                "memory_mb": mem, "created": time.time()}
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

    def inside(volume: str, rel: str) -> tuple[Path, Path]:
        if volume not in roots:
            raise HTTPException(400, f"volume must be one of {sorted(roots)}")
        root = roots[volume].resolve()
        p = (root / rel.lstrip("/")).resolve()
        if p != root and root not in p.parents:
            raise HTTPException(400, "path escapes the volume")
        return root, p

    @api.get("/files", dependencies=[Depends(authorised)])
    def list_files(prefix: str = "", volume: str = "runs", depth: int = 8) -> list[dict]:
        reload()
        root, base = inside(volume, prefix)
        if not base.exists():
            return []
        if base.is_file():
            items = [base]
        else:
            items = sorted(p for p in base.rglob("*")
                           if len(p.relative_to(base).parts) <= max(1, min(depth, 32)))
        return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                 "modified": p.stat().st_mtime} for p in items if p.is_file()][:5000]

    @api.get("/file", dependencies=[Depends(authorised)])
    def get_file(path: str, volume: str = "runs", tail_bytes: int = 0):
        reload()
        _, p = inside(volume, path)
        if not p.is_file():
            raise HTTPException(404, f"no file {path}")
        if tail_bytes > 0:
            size = p.stat().st_size
            with p.open("rb") as f:
                f.seek(max(0, size - tail_bytes))
                data = f.read()
            return PlainTextResponse(data.decode(errors="replace"),
                                     headers={"X-File-Size": str(size)})
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


def _spawn(job_id: str, ref: str, kind: str, target: str, args: list[str],
           env: dict[str, str], cpu: float, memory_mb: int) -> str:
    fn = job.with_options(cpu=cpu, memory=memory_mb)
    return fn.spawn(job_id, ref, kind, target, args, env).object_id


def _reload_both() -> None:
    runs_volume.reload()
    rl_volume.reload()


@app.function(
    image=image,
    volumes={VOL: runs_volume, RL: rl_volume},
    secrets=[modal.Secret.from_name(CONTROL_SECRET, required_keys=["TRACTOR_CONTROL_TOKEN"])],
)
@modal.asgi_app(label="tractor-control")
def control():
    return make_api(
        {"runs": Path(VOL), "rl": Path(RL)},
        os.environ.get("TRACTOR_CONTROL_TOKEN", ""),
        spawn=_spawn,
        call_status=_call_status,
        cancel=lambda call_id: modal.FunctionCall.from_id(call_id).cancel(),
        reload=_reload_both,
        commit=runs_volume.commit,
    )
