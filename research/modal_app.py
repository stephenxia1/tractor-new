r"""Run the research stack on Modal: one container per experiment arm.

Modal is here for parallelism, not price. Every open question in RESULTS.md
is an A/B question, and on one rented box the arms run in sequence -- which
is where "identical instance types differ 12-27% in throughput, so
wall-clock-matched arms end at different step counts" came from. Here each
arm is its own container with its own reservation, so a four-arm sweep
costs four times one arm, finishes in one arm's wall clock, and puts every
arm on the same hardware spec.

It also removes the two operational costs of the AWS path in `scripts/`: no
spot reclaim to babysit (a failed container restarts and resumes from the
checkpoint on the volume), and no source staging through S3.

Setup, once:

    pip install modal
    modal setup                     # opens a browser, writes ~/.modal.toml

Then, from the repo root or from research/:

    # one training run, detached so closing the laptop does not kill it
    modal run --detach research/modal_app.py::train \
        --run-name m1 --minutes 300 --size a10g \
        --args="--workers 12 --games 64 --threads 8 --groups 2"

    # several arms at once, same spec, same start
    modal run --detach research/modal_app.py::sweep \
        --run-prefix spt --minutes 300 --size a10g \
        --arms "one:--steps-per-train 1;two:--steps-per-train 2" \
        --start-from checkpoints/v2_final.pt

    # the ladder, one container per rung, in parallel
    modal run research/modal_app.py::ladder \
        --checkpoint checkpoints/m1_latest.pt --label m1 --pairs 1000

    # artifacts
    modal volume ls  tractor-rl checkpoints
    modal volume get tractor-rl checkpoints/m1_latest.pt .
    modal volume put tractor-rl checkpoints/v2_final.pt checkpoints/v2_final.pt

`--args=` needs the equals sign: a value starting with `--` is read as a
flag otherwise. `--detach` is what makes a run outlive the local process;
without it, Ctrl-C stops the container too.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parents[1]

# Paths inside the container. Not derived from __file__: Modal also imports
# this module in the container, where __file__ is wherever it put the source.
REPO = "/app"
RESEARCH = f"{REPO}/research"
VOL_PATH = "/vol"

# Pinned to the dev laptop's versions so a checkpoint moves both ways.
TORCH = "torch==2.6.0"
NUMPY = "numpy==2.2.5"

app = modal.App("tractor-rl")

# Checkpoints and run metrics. One volume, two directories; the container
# symlinks research/{checkpoints,runs} at it, so no training code needs a
# path flag it does not already have.
volume = modal.Volume.from_name("tractor-rl", create_if_missing=True)

# Mirrors research/Dockerfile: Node and torch in one image, because the two
# halves of the loop talk over pipes and have to live together. The layer
# order is the same too -- manifests, npm ci, then source -- so a source
# edit rebuilds in seconds instead of re-running npm ci.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "git", "procps")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
    )
    # The default Linux wheels are the CUDA builds, which is what we want on
    # the GPU sizes and is harmless on the CPU ones.
    .pip_install(NUMPY, TORCH)
    # Every workspace member's package.json must be present for `npm ci`,
    # even the ones training never imports.
    .add_local_file(LOCAL_REPO / "package.json", f"{REPO}/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "package-lock.json", f"{REPO}/package-lock.json", copy=True)
    .add_local_file(LOCAL_REPO / "tsconfig.base.json", f"{REPO}/tsconfig.base.json", copy=True)
    .add_local_file(LOCAL_REPO / "apps/ws/package.json", f"{REPO}/apps/ws/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "apps/web/package.json", f"{REPO}/apps/web/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "packages/engine/package.json",
                    f"{REPO}/packages/engine/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "packages/protocol/package.json",
                    f"{REPO}/packages/protocol/package.json", copy=True)
    .run_commands(f"cd {REPO} && npm ci --include=dev")
    # Its own layer, after npm ci: putting it next to torch would rebuild
    # the npm layer too, and that is the slow one.
    .pip_install("wandb")
    .add_local_dir(
        LOCAL_REPO / "packages/engine", f"{REPO}/packages/engine", copy=True,
        ignore=["**/node_modules", "**/dist", "**/__pycache__", "**/*.pt", "python"],
    )
    .add_local_dir(
        LOCAL_REPO / "packages/protocol", f"{REPO}/packages/protocol", copy=True,
        ignore=["**/node_modules", "**/dist"],
    )
    .add_local_dir(
        LOCAL_REPO / "research", f"{REPO}/research", copy=True,
        # checkpoints/ and runs/ are the volume's job, and are gigabytes.
        ignore=["**/node_modules", "**/__pycache__", "**/*.pt",
                "checkpoints", "runs", "data"],
    )
    .add_local_file(LOCAL_REPO / "rules.md", f"{REPO}/rules.md", copy=True)
    .env({"PYTHONPATH": f"{RESEARCH}/py", "PYTHONUNBUFFERED": "1"})
    .workdir(RESEARCH)
    # Same reason the Dockerfile does this: fail on a broken image in
    # seconds rather than an hour into a training run. Only re-runs when a
    # layer above it changes.
    .run_commands(
        f"cd {RESEARCH} && npx tsx src/dupcheck.ts --hands 3 --games 1",
        f"cd {RESEARCH} && python py/test_returns.py",
    )
)

# GPU/CPU shapes. The decorator below declares no GPU, so a size can *add*
# one: `with_options(gpu=None)` means "no override", not "no GPU".
#
# Sizing rationale from RESULTS.md: the loop is torch-bound, but the Node
# workers need about a core each to feed it, so cores matter as much as the
# card. A modest GPU with 16 cores beats a large GPU with 8.
SIZES: dict[str, dict] = {
    "cpu8":  {"cpu": 8.0,  "memory": 16384},
    "cpu16": {"cpu": 16.0, "memory": 32768},
    "cpu32": {"cpu": 32.0, "memory": 65536},
    # a10g is the default: measured 1.74x the learner throughput of both
    # t4 and l4 on an identical fixture batch, and cheaper per step even
    # at the higher hourly rate. l4 and t4 are a tie with each other.
    "t4":    {"cpu": 12.0, "memory": 24576, "gpu": "T4"},
    "l4":    {"cpu": 16.0, "memory": 32768, "gpu": "L4"},
    "a10g":  {"cpu": 16.0, "memory": 32768, "gpu": "A10G"},
    "l40s":  {"cpu": 16.0, "memory": 65536, "gpu": "L40S"},
    "a100":  {"cpu": 16.0, "memory": 65536, "gpu": "A100-40GB"},
}

MAX_TIMEOUT = 24 * 3600


def _options(size: str, minutes: float) -> dict:
    if size not in SIZES:
        raise SystemExit(f"unknown --size {size!r}; choose from {', '.join(SIZES)}")
    # Headroom over --minutes for the image pull, the final checkpoint write
    # and the closing volume commit.
    timeout = int(minutes * 60) + 1200
    if timeout > MAX_TIMEOUT:
        raise SystemExit(
            f"--minutes {minutes} exceeds Modal's 24h container limit; use "
            f"--minutes {int((MAX_TIMEOUT - 1200) / 60)} --segments N instead"
        )
    return {**SIZES[size], "timeout": timeout}


# Both spellings of the wandb secret, most-preferred first. `wandb` is what
# this file originally asked for; `wandb-secret` is what Modal's own docs
# example produces. Which one exists is an accident of how it was created.
_WANDB_SECRET_NAMES = ("wandb", "wandb-secret")


def _reject_seed_flags_on_chain(extra: list[str], segments: int) -> None:
    """Refuse a hand-written seed flag on a multi-segment chain.

    `extra` reaches every segment unchanged, and `train_segment` lets an
    explicit `--resume`/`--init-from` win over its own resume logic. For
    one segment that is the point. For a chain it is silently destructive:
    segment 2 starts from the seed again instead of the checkpoint segment
    1 just wrote, so N segments buy one segment of progress while every
    log line looks healthy and the step counter restarts where a reader
    would expect a resume. `--start-from` exists for this -- it is applied
    only when the run has no checkpoint of its own yet, so it seeds
    segment 1 and yields from segment 2 on.

    Raised rather than warned because the run costs money and the symptom
    is invisible until someone reads the step counts across segments.
    """
    for hand in ("--init-from", "--resume"):
        if hand in extra and segments > 1:
            raise SystemExit(
                f"{hand} was passed in --args together with --segments "
                f"{segments}. It is re-applied to every segment, so each one "
                f"would restart from that checkpoint and discard the previous "
                f"segment's work. Use --start-from <path inside the volume>, "
                f"which seeds only the first segment."
            )


def _existing_runs() -> set[str]:
    """Run names that already have a checkpoint, read from the client.

    `resume="auto"` is what makes a preemption survivable, but it also
    means relaunching a sweep under a prefix that was used before silently
    *continues* those runs instead of starting them. The arms would then
    differ by more than the flags being compared, which is the one thing a
    sweep exists to rule out.
    """
    try:
        entries = volume.listdir("checkpoints")
    except Exception:
        return set()
    return {Path(e.path).name[: -len("_latest.pt")]
            for e in entries if e.path.endswith("_latest.pt")}


def _volume_has(path: str) -> bool:
    """Does `path` exist in the volume? Checked from the client.

    A wrong checkpoint path is otherwise discovered once per container,
    after each has started and pulled a multi-gigabyte image -- nine rungs
    of a ladder all failing the same way, several minutes in.
    """
    head, _, tail = path.strip("/").rpartition("/")
    try:
        entries = volume.listdir(head or "/")
    except Exception:
        return False
    return any(e.path.rstrip("/").rsplit("/", 1)[-1] == tail for e in entries)


def _wandb_modes(extra: list[str]) -> list[str]:
    """Every `--wandb-mode` value in `extra`, in both spellings.

    `--wandb-mode` is a trainer flag rather than an entrypoint one, so it
    only ever arrives inside `--args` / `--common`, where this parser is the
    single thing that reads it. argparse accepts `--wandb-mode offline` and
    `--wandb-mode=offline` alike and neither is the obviously correct one to
    write, so missing one spelling would silently change what the run does.
    """
    modes = []
    for i, token in enumerate(extra):
        if token == "--wandb-mode" and i + 1 < len(extra):
            modes.append(extra[i + 1])
        elif token.startswith("--wandb-mode="):
            modes.append(token.split("=", 1)[1])
    return modes


def _wandb_keyless(extra: list[str]) -> bool:
    """Does this run never reach the wandb API? Offline and disabled do not."""
    modes = _wandb_modes(extra)
    return bool(modes) and all(m in ("offline", "disabled") for m in modes)


def _wandb_setup(enabled: bool, project: str, group: str, extra: list[str]):
    """Trainer flags and the Modal secret for a wandb-enabled run.

    The secret is attached only when it is asked for: naming it
    unconditionally would make every function fail to start for anyone who
    has not created it.
    """
    if not enabled:
        return extra, [], ""
    secrets, secret_name = [], ""
    # An offline or disabled run never authenticates, so demanding the
    # secret would refuse the one mode that exists precisely for a workspace
    # with no key -- which is the path MODAL.md documents as key-free.
    if not _wandb_keyless(extra):
        # Both spellings, because `modal secret create wandb-secret ...` is
        # what the Modal docs' own example produces and `wandb` is what
        # this file asked for, and the two are indistinguishable until a
        # run refuses to start. Whichever exists wins; the key check is
        # the same either way.
        secret, failures = None, []
        for name in _WANDB_SECRET_NAMES:
            candidate = modal.Secret.from_name(name, required_keys=["WANDB_API_KEY"])
            try:
                candidate.hydrate()
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}")
                continue
            secret, secret_name = candidate, name
            break
        if secret is None:
            raise SystemExit(
                f"--wandb needs a Modal secret holding WANDB_API_KEY, named "
                f"'wandb' or 'wandb-secret' ({'; '.join(failures)}). Create "
                f"one with:\n"
                f"  modal secret create wandb WANDB_API_KEY=<your key>\n"
                f"A secret that exists but lacks the WANDB_API_KEY field "
                f"fails here too -- check the field name, not just the "
                f"secret name.")
        secrets = [secret]
    # An operator who wrote --wandb into --args/--common has already said
    # what they want, and appending our --wandb-project after theirs would
    # silently win, since argparse keeps the last value. So add no flags --
    # but still return the secret. Returning early on the flag is what this
    # got wrong: whether the container needs WANDB_API_KEY depends on the
    # *mode*, not on who typed the flag, so a hand-written online --wandb
    # produced a container with no key and a run that dropped its metrics.
    if "--wandb" in extra:
        return extra, secrets, secret_name
    flags = ["--wandb", "--wandb-project", project]
    if group:
        flags += ["--wandb-group", group]
    return extra + flags, secrets, secret_name


def _link_volume() -> None:
    """Point research/{checkpoints,runs} at the volume."""
    for name in ("checkpoints", "runs"):
        target = Path(VOL_PATH) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(RESEARCH) / name
        if link.is_symlink():
            continue
        if link.exists():
            shutil.rmtree(link)
        link.symlink_to(target, target_is_directory=True)


def _commit_loop(stop: threading.Event, every: float = 300.0) -> None:
    """Commit the volume on a timer.

    Writes to a Modal volume become durable at commit, not at write. Without
    this, a container that is preempted loses everything written since it
    started; with it, the loss is bounded by --checkpoint-minutes, which is
    the same exposure the spot boxes had.
    """
    while not stop.wait(every):
        try:
            volume.commit()
        except Exception as exc:  # a failed commit must not kill training
            print(f"[modal] volume commit failed: {exc}", flush=True)


def _last_metrics(run_name: str) -> dict | None:
    path = Path(VOL_PATH) / "runs" / f"{run_name}.jsonl"
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    try:
        return json.loads(last) if last else None
    except json.JSONDecodeError:
        return None


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    cpu=8.0,
    memory=16384,
    timeout=3600,
    # A preempted or crashed container restarts and resumes from the
    # checkpoint on the volume, which is what resume="auto" is for.
    retries=modal.Retries(max_retries=3, initial_delay=30.0),
)
def train_segment(
    script: str,
    run_name: str,
    minutes: float,
    extra: list[str],
    segments: int = 1,
    options: dict | None = None,
    resume: str = "auto",
    start_from: str = "",
    cores: float = 8.0,
    wandb_secret: str = "",
) -> dict:
    """One training segment. Chains the next one itself if segments > 1."""
    _link_volume()

    cmd = [sys.executable, "-u", script, "--run-name", run_name,
           "--minutes", str(minutes)]
    # Order matters: the run's own latest checkpoint always wins over the
    # seed. Otherwise a retry after a preemption, or the second segment of
    # a chain, would start the arm again from the seed and silently throw
    # away every step it had taken.
    latest = Path(VOL_PATH) / "checkpoints" / f"{run_name}_latest.pt"
    if resume not in ("auto", "off"):
        # Guarded here as well as in the entrypoints: this used to fall
        # through to no flag at all, so a run asked to continue a
        # checkpoint started from random init instead and said nothing.
        raise ValueError(f"resume must be 'auto' or 'off', got {resume!r}; "
                         f"use start_from for a specific checkpoint")
    if "--resume" in extra or "--init-from" in extra:
        # Explicit flags win over both -- which is fine for one segment and
        # silently destructive for a chain. `extra` is passed to every
        # segment unchanged, so a hand-written `--init-from seed.pt` makes
        # segment 2 start from the seed again instead of resuming segment
        # 1's checkpoint, and the chain then runs `segments` times for one
        # segment of progress while every log line looks healthy. That is
        # the failure the comment above describes; the ordering only
        # prevents it for the `start_from` spelling, which this branch
        # jumps over. `--start-from` is the flag that survives a chain.
        _reject_seed_flags_on_chain(extra, segments)
    elif resume == "auto" and latest.exists():
        cmd += ["--resume", str(latest)]
        print(f"[modal] resuming {latest}", flush=True)
    elif start_from:
        # train2 seeds through --resume; train_pop has a separate
        # --init-from, which copies one net into all N agents.
        flag = "--init-from" if "train_pop" in script else "--resume"
        cmd += [flag, str(Path(VOL_PATH) / start_from)]
        print(f"[modal] seeding {flag} {start_from}", flush=True)
    # A Modal container sees the HOST's cores, not its reservation:
    # measured os.cpu_count() == 24 inside a cpu=8 container, with no
    # cgroup quota. So train2's defaults (--threads 22, --workers 10),
    # written for a big box, would set 22 torch threads on 8 reserved
    # cores -- precisely the actor-starvation pathology in RESULTS.md,
    # where every 6-thread config beat every 8+-thread config. Modal does
    # export OMP_NUM_THREADS=<reservation>, but train2 overrides it with
    # torch.set_num_threads(args.threads).
    #
    # These are a sane split of the reservation, not a tuned config: run
    # py/tune_learner.py on a new size and pass the answer in --args.
    #
    # An even split, on GPU sizes too. Shifting cores from torch to actors
    # was tried on a10g/16 and measured worse by the objective that counts:
    #
    #   8 threads / 8 workers -> 22.8 steps/s, 11.9k dec/s, replay 1.25
    #   4 threads / 12 workers -> 20.0 steps/s, 15.5k dec/s, replay 0.81
    #
    # +30% experience for -12.5% gradient steps, with the replay ratio
    # falling below one -- the box then never samples roughly a fifth of
    # the experience it stored. (These read 0.98 and 0.66 until the ratio
    # was recomputed over the decisions actually STORED rather than all
    # decisions generated; only ~78-81% of them are, since the snapshot's
    # picks are thrown away.) py/tune_learner.py's premise is that
    # once the environment is not the bottleneck the objective is steps/s
    # at a sane replay ratio, and decisions/s stops being informative. The
    # low learner share in the profile (pct_learn 34) says the actors are
    # cheap, not that more of them would help.
    n = max(2, int(cores))
    threads = max(2, n // 2)
    injected = []
    if "--threads" not in extra:
        cmd += ["--threads", str(threads)]
        injected.append(f"--threads {threads}")
    if "--workers" not in extra:
        cmd += ["--workers", str(max(2, n - threads))]
        injected.append(f"--workers {max(2, n - threads)}")
    if injected:
        print(f"[modal] {' '.join(injected)} from the {n}-core reservation "
              f"(not tuned; see py/tune_learner.py)", flush=True)
    cmd += extra

    # An offline wandb run writes under the working directory, which is
    # container-local and discarded on exit -- the whole point of offline
    # mode is to sync it later, so it has to land on the volume. Online
    # runs keep their staging files on the container disk, where they do
    # not churn the volume for data already sent.
    env = None
    if "--wandb" in extra and "offline" in _wandb_modes(extra):
        wandb_dir = Path(VOL_PATH) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "WANDB_DIR": str(wandb_dir)}
        print(f"[modal] offline wandb runs -> {wandb_dir} (on the volume; "
              f"`modal volume get tractor-rl wandb ...` then `wandb sync`)",
              flush=True)

    print(f"[modal] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
    committer.start()
    # Modal DOES retry a container that hits its own timeout -- measured,
    # max_retries=2 produced three attempts. Left to the platform, a hung
    # segment would therefore run four times: at --minutes 720 that is ~49
    # A10G-hours burned in silence. Enforcing the deadline here turns a
    # hang into one wasted segment that stops the chain instead.
    deadline = minutes * 60 + 600
    timed_out = False
    # start_new_session so the trainer leads its own process group: it
    # spawns N Node workers, and proc.kill() would signal only the Python
    # parent, orphaning the workers to keep burning the reservation on a
    # container Modal may hand to the next input.
    proc = subprocess.Popen(cmd, cwd=RESEARCH, start_new_session=True, env=env)
    try:
        rc = proc.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        timed_out = True
        # SIGTERM first, and wait: the trainers install a handler that
        # checkpoints and exits, so a deadline costs the minutes since the
        # last periodic save rather than everything the segment learned.
        # SIGKILL runs neither that handler nor the `finally: save(final=True)`
        # underneath it. The escalation still has to exist, because the point
        # of the deadline is that a hung trainer stops burning the
        # reservation, and the 1200s of headroom _options leaves over
        # --minutes is what pays for the grace period.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            rc = proc.wait(timeout=120)
            print("[modal] SIGTERM: trainer stopped and checkpointed", flush=True)
        except subprocess.TimeoutExpired:
            print("[modal] SIGTERM ignored for 120s; escalating to SIGKILL, "
                  "so the final checkpoint is lost", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                rc = -9
    finally:
        stop.set()
        committer.join(timeout=30)
        volume.commit()

    metrics = _last_metrics(run_name)
    print(f"[modal] exit {rc}; last metrics: {metrics}", flush=True)
    if timed_out:
        # Returned, not raised: raising would hand it back to the retry
        # policy, which is the thing being avoided.
        print(f"[modal] {script} exceeded {deadline:.0f}s and was killed; "
              f"chain stopped. The checkpoint on the volume is intact.",
              flush=True)
        return {"run_name": run_name, "returncode": rc, "timed_out": True,
                "metrics": metrics}
    if rc != 0:
        raise RuntimeError(f"{script} exited {rc}")

    # Chained last, so a crash anywhere above is retried rather than
    # silently handing the run on to a fresh segment.
    if segments > 1:
        # with_options replaces rather than merges, so the secret has to be
        # re-attached here: without it segment 2 starts with no
        # WANDB_API_KEY and the mirror silently degrades to nothing for the
        # rest of a multi-day chain.
        opts = dict(options or {})
        if wandb_secret:
            # The NAME the launching client already resolved, not a fresh
            # guess. Re-deriving it here hardcoded "wandb", so a workspace
            # whose secret is called "wandb-secret" ran segment 1 happily
            # and then lost the whole chain to `Secret 'wandb' not found`
            # at spawn -- after the GPU time was already spent.
            opts["secrets"] = [modal.Secret.from_name(
                wandb_secret, required_keys=["WANDB_API_KEY"])]
        fn = train_segment.with_options(**opts) if opts else train_segment
        try:
            call = fn.spawn(
                script=script, run_name=run_name, minutes=minutes, extra=extra,
                segments=segments - 1, options=options, resume="auto",
                start_from=start_from, cores=cores,
                wandb_secret=wandb_secret,
            )
        except Exception as exc:
            # Raising here would fail the segment that just SUCCEEDED, and
            # the retry policy would re-run its whole --minutes. A dropped
            # chain costs one relaunch; a retried 12-hour segment costs 12
            # GPU-hours.
            print(f"[modal] chain spawn failed ({exc}); {segments - 1} "
                  f"segment(s) not started. Relaunch with: modal run --detach "
                  f"research/modal_app.py::train --run-name {run_name} "
                  f"--minutes {minutes} --segments {segments - 1}", flush=True)
        else:
            print(f"[modal] chained, {segments - 1} segment(s) left: "
                  f"{call.object_id}", flush=True)

    return {"run_name": run_name, "returncode": rc, "metrics": metrics}


@app.function(image=image, volumes={VOL_PATH: volume}, cpu=8.0, memory=16384,
              timeout=6 * 3600)
def arena_rung(opponent: str, checkpoint: str, label: str, pairs: int,
               base: int, seconds: float, threads: int) -> dict | None:
    """One ladder rung. Fanned out, so the ladder costs its slowest rung."""
    _link_volume()
    sys.path.insert(0, f"{RESEARCH}/py")
    import ladder  # the parser and the failed-rung record live there

    row = ladder.run_one(str(Path(VOL_PATH) / checkpoint), opponent,
                         pairs, base, seconds, threads)
    # Committed to the volume before it is returned, because the return
    # value only exists in the client process. A search rung is hours of
    # billed compute, and a client that dies -- laptop asleep, network
    # gone, Ctrl-C -- takes every rung's result with it otherwise. What
    # lands here survives that and is collected with
    # `modal volume get tractor-rl runs/ladder_<label> .`.
    if row is not None:
        out = Path(VOL_PATH) / "runs" / f"ladder_{label}" / f"{opponent}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1), encoding="utf-8")
        volume.commit()
    return row


@app.function(image=image, volumes={VOL_PATH: volume}, cpu=8.0, memory=16384,
              timeout=1800)
def check(expect_gpu: bool = False) -> None:
    """Full verification of the image: wire format, returns, no leak.

    `expect_gpu` is set by `verify` when the requested --size declares a
    card. None of the four subprocesses below touch CUDA, so without it
    this function cannot fail on the GPU path at all -- which is the one
    thing running it on a GPU size is for.
    """
    _link_volume()
    for cmd in (
        ["npx", "tsx", "src/dupcheck.ts", "--hands", "3", "--games", "1"],
        ["npx", "tsx", "src/selftest.ts"],
        [sys.executable, "py/test_returns.py"],
        [sys.executable, "py/test_no_leak.py"],
    ):
        print(f"[modal] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=RESEARCH, check=True)
    import torch
    available = torch.cuda.is_available()
    print(f"[modal] torch {torch.__version__} cuda={available}", flush=True)
    if not expect_gpu:
        # A CPU size with cuda=False is correct, not a failure.
        return
    # Printed and returned 0 before. train2 under `--device auto` falls back
    # to cpu without comment, so a CPU-only wheel or a missing driver turned
    # a 12-hour A10G segment into a 12-hour CPU segment and the check that
    # exists to catch it had already said "pass".
    if not available:
        raise RuntimeError(
            f"torch.cuda.is_available() is False on a GPU size. Do not start "
            f"anything long here: train2 --device auto would silently fall "
            f"back to cpu. Check the image still resolves the CUDA build of "
            f"{TORCH} (a CPU-only wheel installs without complaint) and that "
            f"the size really carries a card.")
    # is_available() only says a driver and a runtime loaded. A wheel built
    # without kernels for this card's SM still passes it and dies at the
    # first launch, so launch one -- and .item() it, because a CUDA failure
    # is asynchronous and would otherwise surface somewhere else entirely.
    try:
        probe = torch.randn(512, 512, device="cuda")
        checksum = float((probe @ probe).sum().item())
    except Exception as exc:
        raise RuntimeError(
            f"CUDA reports available but the first kernel failed "
            f"({type(exc).__name__}: {exc}). That is normally a torch build "
            f"with no kernels for this card's architecture: pin a {TORCH} "
            f"variant that has them, or pick a --size whose GPU it supports."
        ) from exc
    print(f"[modal] cuda ok: {torch.cuda.get_device_name(0)}, "
          f"matmul probe {checksum:.3f}", flush=True)


@app.function(image=image, volumes={VOL_PATH: volume}, cpu=8.0, memory=16384,
              timeout=1800)
def bench_device(label: str, extra: list[str]) -> dict:
    """One `bench_gpu_configs.py` run, for comparing cards."""
    _link_volume()
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cmd = [sys.executable, "-u", "py/bench_gpu_configs.py", *extra]
    print(f"[modal] {label} ({device}): {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], flush=True)
    return {"label": label, "device": device, "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr[-4000:]}


# "  device featurise, bf16 + TF32              7.41     134.9". The label
# cannot be [^\d]*: "fp32", "bf16" and "TF32" all contain digits.
BENCH_ROW = re.compile(r"^\s+((?:host|device) featurise.*?)\s{2,}([\d.]+)\s+([\d.]+)\s*$")


@app.local_entrypoint()
def bench(sizes: str = "t4,l4,a10g", reps: int = 30, threads: int = 8,
          cpu: float = 8.0, memory: int = 16384) -> None:
    """Compare cards on one learner step, everything else held constant.

    Same fixture batch, same initial weights, same core count on every arm,
    so the only difference is the card. `--fixture` matters here: filling
    the buffer from live workers would let a faster host feed a different
    batch to each device and quietly fold actor speed into the answer.
    """
    # fixtures/records.npz is gitignored (*.npz), so it reaches the image
    # only from this working copy. Absent, every container would start,
    # pull the image, and only then die in bench_gpu_configs. Check here.
    fixture = LOCAL_REPO / "research" / "fixtures" / "records.npz"
    if not fixture.exists():
        raise SystemExit(
            f"missing {fixture} -- regenerate it where the engine runs "
            f"with: PYTHONPATH=py python py/make_fixture.py")

    names = [s.strip() for s in sizes.split(",") if s.strip()]
    extra = ["--device", "cuda", "--fixture", "--reps", str(reps),
             "--threads", str(threads)]

    calls = []
    for name in names:
        if name not in SIZES:
            raise SystemExit(f"unknown size {name!r}; choose from {', '.join(SIZES)}")
        gpu = SIZES[name].get("gpu")
        if gpu is None:
            raise SystemExit(f"--sizes must name GPU sizes; {name!r} is CPU-only")
        # cpu/memory held equal across arms; only `gpu` varies.
        options = {"gpu": gpu, "cpu": cpu, "memory": memory, "timeout": 1800}
        calls.append((name, bench_device.with_options(**options)
                      .spawn(label=name, extra=extra)))

    table: dict[str, dict[str, float]] = {}
    devices: dict[str, str] = {}
    for name, call in calls:
        try:
            result = call.get()
        except Exception as exc:
            # Same reasoning as the ladder: a card that dies should cost its
            # own column, not the whole comparison.
            print(f"{name}: container failed: {type(exc).__name__}: {exc}")
            devices[name] = "FAILED"
            continue
        devices[name] = result["device"]
        if result["returncode"] != 0:
            print(f"{name}: FAILED\n{result['stderr']}")
            continue
        for line in result["stdout"].splitlines():
            # The numerics table below reuses the same row labels, so stop
            # at its header -- otherwise loss_q overwrites ms/step.
            if "loss_q" in line:
                break
            m = BENCH_ROW.match(line)
            if m:
                table.setdefault(m.group(1).strip(), {})[name] = float(m.group(2))

    print(f"\nms/step ({reps} reps, {cpu:g} cores, threads {threads})\n")
    header = "".join(f"{n:>14}" for n in names)
    print(f"{'config':<34}{header}")
    print("-" * (34 + 14 * len(names)))
    for config, row in table.items():
        cells = "".join(f"{row[n]:>14.2f}" if n in row else f"{'--':>14}" for n in names)
        print(f"{config:<34}{cells}")
    print()
    for name in names:
        print(f"  {name:<6} {devices.get(name, '?')}")


@app.local_entrypoint()
def verify(size: str = "cpu8") -> None:
    """Run the image checks on a given --size.

    `modal run ...::check` exercises the CPU path only, so a GPU size can
    be wrong -- no driver, a CPU-only torch wheel -- in a way that first
    shows up an hour into an overnight run. Run this against a size once
    before committing anything long to it.

    On a size that declares a GPU this now exits nonzero if CUDA is absent
    or if the first kernel launch fails; on a CPU size cuda=False is the
    expected answer and is not an error.
    """
    options = _options(size, 10.0)
    check.with_options(**options).remote(expect_gpu="gpu" in options)


@app.local_entrypoint()
def train(run_name: str, script: str = "py/train2.py", minutes: float = 300.0,
          size: str = "a10g", args: str = "", segments: int = 1,
          resume: str = "auto", start_from: str = "", wandb: bool = False,
          wandb_project: str = "tractor-rl", wait: bool = False) -> None:
    """Launch one training run. Use --detach so it outlives this process."""
    if resume not in ("auto", "off"):
        raise SystemExit(
            f"--resume takes 'auto' or 'off', not {resume!r}. To start from a "
            f"specific checkpoint use --start-from <path inside the volume>.")
    if start_from and not _volume_has(start_from):
        raise SystemExit(f"--start-from {start_from} is not in the volume")
    options = _options(size, minutes)
    extra, secrets, secret_name = _wandb_setup(
        wandb, wandb_project, "", shlex.split(args))
    # Locally too, so the chain is refused before a container is billed;
    # train_segment repeats the check for `sweep` and for direct calls.
    _reject_seed_flags_on_chain(extra, segments)
    fn = train_segment.with_options(**options, secrets=secrets)
    kwargs = dict(script=script, run_name=run_name, minutes=minutes,
                  extra=extra, segments=segments,
                  options=options, resume=resume, start_from=start_from,
                  cores=options["cpu"], wandb_secret=secret_name)
    if run_name in _existing_runs():
        print(f"note: {run_name} already has a checkpoint; this continues it. "
              f"Use a new --run-name to start fresh.")
    if wait:
        result = fn.remote(**kwargs)
        print(result)
        # Exit nonzero so a shell or a script can tell a killed run from a
        # finished one; the dict alone exits 0 either way.
        if result.get("timed_out"):
            raise SystemExit(1)
        return
    call = fn.spawn(**kwargs)
    print(f"spawned {run_name}: {call.object_id}")
    # By app *id*, not by name. `modal run --detach` creates an ephemeral
    # app, and `modal app logs tractor-rl` answers "No App with name
    # 'tractor-rl' found in the 'main' environment" for it -- the name
    # only resolves for a deployed app. Printing the id costs nothing and
    # is the difference between following the run and thinking it died.
    app_id = getattr(app, "app_id", None)
    print(f"  follow:    modal app logs {app_id or '<app-id from: modal app list>'}")
    print("  artifacts: modal volume ls tractor-rl checkpoints")


@app.local_entrypoint()
def sweep(run_prefix: str, arms: str, script: str = "py/train2.py",
          minutes: float = 300.0, size: str = "a10g", common: str = "",
          start_from: str = "", segments: int = 1, wandb: bool = False,
          wandb_project: str = "tractor-rl") -> None:
    """Launch several arms at once.

    --arms "name:flags;name:flags". Every arm also gets --common flags and
    the same --size, so the only difference between arms is the flags you
    wrote down. --start-from is a checkpoint path inside the volume that
    every arm begins from.
    """
    options = _options(size, minutes)

    specs = [spec.strip() for spec in arms.split(";") if spec.strip()]
    names = [spec.partition(":")[0].strip() for spec in specs]
    # Two arms sharing a name share a run_name, so they share
    # <run>_latest.pt: both write it and both resume from whichever wrote
    # last. The arms interleave into one corrupt run that still reports
    # plausible metrics, so this has to be refused rather than warned about.
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SystemExit(f"duplicate arm name(s): {', '.join(sorted(duplicates))}")
    # An arm name becomes a filename. A "/" in one would send the
    # checkpoint to a directory that does not exist, and the arm would die
    # minutes in rather than here.
    illegal = [n for n in names if not re.fullmatch(r"[A-Za-z0-9._-]+", n)]
    if illegal:
        raise SystemExit(f"arm name(s) must be [A-Za-z0-9._-]: {', '.join(illegal)}")
    if start_from and not _volume_has(start_from):
        raise SystemExit(f"--start-from {start_from} is not in the volume")

    existing = _existing_runs()
    clashes = [f"{run_prefix}_{n}" for n in names if f"{run_prefix}_{n}" in existing]
    if clashes:
        print(f"note: {', '.join(clashes)} already have checkpoints and will "
              f"RESUME, not restart. Use a fresh --run-prefix to start over.")

    # The arms of one sweep share a wandb group, so they arrive as one
    # comparison rather than N unrelated runs. The secret is decided from
    # the flags the arms will actually be given: this used to pass [], so
    # `--wandb-mode offline` in --common could never reach the keyless
    # short-circuit and an offline sweep still died for want of a secret it
    # would never have used. One arm that is not offline is enough to need
    # the key, so the short-circuit only applies when every arm is.
    arm_extras = [shlex.split(common) + shlex.split(spec.partition(":")[2])
                  for spec in specs]
    keyless = bool(arm_extras) and all(_wandb_keyless(e) for e in arm_extras)
    _, secrets, secret_name = _wandb_setup(wandb, wandb_project, run_prefix,
                              ["--wandb-mode", "offline"] if keyless else [])
    fn = train_segment.with_options(**options, secrets=secrets)

    for spec, arm in zip(specs, arm_extras):
        name = spec.partition(":")[0]
        extra, _, _ = _wandb_setup(wandb, wandb_project, run_prefix, arm)
        run_name = f"{run_prefix}_{name.strip()}"
        call = fn.spawn(script=script, run_name=run_name, minutes=minutes,
                        extra=extra, segments=segments, options=options,
                        resume="auto", start_from=start_from,
                        cores=options["cpu"], wandb_secret=secret_name)
        print(f"spawned {run_name}: {call.object_id}  ({' '.join(extra)})")


@app.local_entrypoint()
def ladder(checkpoint: str, label: str, rungs: str = "", pairs: int = 1000,
           base: int = 90210, seconds: float = 3600.0,
           threads: int = 4) -> None:
    """Run the ladder with one container per rung, in parallel.

    --checkpoint is a path inside the volume (`checkpoints/m1_latest.pt`).
    Upload a local one first with `modal volume put`.

    --base is py/ladder.py's and py/arena.py's default on purpose. The deal
    base *is* the comparison: two checkpoints are only on identical cards
    if they were run from the same one, so a Modal ladder at a different
    default would silently not be comparable to a local one, and nothing
    in the output would say so.
    """
    if not _volume_has(checkpoint):
        raise SystemExit(
            f"{checkpoint} is not in the tractor-rl volume. List it with "
            f"`modal volume ls tractor-rl checkpoints`, or upload one with "
            f"`modal volume put tractor-rl <local.pt> {checkpoint}`.")

    sys.path.insert(0, str(LOCAL_REPO / "research" / "py"))
    import ladder as ladder_mod

    names = [r.strip() for r in rungs.split(",") if r.strip()] or ladder_mod.DEFAULT_RUNGS
    print(f"{label}  ({checkpoint})  base {base}, {pairs} pairs, {len(names)} rungs")

    # arena_rung declares 6h, which is a default rather than a budget:
    # MODAL.md says to raise --seconds for the search rungs, and anything
    # near 21600 would have the container killed before arena reached its
    # own cap -- arena starts its clock after the model load and the pool
    # spawn and only reads it between rounds, so a rung always ends a little
    # after the number you passed. Arena's cap is graceful, leaving the loop
    # and printing the partial margin it did measure; the container kill
    # takes the whole rung. Same 1200s of headroom _options gives training.
    timeout = int(seconds) + 1200
    if timeout > MAX_TIMEOUT:
        raise SystemExit(
            f"--seconds {seconds} exceeds Modal's 24h container limit; split "
            f"the rungs across calls with a smaller --pairs instead")

    # return_exceptions: without it, one dead container (preemption, an
    # OOM on the heaviest rung) raises out of the map and discards the
    # rungs that DID finish. ladder.py deliberately records a failed rung
    # rather than dropping it, for the same reason -- a mean over
    # "whichever rungs survived" is not comparable to another ladder's.
    results = arena_rung.with_options(timeout=timeout).map(
        names,
        kwargs=dict(checkpoint=checkpoint, label=label, pairs=pairs,
                    base=base, seconds=seconds, threads=threads),
        return_exceptions=True,
    )
    out = LOCAL_REPO / "research" / "runs" / f"ladder_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, row in zip(names, results):
        if isinstance(row, Exception):
            rows.append({"opponent": name, "margin": None,
                         "error": f"{type(row).__name__}: {row}"[:140]})
        elif row is not None:
            rows.append(row)
        # Rewritten as each rung lands rather than once the map is done,
        # for the same reason py/ladder.py rewrites after every rung: the
        # search rungs at the tail run for hours, and an interruption
        # while one of them is still out must not take the cheap rungs
        # that already came back with it.
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    for row in rows:
        if row.get("margin") is None:
            print(f"  {row['opponent']:<16} NO RESULT  {row.get('error', '')}")
        else:
            sig = "  " if row["lo"] <= 0 <= row["hi"] else ("+" if row["margin"] > 0 else "-")
            print(f"  {row['opponent']:<16} {row['margin']:+7.2f}  "
                  f"[{row['lo']:+.2f}, {row['hi']:+.2f}] {sig}  "
                  f"{row['pairs']} pairs, {row['seconds']:.0f}s")
    print(f"\nwrote {out}")
    short = [(r["opponent"], r["pairs"]) for r in rows
             if r.get("margin") is not None and r.get("pairs") is not None
             and r["pairs"] < 0.8 * pairs]
    if short:
        print("  rung(s) short of the requested pairs (wider intervals): "
              + ", ".join(f"{o} {n}/{pairs}" for o, n in short))
