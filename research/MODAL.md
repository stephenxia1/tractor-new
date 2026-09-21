# Running experiments on Modal

`modal_app.py` runs the research stack as **one container per experiment
arm**. That is the whole reason it exists: every open question in
`RESULTS.md` is an A/B question, and on one rented box the arms run in
sequence. Running them side by side also fixes the measurement trap
recorded there — identical instance types differ 12–27% in throughput, so
wall-clock-matched arms end at different step counts — because every arm
gets the same declared shape.

Two operational costs of the AWS path in `scripts/` go away as well: a
preempted container restarts and resumes from the checkpoint on the volume
without anyone watching, and source reaches the container from the image
build rather than through a presigned S3 URL.

---

## One-time setup

```sh
pip install modal
python -m modal setup          # opens a browser, writes ~/.modal.toml
```

On this laptop the CLI lives in the conda env and is not on `PATH`, so use
the module form with that interpreter:

```sh
alias modal='/c/Users/xiast/anaconda3/envs/tractor/python.exe -m modal'
```

Verify the image before trusting it with an overnight run — this builds it
(~5 min the first time, then cached) and runs the same checks the Dockerfile
runs, plus the privileged-leak test:

```sh
modal run research/modal_app.py::check
```

---

## The volume

One volume, `tractor-rl`, holds everything the runs produce. The container
symlinks `research/checkpoints` and `research/runs` at it, so no training
flag changes.

```sh
modal volume ls   tractor-rl checkpoints
modal volume get  tractor-rl checkpoints/m1_latest.pt .     # download
modal volume put  tractor-rl checkpoints/v2_final.pt checkpoints/v2_final.pt
```

Upload `v2_final.pt` once — it is the plateau every new arm starts from, and
`--start-from` / `--resume` read paths inside the volume.

Writes become durable at *commit*, not at write, so the training function
commits every five minutes in a background thread. A preempted container
therefore loses at most one `--checkpoint-minutes` interval, the same
exposure the spot boxes had.

---

## Training

```sh
modal run --detach research/modal_app.py::train \
    --run-name m1 --minutes 300 --size a10g \
    --args="--games 64 --groups 2"
```

* `--detach` is what makes the run outlive the local process. Without it,
  Ctrl-C kills the container.
* `--args=` needs the equals sign; a value starting with `--` is otherwise
  read as a flag.
* `--script py/train_pop.py` switches to the population trainer; everything
  else is identical.
* `--resume auto` (the default) picks up `checkpoints/<run-name>_latest.pt`
  if it exists, which is also what makes a retry after preemption safe.
  Note that `train2.py` does persist optimiser state, so a resume no longer
  resets RMSprop's second moments.

The launch prints a call id and the follow command, and exits. Follow it
with `modal app logs <app-id>` -- by **id**, which the launch prints, not
by name: `--detach` creates an *ephemeral* app, and `modal app logs
tractor-rl` answers "No App with name 'tractor-rl' found" for one of
those, which reads like the run never started. `modal app list` recovers
the id. The Modal dashboard works either way.

The function is declared with three retries, so an OOM, a transient CUDA
error or a preemption restarts the container and resumes from the volume
without anyone watching. The cost of that is that a typo in `--args` also
fails four times — fast, since argparse rejects it in seconds, but check the
first log line of a new run rather than walking away from it.

### Runs longer than a day

A container is capped at 24 hours. `--segments N` chains segments: each one
runs `--minutes`, writes its checkpoint, and spawns the next with
`--resume auto`. Four days at twelve hours a segment:

```sh
modal run --detach research/modal_app.py::train \
    --run-name long1 --minutes 720 --segments 8 --size a10g \
    --args="--games 64 --groups 2"
```

## Sweeps

The point of the whole setup. Each arm is a separate container, launched
together, differing only in the flags named:

```sh
modal run --detach research/modal_app.py::sweep \
    --run-prefix spt --minutes 300 --size a10g \
    --common "--games 64 --groups 2" \
    --start-from checkpoints/v2_final.pt \
    --arms "one:--steps-per-train 1;two:--steps-per-train 2"
```

Arms land as `spt_one`, `spt_two` in the volume. Compare them at a matched
step count, not at matched wall clock, and judge them on the ladder rather
than on `loss_q` — a higher replay ratio lowers `loss_q` while the agent
gets worse.

## The ladder

One container per rung, run in parallel, so a ladder costs its slowest rung
instead of the sum:

```sh
modal run research/modal_app.py::ladder \
    --checkpoint checkpoints/m1_latest.pt --label m1 --pairs 1000
```

Results print as the usual table and are written locally to
`research/runs/ladder_<label>.json`.

The search rungs are the expensive ones: `smart-v18` / `smart-v34` run
69–75 s/hand filling two seats, so 1000 pairs against them is a multi-day
container, not an afternoon. Give them their own call with a smaller
`--pairs` and a longer `--seconds`, and remember that 120 pairs cannot
resolve anything under ten points.

---

## Metrics to Weights & Biases

Both trainers mirror the row they already append to `runs/<run>.jsonl`. The
JSONL stays the source of truth -- it is what you can diff, grep and replay
offline -- and wandb is a view onto the same numbers, so a metric cannot
mean two things.

```sh
modal secret create wandb WANDB_API_KEY=<your key>     # once

modal run --detach research/modal_app.py::train     --run-name m1 --minutes 300 --wandb

modal run --detach research/modal_app.py::sweep     --run-prefix spt --minutes 300 --wandb     --arms "one:--steps-per-train 1;two:--steps-per-train 2"
```

The secret is attached only when `--wandb` is passed, so nothing breaks for
anyone who has not created it, and the entrypoint checks it exists before
spawning rather than letting every container fail on start-up.

Four details that matter for how the charts read:

* **A resumed run continues one chart.** The wandb run id comes from
  `--run-name` with `resume="allow"`, so a segment chain and every restart
  after a preemption append to the same run. Without it a run preempted six
  times shows up as seven unrelated runs. The chained segment re-attaches
  the secret explicitly, because `with_options` replaces rather than merges.
* **The x-axis is `learner_steps`, not log order.** Declared through
  `define_metric`, because the learner does not step at all while the
  buffer fills -- passing `step=learner.steps` sends that whole phase to
  step 0, and wandb overwrites a repeated step. It is also the axis these
  experiments are compared on.
* **Sweep arms share a wandb group** (the `--run-prefix`), so they arrive as
  one comparison.
* **Population rows are expanded.** `train_pop` logs per-agent lists; those
  become `agent3/loss_q` series plus `loss_q/mean|min|max`.

A logging failure never takes a run down: the first one prints and disables
the mirror for the rest of the process. `--wandb-mode offline` records to
disk with no key at all, which is how the path was tested.

And the standing caution applies here more than anywhere, because wandb
makes it so easy to watch: **`loss_q` does not track playing strength.** It
falls as the replay ratio rises and the net memorises a narrow buffer. Judge
runs on the ladder.

---

## Sizing and cost

| `--size` | shape | ms/learner step | use |
| --- | --- | --- | --- |
| `cpu8` / `cpu16` / `cpu32` | 8/16/32 cores | — | evaluation, ladders |
| `t4` | T4 + 12 cores | 23.51 | cheapest GPU; ties the L4 |
| `l4` | L4 + 16 cores | 23.22 | no reason to prefer it over `t4` |
| **`a10g`** | **A10G + 16 cores** | **13.48** | **the default** |
| `l40s`, `a100` | bigger cards | unmeasured | only if measured to help |

The Node workers need about a core each to feed the learner, so prefer more
cores next to a modest card over a large card with 8 — that is the sizing
lesson from `RESULTS.md`, and it is why every GPU size here carries 12–16
cores.

Modal containers report the *host's* core count rather than their
reservation, so `train_segment` derives `--threads` and `--workers` from
the reservation itself when you do not pass them, and logs what it chose.
The split is even on every size: trading torch threads for actors on
`a10g` measured +30% decisions/s but -12.5% gradient steps, with the replay
ratio falling from 0.98 to 0.66 (see RESULTS.md). These are reasoned
defaults, not tuned ones: run
`py/tune_learner.py` on a size and pass the answer in `--args`, which
overrides the injection. Do not port the laptop's `--workers 6
--threads 6`.

The A10G is the default because it was measured, not because it is the
biggest — see the card comparison in `RESULTS.md`. Re-run it with
`modal run research/modal_app.py::bench` before adding a size.

Roughly, at Modal's published rates (CPU ≈ $0.047/core-hr, RAM ≈
$0.008/GiB-hr, T4/L4/A10G ≈ $0.59/$0.80/$1.10/hr — check modal.com/pricing,
these move): `a10g` is about $1.90/hr and `cpu32` about $2.00/hr. That is a
few times an AWS spot box per core-hour. What the premium buys is arms in
parallel with no instance to reclaim, so it is worth it for a four-arm
sweep and not worth it for one long single-agent run that a spot box can do
unattended.

---

---

## Modal behaviours that were measured, not assumed

Four platform assumptions this file depends on. Two were wrong, both in the
expensive direction, and `modal_app.py` works around them.

**A container sees the host's cores, not its reservation.** `os.cpu_count()`
reports 24 inside a `cpu=8` container, `sched_getaffinity` agrees, and there
is no cgroup quota -- the reservation is a scheduling weight, not a cap.
Modal does export `OMP_NUM_THREADS=8`, but `train2.py` overrides it with
`torch.set_num_threads(args.threads)`, whose default is 22. Left alone that
puts 22 torch threads on 8 reserved cores, which is the actor-starvation
pathology this project already paid for once. `train_segment` therefore
derives `--threads` and `--workers` from the reservation when you do not
pass them, and says so in the log.

**Timeouts are retried.** A function with `max_retries=2` that blew its
timeout ran three times. So a hung segment under the default policy would
run four times over -- about 49 A10G-hours at `--minutes 720`, silently.
The deadline is now enforced inside the function, which kills the process
group, commits, and returns rather than raising; raising would hand it
straight back to the retry policy.

**Detached multi-spawn survives the client.** `modal run --detach` prints a
warning that it "only keeps the last triggered Modal function alive", which
would break every sweep. Tested with three arms and an immediately-killed
client: 3/3 finished. The warning is narrower than it reads.

**Concurrent arms can share one volume.** Three containers writing their own
files and committing every few seconds all ended up visible to each other
(after `reload()`) and to the client. Arms write disjoint paths by
`run_name`, so a sweep needs no volume-per-arm.

## Where each piece lives

| | |
| --- | --- |
| `modal_app.py` | image, volume, functions, entrypoints |
| `scripts/` | the AWS/SSM path — still works, unchanged |
| `Dockerfile` | the same image for a plain `docker run`; `modal_app.py` mirrors its layer order |
