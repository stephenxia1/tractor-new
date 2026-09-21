"""Population training: N agents at once, stationary within an iteration.

    PYTHONPATH=py python py/train_pop.py --agents 8 --workers 8 --minutes 360

The difference from train2.py is what an opponent is. There, one network
trained against itself and a pool of its own recent snapshots, updating
while its opponents updated -- every learner best-responding to a target
that moved underneath it. Here:

* **N agents train simultaneously**, each with its own weights, optimiser,
  critic and replay buffer. Nothing is shared, so the population really is
  a population and not one agent wearing several hats.
* **Acting weights are frozen for a whole iteration.** Each agent keeps two
  copies of its network: `acting`, which plays, and `net`, which trains.
  Only at an iteration boundary does `net` become the new `acting`. So
  within an iteration every opponent at the table is a fixed policy, which
  is the stationarity that RL best-response learning assumes.
* **Two of the four seats come from history.** Freezing alone gives
  iterated best response, which cycles. Fictitious play converges because
  the best response is computed against the *average* of past strategies,
  and `population.Archive` supplies that average by reservoir-sampling
  every publication an agent has ever made. Seating only the newest
  publication would be alpha = 1; sampling uniformly over the history makes
  it ~1/t, which is FP's schedule.

Each agent trains only on decisions taken from its own live seats. It is
therefore slightly stale with respect to its own improvements -- it plays
this iteration with last iteration's weights -- which is the deliberate
trade for facing a fixed opponent while it learns.

What is deliberately *not* here yet, both measured and both for later: the
batched GEMM across agents (worth 1.6-2.0x on a forward and backward, and
the difference between N=8 fitting on one box and not), and the async
learner. This version runs the agents' gradient steps in a plain loop, so
that a learning result cannot be confounded with a batching bug.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc2 import (
    DMC2Config, Learner2, ReplayBuffer2, decision_signs, hand_returns)
from tractor_rl.features2 import (
    action_features,
    field_u8,
    field_u16,
    field_u32,
    history_features,
    merge_batches,
    parse_episodes,
    split_per_worker,
    state_features,
)
from tractor_rl import wandb_log
from tractor_rl.model2 import NetConfig, PrivilegedCritic, TractorNet
from tractor_rl.population import (
    Archive,
    ModelRef,
    choose_archive_refs,
    live_seat_counts,
    plan_seating,
    slot_index,
)
from tractor_rl.protocol import WorkerConfig, WorkerPool

RESEARCH_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RESEARCH_DIR / "checkpoints"
RUN_DIR = RESEARCH_DIR / "runs"


def _mean(xs: list[float]) -> float:
    """Mean of the agents that have trained; NaN before any of them has."""
    good = [x for x in xs if x == x]
    return sum(good) / len(good) if good else float("nan")


class RoundStore:
    """States and chosen actions for the live decisions of one round."""

    __slots__ = ("states", "actions", "agents")

    def __init__(self, states: np.ndarray, actions: np.ndarray,
                 agents: np.ndarray) -> None:
        self.states = states
        self.actions = actions
        self.agents = agents


@dataclass
class Agent:
    idx: int
    net: TractorNet
    acting: TractorNet
    critic: PrivilegedCritic
    learner: Learner2
    buffer: ReplayBuffer2
    decisions: int = 0
    trained: int = 0
    stats: dict = field(default_factory=dict)

    def publish(self) -> None:
        """Trained weights become the weights that play."""
        self.acting.load_state_dict(self.net.state_dict())
        self.acting.eval()


@dataclass
class Slot:
    """One distinct set of weights seated this iteration."""

    ref: ModelRef
    net: TractorNet


def to_half(state: dict) -> dict:
    return {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}


def to_float(state: dict) -> dict:
    return {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def archive_path(arch_dir: Path, stored: str) -> Path:
    """Where an archive entry's weights actually live.

    Entries are stored relative to the archive directory so that an archive
    copied to another box -- the `modal volume get` resume -- still resolves
    them; stored absolute, every entry would vanish on a cross-box resume
    while the reservoir counter stayed in the hundreds. Checkpoints written
    before that change hold absolute container paths, which are taken as
    written so an in-place resume keeps working.
    """
    path = Path(stored)
    return path if path.is_absolute() else arch_dir / path


def build_slots(seating, agents: list[Agent], archive: Archive,
                scratch: list[TractorNet], device,
                arch_dir: Path) -> tuple[list[Slot], np.ndarray]:
    """One Slot per distinct set of weights, and a [table, seat] index into them.

    Distinct weights cost one forward pass per round each, so this is where
    the acting cost of a seating plan becomes visible: eight live agents plus
    an archive budget of four is twelve passes where train2.py had two.
    """
    refs, table = slot_index(seating)
    slots: list[Slot] = []
    spare = list(scratch)
    for ref in refs:
        if ref.is_live:
            net = agents[ref.agent].acting
        else:
            if not spare:
                raise RuntimeError(
                    "more archived opponents seated than scratch nets exist; "
                    "raise --archive-budget or lower the seating's demand")
            net = spare[-1]
            try:
                net.load_state_dict(to_float(torch.load(
                    archive_path(arch_dir, archive.payload(ref)),
                    map_location=device)))
            except Exception as err:  # noqa: BLE001
                # A checkpoint can name an archive file that is truncated or
                # missing, and a torn torch.save surfaces as anything from
                # UnpicklingError to EOFError depending on where the write
                # stopped -- hence the wide net. Losing the whole run over
                # one frozen opponent is the wrong trade on a spot box, so
                # forget the entry and seat that agent's own current weights
                # until the next iteration re-plans.
                print(f"dropping archive entry {ref}: "
                      f"{' '.join(str(err).split())[:140]}", flush=True)
                archive.discard(ref)
                slots.append(Slot(ref, agents[ref.agent].acting))
                continue
            spare.pop()
            net.eval()
        slots.append(Slot(ref, net))
    return slots, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8, help="one worker is one table")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--history", type=str, default="attn", choices=["attn", "none"])
    # Two of four seats live is the balance: four would give no FP averaging
    # at all, one would starve every agent of training data.
    ap.add_argument("--live-per-table", type=int, default=2)
    # Rounds an iteration lasts before weights are published and the seating
    # is reshuffled. ~22 rounds/s in practice, so 2500 is a couple of minutes:
    # short enough that FP takes many small steps rather than a few large ones.
    ap.add_argument("--iteration-rounds", type=int, default=2500)
    ap.add_argument("--archive-capacity", type=int, default=24,
                    help="reservoir slots per agent")
    ap.add_argument("--archive-burn-in", type=int, default=3,
                    help="iterations before an agent is worth seating")
    ap.add_argument("--archive-budget", type=int, default=4,
                    help="distinct archived opponents per iteration (a forward pass each)")
    # Per agent. Each agent sees roughly 1/(2N) of all decisions, so a buffer
    # sized like train2.py's would hold half an hour of increasingly stale play.
    ap.add_argument("--buffer-per-agent", type=int, default=60_000)
    ap.add_argument("--min-buffer", type=int, default=5_000)
    ap.add_argument("--batch-size", type=int, default=DMC2Config().batch_size)
    ap.add_argument("--train-every", type=int, default=1, help="rounds between passes")
    # Agents are trained round-robin, so each one takes a step every
    # `agents / agents_per_pass` rounds. At the defaults that is a step per
    # agent every eight rounds, which lands near replay 3 -- roughly what the
    # replay sweep found best -- and keeps the learner cost per round flat.
    ap.add_argument("--agents-per-pass", type=int, default=1)
    ap.add_argument("--steps-per-train", type=int, default=1, help="per agent, per pass")
    # A fixed cadence cannot be right for long. A population collects an Nth
    # of the data per agent, and how much arrives per round depends on games,
    # groups and the box; the smoke run at --games 24 over-trained at replay
    # 9.5 on a cadence that is about right at --games 64. So hold the ratio
    # rather than the cadence: an agent that is already ahead of the target
    # waits for fresh decisions instead of grinding its buffer.
    ap.add_argument("--target-replay", type=float, default=3.0,
                    help="samples trained per fresh decision, per agent; 0 disables")
    ap.add_argument("--lambda-mc", type=float, default=DMC2Config().lambda_mc)
    ap.add_argument("--lambda-mc-final", type=float, default=None)
    ap.add_argument("--lambda-anneal-start", type=int, default=0)
    ap.add_argument("--lambda-anneal-end", type=int, default=0)
    ap.add_argument("--critic-every", type=int, default=DMC2Config().critic_every)
    # No fp16: autocast in fp16 needs a GradScaler and there is none.
    ap.add_argument("--amp", choices=["off", "bf16"], default="off")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--checkpoint-minutes", type=float, default=10.0)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="learner steps per agent between numbered archives "
                         "(<run>_<steps>.pt); 0 disables. train2.py's "
                         "equivalent defaults to 100000, but a population "
                         "payload is eight agents and their optimisers, so "
                         "this stays opt-in.")
    ap.add_argument("--run-name", type=str, default="pop")
    ap.add_argument("--resume", type=str, default="")
    # Seed every agent from one trained single-agent checkpoint. From random
    # weights a population of N re-traverses the whole learning curve at 1/N
    # the compute, so testing whether population play gets *past* a plateau
    # means starting the population at that plateau. The agents still
    # diverge -- different seats, opponents, buffers and RNG -- just from a
    # competent ancestor rather than from noise.
    ap.add_argument("--init-from", type=str, default="",
                    help="a train2.py checkpoint to copy into every agent")
    wandb_log.add_arguments(ap)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RUN_DIR.mkdir(exist_ok=True)
    arch_dir = CHECKPOINT_DIR / f"{args.run_name}_archive"
    arch_dir.mkdir(exist_ok=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device)
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = DMC2Config(
        lambda_mc=args.lambda_mc, batch_size=args.batch_size,
        train_every=args.train_every, steps_per_train=args.steps_per_train,
        critic_every=args.critic_every, min_buffer=args.min_buffer,
        buffer_capacity=args.buffer_per_agent,
        lambda_mc_final=args.lambda_mc_final,
        lambda_anneal_start=args.lambda_anneal_start,
        lambda_anneal_end=args.lambda_anneal_end)
    if args.lambda_mc_final is not None:
        if args.lambda_anneal_end <= args.lambda_anneal_start:
            raise SystemExit("--lambda-anneal-end must exceed --lambda-anneal-start")
    net_cfg = NetConfig(history=args.history)
    rng = np.random.default_rng(args.seed)

    state_bytes = L.STATE_BYTES_PRIVILEGED
    agents: list[Agent] = []
    for i in range(args.agents):
        # Distinct seeds: a population initialised identically is one agent
        # with N buffers, and would answer none of the questions here.
        torch.manual_seed(args.seed * 1000 + i)
        net = TractorNet(net_cfg).to(device)
        acting = TractorNet(net_cfg).to(device)
        acting.load_state_dict(net.state_dict())
        acting.eval()
        critic = PrivilegedCritic().to(device)
        learner = Learner2(net, critic, cfg, device)
        if args.amp != "off":
            learner.amp_dtype = torch.bfloat16
        agents.append(Agent(
            idx=i, net=net, acting=acting, critic=critic, learner=learner,
            buffer=ReplayBuffer2(args.buffer_per_agent, state_bytes)))

    scratch = [TractorNet(net_cfg).to(device) for _ in range(args.archive_budget)]
    for s in scratch:
        s.eval()
    archive = Archive(args.agents, capacity=args.archive_capacity,
                      burn_in=args.archive_burn_in)

    if args.init_from:
        if args.resume:
            raise SystemExit("--init-from and --resume are mutually exclusive")
        seed_state = torch.load(args.init_from, map_location=device, weights_only=False)
        for agent in agents:
            agent.net.load_state_dict(seed_state["model"])
            agent.acting.load_state_dict(seed_state["model"])
            agent.acting.eval()
            if "critic" in seed_state:
                agent.critic.load_state_dict(seed_state["critic"])
        print(f"seeded all {len(agents)} agents from {args.init_from} "
              f"(step {seed_state.get('steps', '?')})")

    iteration = 0
    rounds_base = 0
    hands_base = 0
    # How many times this run has been restarted. A deal is a pure function
    # of (--seed, worker, game, hand), so without an offset that grows with
    # the resume count every segment replays the deals the last one already
    # trained on -- distinct-deal coverage capped by the longest single
    # segment rather than by their sum.
    segment = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        saved = len(state["agents"])
        if saved != args.agents:
            # Silently truncating loses the surplus agents' weights,
            # optimisers and step counts; silently padding leaves new agents
            # at random init while the archive restores reservoir counters
            # that assume they have a history.
            raise SystemExit(
                f"checkpoint holds {saved} agents but --agents is "
                f"{args.agents}; rerun with --agents {saved} or start fresh")
        for i, agent in enumerate(agents[:saved]):
            blob = state["agents"][i]
            agent.net.load_state_dict(blob["model"])
            agent.acting.load_state_dict(blob.get("acting", blob["model"]))
            agent.critic.load_state_dict(blob["critic"])
            for key, opt in (("opt", agent.learner.opt),
                             ("critic_opt", agent.learner.critic_opt)):
                if blob.get(key) is not None:
                    try:
                        opt.load_state_dict(blob[key])
                    except (ValueError, KeyError, RuntimeError) as err:
                        print(f"  agent {i}: dropping {key}: "
                              f"{' '.join(str(err).split())[:100]}")
            agent.learner.steps = int(blob.get("steps", 0))
            agent.decisions = int(blob.get("decisions", 0))
            agent.trained = int(blob.get("trained", 0))
        iteration = int(state.get("iteration", 0))
        rounds_base = int(state.get("rounds", 0))
        # Cumulative like rounds and iteration. Left out, a resume restarts
        # the hand count at zero while rounds keeps climbing, and the two
        # stop being comparable in the same log.
        hands_base = int(state.get("hands", 0))
        # Checkpoints written before the counter existed default to 0, so an
        # old file on the volume still resumes -- it just re-runs segment 1's
        # deals once more before the offsets start advancing.
        segment = int(state.get("segment", 0)) + 1
        # The same replay problem on the python side: the exploration draws
        # and the seating plan would otherwise repeat the previous segment's
        # sequence exactly. Absent from older checkpoints, which then simply
        # start the stream over.
        if state.get("rng_state") is not None:
            rng.bit_generator.state = state["rng_state"]
        # The archive is on local disk and is not carried in the checkpoint --
        # eight agents times twenty-four slots is over a gigabyte. A resume on
        # the same box recovers it from the index; a fresh box refills over
        # the burn-in, playing pure population self-play until it does.
        surviving = [(e["agent"], e["iteration"], e["path"])
                     for e in state.get("archive", [])
                     if archive_path(arch_dir, e["path"]).exists()]
        kept = archive.restore(surviving, state.get("archive_offered", []))
        print(f"resumed {args.resume}: iteration {iteration}, segment {segment}, "
              f"steps {[a.learner.steps for a in agents]}, archive {kept} entries")

    # The segment stride is far larger than `workers * 101` could ever reach,
    # so no two segments can land on the same worker seed and re-deal a
    # sequence already trained on.
    configs = [
        WorkerConfig(games=args.games,
                     seed=args.seed + i * 101 + 7_919_000 * segment,
                     learner_seats=0b1111, baseline="smart-v34",
                     preplay="smart-v34", privileged=True)
        for i in range(args.workers)
    ]
    pool = WorkerPool(args.workers, configs[0], configs=configs, num_groups=args.groups)

    seating = plan_seating(rng, args.agents, args.workers, args.live_per_table,
                           choose_archive_refs(rng, archive, args.archive_budget))
    slots, slot_table = build_slots(seating, agents, archive, scratch, device, arch_dir)

    print(f"net {agents[0].net.num_parameters()/1e6:.2f}M params, "
          f"critic {agents[0].critic.num_parameters()/1e6:.2f}M | device {device}")
    print(f"{args.agents} agents, {args.workers} tables, "
          f"{args.live_per_table} live seats per table, "
          f"iteration {args.iteration_rounds} rounds")
    print(f"seating: {' | '.join(' '.join(str(r) for r in row) for row in seating[:4])}"
          f"{' ...' if len(seating) > 4 else ''}")
    print(f"{len(slots)} distinct models on the board")

    log_file = (RUN_DIR / f"{args.run_name}.jsonl").open("a", encoding="utf-8")
    # Per-agent lists in the row are expanded to agentN/<metric>
    # series plus mean/min/max by wandb_log.flatten.
    mirror = wandb_log.start(args, {
        "trainer": "train_pop", "arch": "v2", "device": str(device),
        "agents": args.agents, **vars(args),
    }, step_metric="steps/max")  # agents step at their own pace
    # Defined before `save`, which closes over `rounds`: a SIGTERM arriving
    # during start-up would otherwise checkpoint into a NameError.
    rounds = decisions = hands = train_cursor = throttled = 0

    def save(final: bool = False, archive_steps: int | None = None) -> None:
        payload = {
            "arch": "pop",
            "net_config": {"history": args.history},
            "iteration": iteration,
            "rounds": rounds_base + rounds,
            "hands": hands_base + hands,
            # Which restart this is, and where the python RNG had got to.
            # Both exist so the next segment deals hands this run has not
            # already seen -- a deal is a pure function of the worker seed,
            # so a resume that does not move the seeds trains twice on one
            # deal stream.
            "segment": segment,
            "rng_state": rng.bit_generator.state,
            "agents": [{
                "model": a.net.state_dict(),
                "acting": a.acting.state_dict(),
                "critic": a.critic.state_dict(),
                # Each agent's RMSprop moments. A resume without them
                # restarts every optimiser from zero at once, which is a
                # population-wide transient rather than a single agent's.
                "opt": a.learner.opt.state_dict(),
                "critic_opt": a.learner.critic_opt.state_dict(),
                "steps": a.learner.steps,
                "decisions": a.decisions,
                "trained": a.trained,
            } for a in agents],
            "archive": [{"agent": r.agent, "iteration": r.iteration,
                         "path": archive.payload(r)}
                        for r in archive.all_refs()],
            "archive_offered": [archive.offered(a) for a in range(args.agents)],
        }
        atomic_save(payload, CHECKPOINT_DIR / f"{args.run_name}_latest.pt")
        if final:
            atomic_save(payload, CHECKPOINT_DIR / f"{args.run_name}_final.pt")
        if archive_steps is not None:
            # A numbered, immutable copy at a step watermark, which
            # `<run>_latest.pt` cannot be: it is overwritten every
            # `--checkpoint-minutes` and so only ever answers "where is the
            # run now", never "what did it look like at 50k". A matched-step
            # comparison between two runs needs the latter, and train2.py
            # has had it as `--checkpoint-every` throughout.
            atomic_save(payload, CHECKPOINT_DIR / f"{args.run_name}_{archive_steps}.pt")

    # Spot gives a termination notice and then SIGTERM; save on the way out
    # so the last minutes are not lost on top of whatever the periodic
    # checkpoint missed.
    interrupted = {"flag": False}

    def on_term(signum, _frame):
        # Record it and return; the loop below tests the flag each round
        # and leaves through its `finally`, which writes the final
        # checkpoint with nothing else in flight.
        #
        # Saving inline from here, which is what this did, re-enters
        # `save()`. The guard below only catches a *second* signal, and
        # the dangerous one is the first: arriving during the periodic
        # save it finds the flag clear, and both calls then write the
        # same deterministic `<run>_latest.pt.tmp` -- the inner one
        # renames it into place and the outer `torch.save` writes its
        # end-of-zip record at a stale offset into the file just
        # published. `atomic_save` at the iteration boundary is exposed
        # the same way and is the more expensive one to corrupt: a torn
        # `a*.pt` is loaded by every later boot whose seeded draw
        # includes it. (train2.py's handler had a second failure this
        # file does not share -- it re-entered `weights_lock`; `save`
        # here takes no lock, because the learners step on this thread.)
        if interrupted["flag"]:
            raise SystemExit(0)
        interrupted["flag"] = True
        print(f"caught signal {signum} at iteration {iteration}; "
              f"finishing the round and checkpointing", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def score(model, rows, merged, tokens_all, mask_all, sf_all, af_all):
        """Greedy candidate index for a subset of decisions, under `model`."""
        if rows.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        counts_sub = merged.cand_counts[rows]
        # A decision with one legal candidate has argmax 0 whatever the
        # network scores it, and the caller's exploration branch draws the
        # same 0 out of `rng.random() * 1`, so the forward pass cannot
        # change the action. Dropping those rows from the batch leaves
        # every other row's score untouched -- nothing in this network
        # mixes rows, the normalisations being per-row and the candidate
        # gather per-decision -- so this is an exact saving, not an
        # approximation. Measured at 9.6% of decisions on a 30k-record
        # sample, and they are the cheapest rows to lose only in
        # candidate work: each still carries the full fixed cost of the
        # trunk and the history encoder.
        multi = counts_sub > 1
        best = np.zeros(rows.shape[0], dtype=np.int64)
        if not multi.any():
            return best, counts_sub
        sub = rows[multi]
        keep = np.zeros(merged.num_decisions, dtype=bool)
        keep[sub] = True
        cand_mask = keep[merged.owner]
        pos = np.full(merged.num_decisions, -1, dtype=np.int64)
        pos[sub] = np.arange(sub.shape[0])
        new_owner = pos[merged.owner[cand_mask]]
        with torch.inference_mode():
            out = model(
                sf_all[sub], af_all[cand_mask],
                torch.from_numpy(new_owner).to(device),
                torch.from_numpy(
                    merged.states[sub, L.H_ROLE].astype(np.int64)).to(device),
                None if tokens_all is None else tokens_all[sub],
                None if mask_all is None else mask_all[sub],
                want="q")
        q = out["q"].cpu().numpy()
        counts_x = merged.cand_counts[sub]
        offsets_x = np.cumsum(counts_x) - counts_x
        max_k = int(counts_x.max())
        padded = np.full((sub.shape[0], max_k), -np.inf, dtype=np.float32)
        within = np.arange(q.shape[0]) - offsets_x[new_owner]
        padded[new_owner, within] = q
        best[multi] = padded.argmax(axis=1)
        return best, counts_sub

    pending: dict[tuple[int, int, int], list] = {}
    started = time.perf_counter()
    deadline = started + args.minutes * 60
    last_checkpoint = started
    # The grid is keyed on the slowest agent, so a file named `_50000.pt`
    # is one in which *every* agent has taken at least 50,000 steps. The
    # eight drift apart by a handful of steps (measured: 1952 to 1954 at
    # iteration 204), which is immaterial here but would make a label read
    # off agent 0 a small overstatement for the rest.
    def reached_steps() -> int:
        return min(a.learner.steps for a in agents)

    next_archive_at = 0
    if args.checkpoint_every > 0:
        next_archive_at = ((reached_steps() // args.checkpoint_every) + 1
                           ) * args.checkpoint_every
    recent_points: deque[int] = deque(maxlen=500)
    t_env = t_feat = t_fwd = t_book = t_learn = 0.0

    try:
        while not interrupted["flag"] and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            group = pool.next_group()
            group_workers = pool.workers_of(group)
            batches = pool.read_group(group)
            merged = merge_batches(batches)
            t1 = time.perf_counter()
            if merged.num_decisions == 0:
                pool.write_group(group, batches,
                                 [np.zeros(0, dtype=np.int64)] * len(batches))
                continue

            sf_all = torch.from_numpy(state_features(merged.states)).to(device)
            af_all = torch.from_numpy(
                action_features(merged.cands, merged.states, merged.owner)).to(device)
            if args.history == "attn":
                tok_np, msk_np = history_features(merged.states)
                tokens_all = torch.from_numpy(tok_np).to(device)
                mask_all = torch.from_numpy(msk_np).to(device)
            else:
                tokens_all = mask_all = None
            t2 = time.perf_counter()

            seats = field_u8(merged.states, L.H_SEAT)
            local_of = np.searchsorted(
                merged.bounds[1:], np.arange(merged.num_decisions), side="right")
            table_of = np.array([group_workers[int(j)].index for j in local_of])
            slot_of = slot_table[table_of, seats]

            picks = np.zeros(merged.num_decisions, dtype=np.int64)
            # One slot per live agent however many tables it sits at, since
            # every seat of that agent shares the same acting weights.
            live_rows_by_agent: dict[int, np.ndarray] = {}
            for k, slot in enumerate(slots):
                rows = np.nonzero(slot_of == k)[0]
                if rows.shape[0] == 0:
                    continue
                best, counts = score(slot.net, rows, merged, tokens_all,
                                     mask_all, sf_all, af_all)
                if slot.ref.is_live:
                    # Exploration belongs to the agents that learn from it;
                    # frozen opponents play their policy straight.
                    explore = rng.random(rows.shape[0]) < cfg.epsilon
                    rand_pick = (rng.random(rows.shape[0]) * counts).astype(np.int64)
                    picks[rows] = np.where(explore, rand_pick, best)
                    live_rows_by_agent[slot.ref.agent] = rows
                else:
                    picks[rows] = best
            pool.write_group(group, batches, split_per_worker(picks, merged.bounds))
            t3 = time.perf_counter()

            live_rows = (np.sort(np.concatenate(list(live_rows_by_agent.values())))
                         if live_rows_by_agent else np.zeros(0, dtype=np.int64))
            owner_agent = np.empty(live_rows.shape[0], dtype=np.int64)
            for agent_idx, rows in live_rows_by_agent.items():
                owner_agent[np.searchsorted(live_rows, rows)] = agent_idx
            store = RoundStore(
                merged.states[live_rows].copy(),
                merged.cands[merged.cand_offsets[live_rows] + picks[live_rows]].copy(),
                owner_agent)

            game_ids = field_u16(merged.states, L.H_GAME_ID)
            hand_ids = field_u32(merged.states, L.H_HAND_ID)
            trick_idx = field_u8(merged.states, L.H_TRICK_INDEX)
            signs = decision_signs(merged.states)
            for local_i, i in enumerate(live_rows):
                key = (int(table_of[i]), int(game_ids[i]), int(hand_ids[i]))
                pending.setdefault(key, []).append(
                    (store, local_i, int(trick_idx[i]), float(signs[i])))
            decisions += merged.num_decisions

            for local_w, batch in enumerate(batches):
                w = group_workers[local_w].index
                for ep in parse_episodes(batch.episodes):
                    hands += 1
                    recent_points.append(ep.attacker_points)
                    entries = pending.pop((w, ep.game_id, ep.hand_id), None)
                    if not entries:
                        continue
                    returns, level = hand_returns(ep, cfg)
                    if returns.shape[0] == 0:
                        continue
                    # A hand's decisions belong to the two live agents at that
                    # table, so the episode has to be split before it is stored:
                    # each agent trains only on what it actually decided.
                    by_agent: dict[int, list] = {}
                    for e in entries:
                        by_agent.setdefault(
                            int(e[0].agents[e[1]]), []).append(e)
                    for agent_idx, es in by_agent.items():
                        st = np.stack([e[0].states[e[1]] for e in es])
                        ac = np.stack([e[0].actions[e[1]] for e in es])
                        tr = np.clip(
                            np.fromiter((e[2] for e in es), np.int64, len(es)),
                            0, returns.shape[0] - 1)
                        sg = np.fromiter((e[3] for e in es), np.float32, len(es))
                        agents[agent_idx].buffer.add(
                            st, ac, (returns[tr] * sg).astype(np.float32),
                            (level * sg).astype(np.float32))
                        agents[agent_idx].decisions += len(es)
            t4 = time.perf_counter()

            if rounds % cfg.train_every == 0:
                for _ in range(args.agents_per_pass):
                    agent = agents[train_cursor % args.agents]
                    train_cursor += 1
                    if agent.buffer.size < cfg.min_buffer:
                        continue
                    if (args.target_replay > 0
                            and agent.trained >= args.target_replay * agent.decisions):
                        throttled += 1
                        continue
                    for _ in range(cfg.steps_per_train):
                        stats = agent.learner.step(agent.buffer, rng)
                        if stats:
                            agent.stats = stats
                        agent.trained += cfg.batch_size
            t5 = time.perf_counter()

            t_env += t1 - t0; t_feat += t2 - t1; t_fwd += t3 - t2
            t_book += t4 - t3; t_learn += t5 - t4
            rounds += 1

            if rounds % args.iteration_rounds == 0:
                # Names already spoken for before this burst. `iteration`
                # is incremented *after* the burst, so a run checkpointed
                # inside one resumes onto the same boundary and re-derives
                # the same `a{idx}_{iteration}.pt` for entries the restored
                # archive still references. Rewriting one silently changes
                # which weights that entry means, and the declined-publish
                # unlink below then deletes it outright -- after which every
                # later boot whose seeded draw includes it dies on the load.
                # Distinct `idx` keeps the names apart within a burst, so
                # this is read once rather than per agent.
                already = {archive.payload(r) for r in archive.all_refs()}
                for agent in agents:
                    agent.publish()
                    path = arch_dir / f"a{agent.idx}_{iteration:05d}.pt"
                    if path.name in already or str(path) in already:
                        continue
                    # Written, and written atomically, before anything
                    # references it. A reference committed first -- by a
                    # signal landing inside this burst, or by an ENOSPC
                    # unwinding into the final save -- names a file that is
                    # torn or was never written at all, and every later boot
                    # whose seeded draw includes it dies on the load. The
                    # stale sweep below cannot clean that up either, since a
                    # registered path counts as live.
                    atomic_save(to_half(agent.acting.state_dict()), path)
                    if not archive.publish(agent.idx, iteration, path.name, rng):
                        # The reservoir declined it, so nothing will ever
                        # reference the file.
                        path.unlink(missing_ok=True)
                # Reservoir eviction leaves files nothing references.
                # Pre-relative-path checkpoints hold absolute payloads, so a
                # resumed archive can still be naming files either way.
                live_paths = {archive.payload(r) for r in archive.all_refs()}
                for stale in arch_dir.glob("a*.pt"):
                    if stale.name not in live_paths and str(stale) not in live_paths:
                        stale.unlink(missing_ok=True)
                iteration += 1
                seating = plan_seating(
                    rng, args.agents, args.workers, args.live_per_table,
                    choose_archive_refs(rng, archive, args.archive_budget))
                slots, slot_table = build_slots(
                    seating, agents, archive, scratch, device, arch_dir)
                print(f"[iter {iteration:4d}] archive {archive.size():3d} entries, "
                      f"{len(slots)} models | seats/agent "
                      f"{live_seat_counts(seating, args.agents).tolist()}",
                      flush=True)

            if rounds % args.log_every == 0:
                el = time.perf_counter() - started
                row = {
                    "elapsed_s": round(el, 1),
                    # Which invocation wrote this row. `elapsed_s` restarts
                    # every segment while `rounds`, `iteration` and `hands`
                    # carry over, so without this a reader has to infer the
                    # boundaries from a clock going backwards -- which is
                    # exactly what auditing the first chained run required,
                    # because `modal app logs` returns only the last task's
                    # output and shows a three-segment run as one.
                    "segment": segment,
                    "rounds": rounds_base + rounds,
                    "iteration": iteration,
                    "hands": hands_base + hands,
                    "decisions": decisions,
                    "decisions_per_s": round(decisions / el, 1),
                    "archive": archive.size(),
                    "attacker_points_mean": (
                        round(float(np.mean(recent_points)), 1)
                        if recent_points else None),
                    "pct_env": round(100 * t_env / el, 1),
                    "pct_features": round(100 * t_feat / el, 1),
                    "pct_forward": round(100 * t_fwd / el, 1),
                    "pct_bookkeeping": round(100 * t_book / el, 1),
                    "pct_learn": round(100 * t_learn / el, 1),
                    "steps": [a.learner.steps for a in agents],
                    "buffers": [int(a.buffer.size) for a in agents],
                    # Samples trained per fresh decision, per agent. The one
                    # number that says whether the population is data-starved.
                    "replay": [round(a.trained / max(1, a.decisions), 2)
                               for a in agents],
                    # Passes skipped because the agent was already ahead of
                    # the target. High is healthy -- it means the learner is
                    # no longer the thing holding the game loop back.
                    "throttled": throttled,
                    "loss_q": [round(float(a.stats.get("loss_q", float("nan"))), 5)
                               for a in agents],
                    "belief_acc": [round(float(a.stats.get("belief_acc", float("nan"))), 4)
                                   for a in agents],
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                mirror.log(row)
                print(f"[{el:7.1f}s] iter {iteration:4d} hands {hands:7d} "
                      f"dec/s {row['decisions_per_s']:7.0f} "
                      f"steps {min(row['steps'])}-{max(row['steps'])} "
                      f"replay {min(row['replay']):.1f}-{max(row['replay']):.1f} "
                      f"q {_mean(row['loss_q']):.5f} "
                      f"belief {_mean(row['belief_acc']):.3f}", flush=True)

            now = time.perf_counter()
            # A watermark, not `steps % every == 0`: the counter strides by
            # `steps_per_train` and is only sampled once a round, so a
            # modulo test skips any grid point it happens to jump over.
            # That exact bug cost this project two experiments.
            archive_at = None
            if args.checkpoint_every > 0 and reached_steps() >= next_archive_at:
                archive_at = next_archive_at
                next_archive_at = ((reached_steps() // args.checkpoint_every) + 1
                                   ) * args.checkpoint_every
            if (archive_at is not None
                    or now - last_checkpoint >= args.checkpoint_minutes * 60):
                save(archive_steps=archive_at)
                last_checkpoint = now
    finally:
        save(final=True)
        pool.close()
        el = time.perf_counter() - started
        print(f"done: {hands} hands, {decisions} decisions, {iteration} iterations "
              f"in {el/60:.1f} min; steps {[a.learner.steps for a in agents]}")
        log_file.close()
        mirror.summary(final_steps=[a.learner.steps for a in agents],
                       final_hands=hands_base + hands)
        mirror.finish()


if __name__ == "__main__":
    main()
