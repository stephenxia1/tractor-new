"""Checkpoints against the smart bots, at a width that can settle a rung.

The ladder the training loop measures is 16 to 48 deals a rung, which
leaves a 95% interval about 0.2 wide on a win rate -- too wide to say
whether one rung is above even, or one checkpoint above another. This
measures each (checkpoint, bot) pair on its own CPU container, at
770 deals (1,540 hands) by default: +/-0.025 at a win rate near 0.5.

It is a separate app from `modal_train.py` on purpose. Two `modal run`s
of the same app stop each other, so an evaluation under `tractor-train`
could not run while a training run is live.

Every pair plays the same deals (one seed base), so two checkpoints are
compared hand for hand: the cards cancel, and the paired difference is
tighter than either interval alone.

    modal run modal_eval.py::smart \\
      --checkpoints /vol/v2-points/checkpoints/step_0000500.pt,/vol/v2-points/checkpoints/final.pt

A population's agents, every one against the whole ladder, ranked --
run on Modal end to end, the record and ranking on the volume:

    modal run --detach modal_eval.py::population --checkpoint /vol/pop8/checkpoints/final.pt
"""

from __future__ import annotations

import json
import math
import os

import modal

import modal_train

# Overridable because two `modal run`s of one app stop each other: a
# second measurement started while one is live needs a name of its own.
app = modal.App(os.environ.get("TRACTOR_EVAL_APP", "tractor-eval"))
image = modal_train.image.add_local_python_source("modal_train")
volume = modal_train.volume
VOLUME = modal_train.VOLUME

# The smart bots that play from their own seat's view. smart-v2, -v4 and
# -v9 search on the true deal and are left out; see `LADDER` in
# tractor/eval/bots.py.
SMART = ("smart", "smart-v18", "smart-v28", "smart-v34")
# GPU containers a measurement may hold at once: the workspace's limit of
# 10. A training run's GPU counts against the same limit, so set
# TRACTOR_EVAL_GPUS=9 beside one (or the eval queues behind it).
GPUS = int(os.environ.get("TRACTOR_EVAL_GPUS", "10"))
Z95 = 1.959963984540054

# Where each pair's result is kept, keyed by everything that defines it.
# Written from inside the container that measured it, so a result
# survives the client that asked for it: twice a local client lost its
# connection and the app stopped with it, once after 2.5 hours with
# every pair still running.
RESULTS = f"{VOLUME}/evals"
# A whole measurement's record and ranking, when it runs on Modal
# (`population_job`).
RUNS = f"{RESULTS}/runs"


def _key(checkpoint: str, bot: str, deals: int, seed: int, pre_play: str,
         search: dict | None) -> str:
    import hashlib

    spec = json.dumps([checkpoint, bot, deals, seed, pre_play, search], sort_keys=True)
    return hashlib.sha1(spec.encode()).hexdigest()[:16]


def _play(checkpoint: str, bot: str, deals: int, seed: int, workers: int,
          pre_play: str, search: dict | None, device: str) -> dict:
    import time

    import torch

    from tractor.eval.bots import play_against_bot
    from tractor.eval.search import SearchConfig
    from tractor.sim.bury import BuryConfig

    # Each host shard scores its own decisions on its own thread; intra-op
    # threads on top of that would only fight over the same cores.
    torch.set_num_threads(1)
    net, cfg = modal_train._load_net(checkpoint, device)
    net.eval()
    # --pre-play agent: the checkpoint's own declaration network (none
    # means its seats never declare) and the value climb for burial.
    # agent-pass is the same with the declaration network withheld, so
    # the network's seats never declare: what declaring is worth.
    # agent-heur keeps the declaration network and buries by the plain
    # heuristic the climb starts from: what the climb is worth.
    # agent-critic climbs on the privileged critic averaged over 8 deals
    # of the landlord's unseen cards instead of the actor's value head.
    # agent-policy buries by the checkpoint's learned burial policy.
    host_pre = "agent" if pre_play.startswith("agent") else pre_play
    declare_net = None
    bury_net = None
    if pre_play in ("agent", "agent-heur", "agent-critic", "agent-policy"):
        from tractor.nn.arch import bury_net_from, declare_net_from
        from tractor.train.population import read_checkpoint

        # `pop/final.pt#k`: agent k of a population, in the usual layout.
        # Each network is rebuilt from the config saved beside it (v5's
        # when the checkpoint predates saved configs).
        blob = read_checkpoint(checkpoint)
        if blob.get("declare") is not None:
            declare_net = declare_net_from(blob["declare"]).to(device).eval()
        if pre_play == "agent-policy":
            if blob.get("bury") is None:
                raise ValueError(f"{checkpoint} carries no burial policy")
            bury_net = bury_net_from(blob["bury"]).to(device).eval()
    started = time.perf_counter()
    stats = play_against_bot(net, bot, device=device, cfg=cfg,
                             num_deals=deals, seed=seed, workers=workers,
                             per_hand=True, pre_play=host_pre,
                             search=SearchConfig(**search) if search else None,
                             declare_net=declare_net,
                             bury_net=bury_net,
                             bury=(
                                 BuryConfig(evaluator="critic", worlds=8, rounds=8)
                                 if pre_play == "agent-critic" else
                                 BuryConfig(
                                     rounds=0 if pre_play == "agent-heur" else 8,
                                     consider=12)))
    stats["seconds"] = time.perf_counter() - started
    stats["checkpoint"] = checkpoint
    stats["bot"] = bot
    stats["pre_play"] = pre_play
    stats["search"] = search
    stats["declare_net"] = declare_net is not None
    stats["seed"] = seed
    os.makedirs(RESULTS, exist_ok=True)
    key = _key(checkpoint, bot, deals, seed, pre_play, search)
    with open(f"{RESULTS}/{key}.json", "w") as fh:
        json.dump(stats, fh)
    volume.commit()
    return stats


@app.function(image=image, volumes={VOLUME: volume}, timeout=600)
def fetch(keys: list) -> list:
    """The stored result for each key, or None."""
    volume.reload()
    out = []
    for key in keys:
        path = f"{RESULTS}/{key}.json"
        if os.path.exists(path):
            with open(path) as fh:
                out.append(json.load(fh))
        else:
            out.append(None)
    return out


@app.function(image=image, volumes={VOLUME: volume}, cpu=32.0,
              memory=32768, timeout=4 * 3600)
def play(checkpoint: str, bot: str, deals: int, seed: int,
         workers: int, pre_play: str, search: dict | None) -> dict:
    return _play(checkpoint, bot, deals, seed, workers, pre_play, search, "cpu")


# Search needs the GPU. A searched decision is tens of forward rows, and
# on one CPU thread a row costs about 100 ms -- measured, batch 1 to 64
# -- so the search that takes seconds there is milliseconds here. The
# shards share the device from their own threads; the work is small
# batches, bound by kernel launches, which an A10G serves as well as a
# larger card.
#
# At most 8 at once. The workspace may run 10 GPUs, and a launch that
# asked for 12 of them beside a training run queued the rest; a cap here
# keeps one launch from taking the device a training run needs. Pairs
# past the cap wait their turn rather than fail.
@app.function(image=image, volumes={VOLUME: volume}, gpu="A10G", cpu=32.0,
              memory=32768, timeout=8 * 3600, max_containers=GPUS)
def play_gpu(checkpoint: str, bot: str, deals: int, seed: int,
             workers: int, pre_play: str, search: dict | None) -> dict:
    return _play(checkpoint, bot, deals, seed, workers, pre_play, search, "cuda")


def _label(path: str) -> str:
    """`/vol/v2-points/checkpoints/final.pt` -> `v2-points/final`, and a
    population's `/vol/pop8/checkpoints/final.pt#3` -> `pop8/final#3`."""
    parts = path.rstrip("/").split("/")
    return (f"{parts[-3]}/{parts[-1].replace('.pt', '')}"
            if len(parts) >= 3 else path)


def _measure(ckpts: list, names: list, deals: int, seed: int, workers: int,
             path: str, pre_play: str, search: dict | None,
             gpu: bool = False, on_write=None) -> dict | None:
    """Every (checkpoint, bot) pair, measured or read back. -> the results,
    or None while some pair is still unmeasured. ``on_write`` runs after
    each write to ``path`` (a volume commit, on Modal)."""
    on_write = on_write or (lambda: None)
    # Search, and the burial climb of --pre-play agent, are batched
    # forwards that a CPU thread runs at ~100 ms a row. `gpu` sends the
    # rest there too: 32-core CPU containers are sometimes not to be had
    # (Modal queues them "waiting to be scheduled on a CPU worker").
    fn = play_gpu if (gpu or search or pre_play.startswith("agent")) else play
    pairs = [(c, b) for c in ckpts for b in names]

    # Each pair is appended the moment it lands, and a rerun with the
    # same `out` skips what is already there. A pair takes about twenty
    # minutes; one preempted container once took the client down with
    # it and lost the other twenty-seven.
    results = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if (r["deals"] == deals
                        and r.get("pre_play", "bot") == pre_play
                        and r.get("search") == search):
                    results[(r["checkpoint"], r["bot"])] = r
    # Results a previous client asked for and never received are on the
    # volume; take them before measuring anything again.
    missing = [p for p in pairs if p not in results]
    stored = fetch.remote([_key(c, b, deals, seed, pre_play, search) for c, b in missing])
    with open(path, "a") as fh:
        for (c, b), r in zip(missing, stored):
            if r is not None:
                results[(c, b)] = r
                fh.write(json.dumps(r) + "\n")
    on_write()
    todo = [p for p in pairs if p not in results]
    print(f"{len(pairs)} pairs, {len(pairs) - len(todo)} already measured, "
          f"{deals} deals each, seed {seed}, pre-play {pre_play}, "
          f"search {search}")

    with open(path, "a") as fh:
        for r in fn.starmap([(c, b, deals, seed, workers, pre_play, search)
                                 for c, b in todo],
                              order_outputs=False, return_exceptions=True):
            if isinstance(r, BaseException):
                print(f"  a pair failed: {type(r).__name__}: {r}")
                continue
            results[(r["checkpoint"], r["bot"])] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            on_write()
            print(f"  done {_label(r['checkpoint']):28} {r['bot']:10} "
                  f"win {r['winrate']:.3f}  levels {r['levels']:+.3f}  "
                  f"points {r['points']:+.1f}  {r['seconds'] / 60:5.1f} min")
    if any(p not in results for p in pairs):
        print("some pairs are unmeasured; rerun the same command to fill them")
        return None
    return results


def _pooled(results: dict, c: str, names: list) -> dict:
    """One checkpoint over the whole ladder: the win rate over every hand
    against every bot, and levels and points averaged over the bots
    (each played on the same number of deals), with 95% intervals."""
    rs = [results[(c, b)] for b in names]
    wins = [w for r in rs for w in r["per_hand_wins"]]
    m = sum(wins) / len(wins)
    half = Z95 * math.sqrt(m * (1 - m) / len(wins))
    k = len(rs)
    lv = sum(r["levels"] for r in rs) / k
    lv_half = Z95 * math.sqrt(sum(r["levels_stderr"] ** 2 for r in rs)) / k
    pt = sum(r["points"] for r in rs) / k
    pt_half = Z95 * math.sqrt(sum(r["points_stderr"] ** 2 for r in rs)) / k
    return {"checkpoint": c, "win": m, "win_ci": [m - half, m + half],
            "levels": lv, "levels_half": lv_half, "points": pt, "points_half": pt_half,
            "hands": len(wins), "bots": list(names)}


def _report(ckpts: list, names: list, results: dict, paired: bool) -> list:
    """Each checkpoint against each bot and over the whole ladder, then the
    checkpoints ranked. -> the pooled rows, best first."""
    print()
    print(f"{'checkpoint':28} {'bot':10} {'win':>6} {'95% CI':>15} "
          f"{'levels':>8} {'points':>8}")
    pooled = []
    for c in ckpts:
        for b in names:
            r = results[(c, b)]
            print(f"{_label(c):28} {b:10} {r['winrate']:6.3f} "
                  f"[{r['winrate_ci_low']:.3f},{r['winrate_ci_high']:.3f}] "
                  f"{r['levels']:+8.3f} {r['points']:+8.1f}")
        row = _pooled(results, c, names)
        pooled.append(row)
        print(f"{_label(c):28} {'ladder':10} {row['win']:6.3f} "
              f"[{row['win_ci'][0]:.3f},{row['win_ci'][1]:.3f}] "
              f"{row['levels']:+8.3f} {row['points']:+8.1f}   "
              f"(levels +/-{row['levels_half']:.3f}, points +/-{row['points_half']:.1f})")
        print()

    ranked = sorted(pooled, key=lambda r: (-r["win"], -r["points"]))
    if len(ckpts) > 1:
        print(f"ranked over the ladder ({', '.join(names)}):")
        print(f"  {'rank':>4} {'checkpoint':28} {'win':>6} {'95% CI':>15} "
              f"{'levels':>8} {'points':>8}")
        for i, r in enumerate(ranked, 1):
            print(f"  {i:>4} {_label(r['checkpoint']):28} {r['win']:6.3f} "
                  f"[{r['win_ci'][0]:.3f},{r['win_ci'][1]:.3f}] "
                  f"{r['levels']:+8.3f} {r['points']:+8.1f}")
        print()

    # Paired: the same hands under each checkpoint, so the difference
    # carries only the policy's change and not the deal's luck.
    if paired and len(ckpts) > 1:
        print("paired win-rate change, each checkpoint minus the first")
        base = ckpts[0]
        for c in ckpts[1:]:
            for b in (*names, "ladder"):
                bs = names if b == "ladder" else [b]
                d = [y - x for k in bs
                     for x, y in zip(results[(base, k)]["per_hand_wins"],
                                     results[(c, k)]["per_hand_wins"])]
                n = len(d)
                mean = sum(d) / n
                var = sum((v - mean) ** 2 for v in d) / (n - 1)
                half = Z95 * math.sqrt(var / n)
                print(f"  {_label(c):28} {b:10} {mean:+.3f} "
                      f"[{mean - half:+.3f},{mean + half:+.3f}]")
    return ranked


@app.local_entrypoint()
def smart(checkpoints: str, bots: str = ",".join(SMART), deals: int = 770,
          seed: int = 20260923, workers: int = 32, out: str = "",
          pre_play: str = "random", search_worlds: int = 0,
          search_top: int = 3, search_extra_tricks: int = 0,
          search_confident: float = 0.9, gpu: bool = False) -> None:
    # `--search-worlds 0` is the plain policy, on CPU as before.
    search = None
    if search_worlds > 0:
        search = {"worlds": search_worlds, "top_k": search_top,
                  "extra_tricks": search_extra_tricks,
                  "confident": search_confident}
    ckpts = [c.strip() for c in checkpoints.split(",") if c.strip()]
    names = [b.strip() for b in bots.split(",") if b.strip()]
    path = out or f"eval-smart-{seed}.jsonl"
    results = _measure(ckpts, names, deals, seed, workers, path, pre_play, search,
                       gpu)
    if results is None:
        return
    _report(ckpts, names, results, paired=True)
    print(f"\nraw results: {path}")


def _population(checkpoint: str, agents: int, names: list, deals: int, seed: int,
                workers: int, path: str, pre_play: str, gpu: bool,
                on_write=None) -> list | None:
    """Measure and rank a population's agents; the record at ``path``, the
    ranking at ``<path>.summary.json``. -> the ranking, or None while some
    pair is unmeasured."""
    ckpts = [f"{checkpoint}#{k}" for k in range(agents)]
    results = _measure(ckpts, names, deals, seed, workers, path, pre_play, None,
                       gpu, on_write)
    if results is None:
        return None
    ranked = _report(ckpts, names, results, paired=False)
    summary = {"checkpoint": checkpoint, "deals": deals, "seed": seed,
               "pre_play": pre_play, "bots": names, "ranked": ranked}
    with open(path + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    (on_write or (lambda: None))()
    return ranked


@app.function(image=image, volumes={VOLUME: volume}, timeout=86400, cpu=1.0,
              memory=2048)
def population_job(checkpoint: str, agents: int, names: list, deals: int, seed: int,
                   workers: int, out: str, pre_play: str, gpu: bool) -> list:
    """``population`` run on Modal, start to finish: the pairs fan out
    from here, and the record and ranking go to ``/vol/evals/runs/<out>``.
    Nothing local has to stay up -- three times a local client's sleep or
    a memory kill cancelled the pairs in flight. A pair that fails is
    tried again, twice."""
    volume.reload()
    os.makedirs(RUNS, exist_ok=True)
    path = f"{RUNS}/{out}"
    for attempt in range(3):
        ranked = _population(checkpoint, agents, names, deals, seed, workers, path,
                             pre_play, gpu, on_write=volume.commit)
        if ranked is not None:
            return ranked
        print(f"attempt {attempt + 1}: some pairs unmeasured; measuring them again")
    raise RuntimeError(f"pairs still unmeasured after three attempts; see {path}")


@app.local_entrypoint()
def population(checkpoint: str, agents: int = 8, bots: str = ",".join(SMART),
               deals: int = 770, seed: int = 20260923, workers: int = 32,
               out: str = "", pre_play: str = "agent-policy",
               gpu: bool = False, remote: bool = True) -> None:
    """Every agent of a population checkpoint against the whole ladder.

        modal run --detach modal_eval.py::population \\
          --checkpoint /vol/pop8/checkpoints/final.pt --out pop8-final.jsonl
        # when it is done (modal app logs shows the ranking):
        modal volume get tractor-rl /evals/runs/pop8-final.jsonl* evals/...

    Agents ``#0 .. #agents-1``, each against every fair bot (the four
    smart bots that play from their own seat) at ``deals`` deals, the
    full hand as trained (``--pre-play agent-policy``: its own
    declarations and learned burial). Each agent's win rate, levels and
    points against each bot and over the ladder, and the agents ranked;
    the ranking also lands in ``<out>.summary.json``. At the defaults,
    32 pairs on up to ten A10Gs (`TRACTOR_EVAL_GPUS`): about an hour.

    By default the whole measurement runs on Modal (``population_job``)
    and this returns at once; ``--no-remote`` runs the old way, the
    client fanning the pairs out and writing ``out`` locally.
    """
    names = [b.strip() for b in bots.split(",") if b.strip()]
    name = out or f"eval-{_label(checkpoint).replace('/', '-')}-{seed}.jsonl"
    if remote:
        name = os.path.basename(name)
        call = population_job.spawn(checkpoint, agents, names, deals, seed, workers,
                                    name, pre_play, gpu)
        print(f"spawned {call.object_id}: {RUNS}/{name} (+ .summary.json)\n"
              f"  fetch: modal volume get tractor-rl /evals/runs/{name} <dir>\n"
              f"         modal volume get tractor-rl /evals/runs/{name}.summary.json <dir>")
        return
    if _population(checkpoint, agents, names, deals, seed, workers, name, pre_play,
                   gpu) is not None:
        print(f"raw results: {name}\nranking: {name}.summary.json")
