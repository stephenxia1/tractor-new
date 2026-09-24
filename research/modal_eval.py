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
Z95 = 1.959963984540054

# Where each pair's result is kept, keyed by everything that defines it.
# Written from inside the container that measured it, so a result
# survives the client that asked for it: twice a local client lost its
# connection and the app stopped with it, once after 2.5 hours with
# every pair still running.
RESULTS = f"{VOLUME}/evals"


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
    from tractor.nn.declare import DeclareNet
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
    host_pre = "agent" if pre_play.startswith("agent") else pre_play
    declare_net = None
    if pre_play in ("agent", "agent-heur"):
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob.get("declare") is not None:
            declare_net = DeclareNet()
            declare_net.load_state_dict(blob["declare"]["model"])
            declare_net = declare_net.to(device).eval()
    started = time.perf_counter()
    stats = play_against_bot(net, bot, device=device, cfg=cfg,
                             num_deals=deals, seed=seed, workers=workers,
                             per_hand=True, pre_play=host_pre,
                             search=SearchConfig(**search) if search else None,
                             declare_net=declare_net,
                             bury=BuryConfig(
                                 rounds=0 if pre_play == "agent-heur" else 8,
                                 consider=12))
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
@app.function(image=image, volumes={VOLUME: volume}, gpu="A10G", cpu=32.0,
              memory=32768, timeout=8 * 3600)
def play_gpu(checkpoint: str, bot: str, deals: int, seed: int,
             workers: int, pre_play: str, search: dict | None) -> dict:
    return _play(checkpoint, bot, deals, seed, workers, pre_play, search, "cuda")


def _label(path: str) -> str:
    """`/vol/v2-points/checkpoints/final.pt` -> `v2-points/final`."""
    parts = path.rstrip("/").split("/")
    return f"{parts[-3]}/{parts[-1].removesuffix('.pt')}" if len(parts) >= 3 else path


@app.local_entrypoint()
def smart(checkpoints: str, bots: str = ",".join(SMART), deals: int = 770,
          seed: int = 20260923, workers: int = 32, out: str = "",
          pre_play: str = "random", search_worlds: int = 0,
          search_top: int = 3, search_extra_tricks: int = 0,
          search_confident: float = 0.9) -> None:
    # `--search-worlds 0` is the plain policy, on CPU as before.
    search = None
    if search_worlds > 0:
        search = {"worlds": search_worlds, "top_k": search_top,
                  "extra_tricks": search_extra_tricks,
                  "confident": search_confident}
    # Search, and the burial climb of --pre-play agent, are batched
    # forwards that a CPU thread runs at ~100 ms a row.
    fn = play_gpu if (search or pre_play.startswith("agent")) else play
    ckpts = [c.strip() for c in checkpoints.split(",") if c.strip()]
    names = [b.strip() for b in bots.split(",") if b.strip()]
    pairs = [(c, b) for c in ckpts for b in names]
    path = out or f"eval-smart-{seed}.jsonl"

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
            print(f"  done {_label(r['checkpoint']):28} {r['bot']:10} "
                  f"win {r['winrate']:.3f}  {r['seconds'] / 60:5.1f} min  "
                  f"searched {int(r.get('searched_decisions', 0))}"
                  f"/{int(r['decisions'])}")
    if any(p not in results for p in pairs):
        print("some pairs are unmeasured; rerun the same command to fill them")
        return

    print()
    print(f"{'checkpoint':28} {'bot':10} {'win':>6} {'95% CI':>15} "
          f"{'levels':>7} {'points':>7}")
    for c in ckpts:
        pooled = []
        for b in names:
            r = results[(c, b)]
            pooled.extend(r["per_hand_wins"])
            print(f"{_label(c):28} {b:10} {r['winrate']:6.3f} "
                  f"[{r['winrate_ci_low']:.3f},{r['winrate_ci_high']:.3f}] "
                  f"{r['levels']:+7.3f} {r['points']:+7.1f}")
        m = sum(pooled) / len(pooled)
        half = Z95 * math.sqrt(m * (1 - m) / len(pooled))
        print(f"{_label(c):28} {'pooled':10} {m:6.3f} "
              f"[{m - half:.3f},{m + half:.3f}]")
        print()

    # Paired: the same hands under each checkpoint, so the difference
    # carries only the policy's change and not the deal's luck.
    if len(ckpts) > 1:
        print("paired win-rate change, each checkpoint minus the first")
        base = ckpts[0]
        for c in ckpts[1:]:
            for b in (*names, "pooled"):
                bs = names if b == "pooled" else [b]
                d = [y - x for k in bs
                     for x, y in zip(results[(base, k)]["per_hand_wins"],
                                     results[(c, k)]["per_hand_wins"])]
                n = len(d)
                mean = sum(d) / n
                var = sum((v - mean) ** 2 for v in d) / (n - 1)
                half = Z95 * math.sqrt(var / n)
                print(f"  {_label(c):28} {b:10} {mean:+.3f} "
                      f"[{mean - half:+.3f},{mean + half:+.3f}]")

    print(f"\nraw results: {path}")
