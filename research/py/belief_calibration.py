"""Is the belief head calibrated enough to steer the determinized sampler?

    python py/belief_calibration.py --checkpoint checkpoints/v2_latest.pt

`belief_acc` in the training log is thresholded accuracy on a heavily
imbalanced target -- roughly 25 cards spread over 66 chain positions per
opponent, so a head that mostly answers "no" scores well without knowing
anything. Weighting a sampler needs *probabilities* that mean what they
say, and a confidently wrong belief is worse than a uniform one: the
search will spend its worlds on positions that never occur. Search has
already been worse than no search twice in this project for closely
related reasons, so this measurement is the gate on the whole idea.

Reported, on positions the agent actually reaches while playing itself:

* **log loss** against the base-rate baseline -- does the head beat "always
  predict the prior"?
* **ECE** and a reliability table -- when it says 0.7, is it right 70% of
  the time?
* the same, restricted to *uncertain* positions, since the aggregate is
  dominated by the easy mass (cards already played, or that I hold, are
  trivially absent from an opponent's hand)
* the temperature that minimises log loss, which is the cheap fix if the
  head is sharp but miscalibrated
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tractor_rl.features2 import (
    belief_targets,
    history_features,
    merge_batches,
    parse_frame,
    state_features,
)
from tractor_rl.model2 import NetConfig, TractorNet
from tractor_rl.protocol import WorkerConfig, WorkerPool

RESEARCH_DIR = Path(__file__).resolve().parents[1]


def collect(net, device, workers: int, games: int, rounds: int, seed: int):
    """Self-play with the trained net, keeping states and belief targets."""
    cfg = WorkerConfig(games=games, seed=seed, learner_seats=15, privileged=True)
    pool = WorkerPool(workers, cfg, num_groups=1)
    states = []
    try:
        for _ in range(rounds):
            batches = pool.read_round()
            merged = merge_batches(batches)
            if merged.num_decisions == 0:
                pool.write_round(batches, [np.zeros(0, dtype=np.int64) for _ in batches])
                continue
            sf = torch.from_numpy(state_features(merged.states)).to(device)
            tok, msk = history_features(merged.states)
            with torch.no_grad():
                out = net(
                    sf,
                    torch.from_numpy(
                        __import__("tractor_rl.features2", fromlist=["action_features"])
                        .action_features(merged.cands, merged.states, merged.owner)
                    ).to(device),
                    torch.from_numpy(merged.owner).to(device),
                    torch.from_numpy(merged.states[:, 2].astype(np.int64)).to(device),
                    torch.from_numpy(tok).to(device),
                    torch.from_numpy(msk).to(device),
                )
            q = out["q"].cpu().numpy()
            picks = []
            at = 0
            for count in merged.cand_counts:
                picks.append(int(np.argmax(q[at: at + count])) if count else 0)
                at += count
            states.append(merged.states)
            # Hand the choices back per worker.
            per_worker, off = [], 0
            for b in batches:
                n = b.num_decisions
                per_worker.append(np.array(picks[off: off + n], dtype=np.int64))
                off += n
            pool.write_round(batches, per_worker)
    finally:
        pool.close()
    return np.concatenate(states, axis=0) if states else np.zeros((0, 0), dtype=np.uint8)


def ece(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> tuple[float, list]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    total, err, rows = probs.size, 0.0, []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        conf = float(probs[sel].mean())
        acc = float(targets[sel].mean())
        err += (n / total) * abs(conf - acc)
        rows.append((edges[b], edges[b + 1], n, conf, acc))
    return err, rows


def report(name: str, probs: np.ndarray, targets: np.ndarray) -> None:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    ll = float(-(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean())
    base = float(targets.mean())
    bp = np.clip(base, eps, 1 - eps)
    ll_base = float(-(targets * np.log(bp) + (1 - targets) * np.log(1 - bp)).mean())
    err, rows = ece(p, targets)
    acc = float(((p > 0.5) == (targets > 0.5)).mean())

    print(f"\n--- {name} ({targets.size:,} predictions) ---")
    print(f"  base rate         {base:.4f}")
    print(f"  thresholded acc   {acc:.4f}")
    print(f"  log loss          {ll:.4f}   (base-rate baseline {ll_base:.4f})")
    print(f"  improvement       {(1 - ll / ll_base) * 100:5.1f}% vs predicting the base rate")
    print(f"  ECE               {err:.4f}")
    print(f"  {'bin':>12}  {'n':>9}  {'said':>6}  {'was':>6}")
    for lo, hi, n, conf, obs in rows:
        print(f"  {lo:.2f}-{hi:.2f}  {n:9,}  {conf:6.3f}  {obs:6.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v2_latest.pt")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = TractorNet(NetConfig(**ck["net_config"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"checkpoint step {ck.get('steps', 0):,}")

    states = collect(net, device, args.workers, args.games, args.rounds, args.seed)
    print(f"collected {states.shape[0]:,} privileged decisions")
    if states.shape[0] == 0:
        raise SystemExit("no decisions collected")

    sf = torch.from_numpy(state_features(states)).to(device)
    tok, msk = history_features(states)
    with torch.no_grad():
        z = net.encode_state(sf, torch.from_numpy(tok).to(device), torch.from_numpy(msk).to(device))
        logits = net.belief_head(z)
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    targets = belief_targets(states).reshape(-1)

    report("all positions", probs, targets)

    # The aggregate is dominated by positions that are trivially absent --
    # cards I hold, or that have already been played. Those need no model.
    # What matters for sampling is the genuinely uncertain mass.
    uncertain = (probs > 0.02) & (probs < 0.98)
    if uncertain.sum() > 0:
        report("uncertain positions only", probs[uncertain], targets[uncertain])
        print(f"\n  uncertain mass: {uncertain.mean() * 100:.1f}% of predictions")

    # Temperature scaling: the cheap fix when a head is sharp but overconfident.
    lg = torch.from_numpy(logits.cpu().numpy().reshape(-1))
    tg = torch.from_numpy(targets)
    best_t, best_ll = 1.0, None
    for t in np.arange(0.5, 3.01, 0.05):
        p = torch.sigmoid(lg / float(t))
        ll = float(torch.nn.functional.binary_cross_entropy(p, tg))
        if best_ll is None or ll < best_ll:
            best_ll, best_t = ll, float(t)
    p1 = torch.sigmoid(lg)
    ll1 = float(torch.nn.functional.binary_cross_entropy(p1, tg))
    print(f"\n  best temperature  T={best_t:.2f}  log loss {best_ll:.4f} "
          f"(T=1 gives {ll1:.4f}, {(1 - best_ll / ll1) * 100:.1f}% better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
