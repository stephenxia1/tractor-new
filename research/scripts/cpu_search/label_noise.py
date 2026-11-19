"""How noisy are search's labels at 8 x 3 and at 12 x 4?

Positions where search acts, from zero2-dst2 step 5000's own play
(sampled moves, as `train/distill.py::generate` plays): each is searched
twice at each width with independent world samples. Self-agreement is
the chance two labels for one position agree (sum of p^2 over search's
choice distribution): the label noise distillation trains on.

  python label_noise.py HANDS_PER_WORKER WORKERS OUT.json
"""
import json, sys, time
import multiprocessing as mp
import numpy as np


SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 1000


def work(args):
    wid, hands = args
    import torch
    torch.set_num_threads(1)
    from load_zero import load_zero
    from tractor.eval.search import CandidatePolicy, SearchConfig, search_move
    from tractor.gen.episode import generate
    from tractor.nn.features import Cache
    from tractor.train.buffer import obs_batch_from_arrays
    from tractor.train.rollout import _InFlight, _pack

    net = load_zero(); cfg = net.cfg; cache = Cache()
    policy = CandidatePolicy(net, cache, cfg, "cpu")
    widths = {"8x3": SearchConfig(worlds=8, top_k=3, confident=0.99),
              "12x4": SearchConfig(worlds=12, top_k=4, confident=0.99)}
    rng = np.random.default_rng(SEED + wid)
    rows = []
    for _ in range(hands):
        state, _ = generate(rng, 2)
        while not state.done:
            moves = state.legal_moves(cap=cfg.max_candidates)
            packed = _pack(_InFlight(state=state, episode=0), cache, cfg, moves=moves)
            n = int(packed.candidate_mask.sum())
            obs = obs_batch_from_arrays(
                packed.history[None], np.array([packed.history_len]), packed.context[None],
                packed.unseen[None], packed.planes[None], packed.cards[None],
                packed.candidates[None], packed.candidate_mask[None], device="cpu")
            with torch.no_grad():
                logp = torch.log_softmax(policy.fast.logits(obs).float(), -1)[0, :n].numpy()
            top = int(np.argmax(logp))
            if n > 1 and float(np.exp(logp[top])) < 0.99:
                row = {"top": top, "p_top": float(np.exp(logp[top])), "n": n}
                for name, sc in widths.items():
                    picks = []
                    for rep in range(2):
                        r = np.random.default_rng((SEED, wid, len(rows), rep, sc.worlds))
                        i, did = search_move(net, state, list(moves.moves[:n]), logp, sc, r,
                                             cache, cfg, "cpu", policy=policy)
                        picks.append(int(i) if did else -1)
                    row[name] = picks
                rows.append(row)
            p = np.exp(logp - logp.max())
            state.step(moves.moves[int(rng.choice(n, p=p / p.sum()))])
    return rows


def ci(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


if __name__ == "__main__":
    hands, workers, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(work, [(w, hands) for w in range(workers)]) for r in part]
    rows = [r for r in rows if min(r["8x3"] + r["12x4"]) >= 0]
    a8 = [r["8x3"][0] == r["8x3"][1] for r in rows]
    a12 = [r["12x4"][0] == r["12x4"][1] for r in rows]
    cross = [r["8x3"][0] == r["12x4"][0] for r in rows]
    ch8 = [r["8x3"][0] != r["top"] for r in rows]
    ch12 = [r["12x4"][0] != r["top"] for r in rows]
    summary = {
        "positions": len(rows), "hands": hands * workers, "seconds": time.perf_counter() - t,
        "self_agree_8x3": ci(a8), "self_agree_12x4": ci(a12),
        "self_agree_12x4_minus_8x3": ci(np.array(a12, float) - np.array(a8, float)),
        "agree_8x3_12x4": ci(cross),
        "overrules_policy_8x3": ci(ch8), "overrules_policy_12x4": ci(ch12),
        "mean_p_top": float(np.mean([r["p_top"] for r in rows])),
    }
    json.dump({"summary": summary, "rows": rows}, open(out, "w"))
    print(json.dumps(summary, indent=1))
