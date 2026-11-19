"""Milliseconds per searched decision on one CPU thread, by forward.

zero2-dst2 step 5000 (the app's float16 export) over N positions where
search acts in its own play; each variant searches the same positions
with the same seeds and reports its agreement with the reference.

  python bench_search.py N WORLDS TOP_K ref,fast,bf16,int8
"""
import sys, time, copy, pickle
import numpy as np, torch
from load_zero import load_zero
from tractor.eval.search import SearchConfig, search_move, CandidatePolicy
from tractor.nn.features import Cache
from tractor.gen.episode import generate
from tractor.train.buffer import obs_batch_from_arrays
from tractor.train.rollout import _InFlight, _evaluate, _pack

torch.set_num_threads(1)
net = load_zero(); cfg = net.cfg; cache = Cache()
N = int(sys.argv[1]); W, K = int(sys.argv[2]), int(sys.argv[3]); variants = sys.argv[4].split(",")
rng = np.random.default_rng(11); positions = []
while len(positions) < N:
    state, _ = generate(rng, 2)
    while not state.done and len(positions) < N:
        moves = state.legal_moves(cap=cfg.max_candidates)
        packed = _pack(_InFlight(state=state, episode=0), cache, cfg, moves=moves)
        n = int(packed.candidate_mask.sum())
        obs = obs_batch_from_arrays(packed.history[None], np.array([packed.history_len]), packed.context[None],
            packed.unseen[None], packed.planes[None], packed.cards[None], packed.candidates[None], packed.candidate_mask[None], device="cpu")
        priv = torch.from_numpy(packed.privileged[None].astype(np.float32))
        with torch.no_grad(): out, _ = _evaluate(net, obs, priv)
        logp = out.log_probs[0, :n].float().numpy()
        if n > 1 and np.exp(logp.max()) < 0.99:
            positions.append((state.clone(), list(moves.moves[:n]), logp.copy()))
        p = np.exp(logp - logp.max()); state.step(moves.moves[int(rng.choice(n, p=p / p.sum()))])
qnet = None
def policy(v):
    global qnet
    if v == "ref": return CandidatePolicy(net, cache, cfg, "cpu", fast=False)
    if v == "fast": return CandidatePolicy(net, cache, cfg, "cpu")
    if v == "bf16": return CandidatePolicy(net, cache, cfg, "cpu", bf16=True)
    if v == "int8":
        if qnet is None:
            qnet = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net), {torch.nn.Linear}, dtype=torch.qint8)
        return CandidatePolicy(qnet, cache, cfg, "cpu")
sc = SearchConfig(worlds=W, top_k=K, extra_tricks=0, confident=0.99)
choices = {}
for v in variants:
    pol = policy(v)
    search_move(net, *positions[0], sc, np.random.default_rng(0), cache, cfg, "cpu", policy=pol)  # warm-up
    t = time.perf_counter(); ch = []
    for i, (st, mv, lp) in enumerate(positions):
        ch.append(search_move(net, st, mv, lp, sc, np.random.default_rng(i), cache, cfg, "cpu", policy=pol)[0])
    dt = (time.perf_counter() - t) / len(positions); choices[v] = np.array(ch)
    agree = "" if "ref" not in choices else f"  same choice as ref {np.mean(choices[v] == choices['ref']):.3f}"
    print(f"{W}x{K} {v:5s}: {dt*1000:6.0f} ms per searched decision{agree}", flush=True)
