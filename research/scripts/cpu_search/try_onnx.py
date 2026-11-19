"""ONNX Runtime against torch for search's forward, on one CPU thread.

zero2-dst2 step 5000 (the app's float16 export, as float32). Search-shaped
batches from real play: one decision's 48 worlds (12 x 4's ply batch).

  A  torch, the training forward (history padded to 128, all 54+7 query tokens)
  B  ONNX Runtime, the same graph exported (dynamic batch)
  C  ONNX Runtime, the same graph with the history trimmed to the batch's
     longest (a dynamic history axis), as search's own forward trims it
  D  torch, search's own forward (eval/fastnet.py), for reference

Timing per batch: actor logits + critic value, as a ply then a leaf need.
"""
import sys, time
import numpy as np, torch
import onnxruntime as ort
from load_zero import load_zero
from tractor.eval.fastnet import SearchNet
from tractor.eval.search import sample_world, _in_world
from tractor.nn.features import Cache
from tractor.nn.model import ObsBatch
from tractor.gen.episode import generate
from tractor.train.buffer import obs_batch_from_arrays
from tractor.train.rollout import _InFlight, _evaluate, _pack

torch.set_num_threads(1)
import os, tempfile
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="onnx-")
os.makedirs(OUT, exist_ok=True)
net = load_zero(); cfg = net.cfg; fast = SearchNet(net); cache = Cache()
rng = np.random.default_rng(3)
states = []
for _ in range(4):
    st, _ = generate(rng, 2)
    while not st.done:
        states.append(st.clone())
        mv = st.legal_moves(cap=cfg.max_candidates).moves
        st.step(mv[int(rng.integers(len(mv)))])
batches = []
for st in states[::9]:
    batches.append([_in_world(st, sample_world(st, int(st.to_act), rng)) for _ in range(48)])


def obs_of(sims):
    packed = [_pack(_InFlight(state=s, episode=0), cache, cfg) for s in sims]
    obs = obs_batch_from_arrays(
        np.stack([p.history for p in packed]), np.array([p.history_len for p in packed]),
        np.stack([p.context for p in packed]), np.stack([p.unseen for p in packed]),
        np.stack([p.planes for p in packed]), np.stack([p.cards for p in packed]),
        np.stack([p.candidates for p in packed]), np.stack([p.candidate_mask for p in packed]),
        device="cpu")
    priv = torch.from_numpy(np.stack([p.privileged for p in packed]).astype(np.float32))
    return obs, priv


data = [obs_of(b) for b in batches]
print(f"{len(data)} batches of 48; mean longest history "
      f"{np.mean([int(o.history_len.max()) for o, _ in data]):.0f} of {cfg.max_history}")


class Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, history, history_len, context, unseen, planes, cards, candidates,
                candidate_mask, privileged):
        obs = ObsBatch(history, history_len, context, unseen, planes, cards, candidates,
                       candidate_mask)
        out, value = self.net(obs, privileged)
        return out.logits, value


names = ["history", "history_len", "context", "unseen", "planes", "cards", "candidates",
         "candidate_mask", "privileged"]


def arrays(obs, priv, trim=False):
    xs = list(obs) + [priv]
    if trim:
        t = int(obs.history_len.max())
        xs[0] = xs[0][:, :t].contiguous()
    return xs


def export(path, trim):
    obs, priv = data[0]
    xs = arrays(obs, priv, trim)
    B = torch.export.Dim("B", min=2, max=256)
    dyn = {n: {0: B} for n in names}
    if trim:
        dyn["history"] = {0: B, 1: torch.export.Dim("T", min=2, max=cfg.max_history)}
    t = time.perf_counter()
    torch.onnx.export(Wrap(net).eval(), tuple(xs), path, input_names=names,
                      output_names=["logits", "value"], dynamic_shapes=tuple(dyn[n] for n in names),
                      dynamo=True, optimize=True)
    print(f"exported {path.split('/')[-1]} in {time.perf_counter() - t:.0f} s")


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def time_it(fn, reps=3):
    fn(data[0])  # warm
    t = time.perf_counter()
    outs = None
    for _ in range(reps):
        outs = [fn(d) for d in data]
    return (time.perf_counter() - t) / reps / len(data) * 1000, outs


with torch.no_grad():
    def torch_ref(d):
        out, v = _evaluate(net, *d)
        return out.logits.float().numpy(), v.float().numpy()

    def torch_fast(d):
        return fast.logits(d[0]).float().numpy(), fast.value(*d).float().numpy()

    ms_a, ref = time_it(torch_ref)
    ms_d, fst = time_it(torch_fast)

results = {"A torch, training forward": ms_a, "D torch, search's forward": ms_d}
for tag, trim in (("B", False), ("C", True)):
    path = f"{OUT}/zero2-dst2-5000-{'trim' if trim else 'full'}.onnx"
    try:
        export(path, trim)
        sess = session(path)

        def run(d, sess=sess, trim=trim):
            xs = arrays(*d, trim=trim)
            feed = {n: x.numpy() for n, x in zip(names, xs)}
            return tuple(sess.run(["logits", "value"], feed))

        ms, outs = time_it(run)
        dl = max(float(np.abs(np.where(d[0].candidate_mask.numpy(), o[0] - r[0], 0)).max())
                 for d, o, r in zip(data, outs, ref))
        dv = max(float(np.abs(o[1] - r[1]).max()) for o, r in zip(outs, ref))
        agree = sum(int((o[0].argmax(1) == r[0].argmax(1)).sum()) for o, r in zip(outs, ref))
        total = sum(len(r[0]) for r in ref)
        label = f"{tag} ONNX Runtime, {'history trimmed' if trim else 'training shapes'}"
        results[label] = ms
        print(f"{label}: max|dlogit| {dl:.1e} max|dvalue| {dv:.1e} argmax {agree}/{total}")
    except Exception as e:  # noqa: BLE001
        print(f"{tag} failed: {type(e).__name__}: {str(e)[:600]}")
print("\nms per batch of 48 (actor logits + critic value), one thread:")
for k, v in results.items():
    print(f"  {k:45s} {v:7.1f}")
