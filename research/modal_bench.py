"""GPU benchmark for the Tractor network, run on Modal.

Measures what the local machine cannot: real CUDA timings, memory
peaks, model FLOP utilisation against the analytic cost, and what
``torch.compile`` and reduced precision are worth. Every figure is
printed next to the estimate it tests.

    modal run modal_bench.py                      # A10G, the default size
    modal run modal_bench.py --gpu a100-40gb
    modal run modal_bench.py --batch 512 --quick
"""

from __future__ import annotations

import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent
REPO = "/repo"
RESEARCH = f"{REPO}/research"

app = modal.App("tractor-bench")

# Lean next to `modal_app.py`'s image: the benchmark touches only torch
# and the Python package, so Node and the npm workspace are left out and
# the image builds in a fraction of the time.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "torch==2.6.0")
    .add_local_dir(
        LOCAL_REPO / "research/py",
        f"{RESEARCH}/py",
        copy=True,
        ignore=["**/__pycache__", "**/*.pt", "**/fixtures/games.jsonl"],
    )
    .env({"PYTHONPATH": f"{RESEARCH}/py", "PYTHONUNBUFFERED": "1"})
    .workdir(RESEARCH)
)


@app.function(image=image, gpu="A10G", timeout=1800)
def probe() -> dict:
    """Confirm the GPU is real before spending a benchmark on it."""
    import torch

    assert torch.cuda.is_available(), "no CUDA device"
    name = torch.cuda.get_device_name(0)
    # A property read is not proof the device runs anything; launch a
    # kernel whose result would be wrong if it had not.
    a = torch.randn(1024, 1024, device="cuda")
    out = float((a @ a).sum().item())
    assert out == out, "matmul produced NaN"
    props = torch.cuda.get_device_properties(0)
    return {
        "device": name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "multi_processor_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _bench_body(batch: int, candidates: int, history: int, quick: bool) -> dict:
    import time

    import torch

    from tractor.nn.model import (
        ActorCritic,
        ModelConfig,
        analytic_flops,
        dummy_batch,
    )
    from tractor.telemetry import memory_snapshot, mfu, peak_tflops

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cfg = ModelConfig()
    reps = 5 if quick else 20

    obs, priv = dummy_batch(cfg, batch=batch, n_candidates=candidates,
                            history_len=history)
    obs, priv = obs.to(dev), priv.to(dev)
    net = ActorCritic(cfg).to(dev).eval()

    # HistoryEncoder.forward runs all `max_history` positions densely
    # and applies the length mask multiplicatively, so the work does
    # not shrink when rows are shorter. Pricing it at the ragged
    # length would divide real work by a smaller number and
    # understate every rate derived from it.
    flops = analytic_flops(cfg, batch=batch, n_candidates=candidates,
                           history_len=cfg.max_history)
    fwd_flops = flops["total"]

    def timed(fn, warmup: int = 3) -> tuple[float, dict]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        stop.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(stop) / 1e3 / reps
        return seconds, memory_snapshot()

    props = torch.cuda.get_device_properties(0)
    # `clock_rate` is not on the properties object in torch 2.6, so
    # the clock comes from the driver. GA102 dense rates per SM per
    # clock are 256 fp32, 512 TF32, 1024 BF16, which is what the
    # peak table is built from; deriving it here independently means
    # a container whose silicon disagrees with the table says so.
    clock_mhz = 0.0
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        clock_mhz = float(out.stdout.strip().splitlines()[0])
    except Exception:  # pragma: no cover - driver may not answer
        pass
    sm = props.multi_processor_count
    derived_fp32 = 256.0 * sm * clock_mhz * 1e6 / 1e12

    results: dict = {
        "device": name,
        "sm_count": props.multi_processor_count,
        "max_sm_clock_mhz": clock_mhz,
        "derived_peak_fp32": round(derived_fp32, 2),
        "derived_peak_tf32": round(2 * derived_fp32, 2),
        "derived_peak_bf16": round(4 * derived_fp32, 2),
        "batch": batch,
        "candidates": candidates,
        "history": history,
        "analytic_gflop_forward": round(fwd_flops / 1e9, 3),
        "analytic_gflop_per_decision": round(fwd_flops / batch / 1e9, 4),
        "peak_tflops_fp32": peak_tflops(name, "fp32"),
        "peak_tflops_bf16": peak_tflops(name, "bf16"),
        "variants": {},
        "components": {},
        "cache": {},
    }

    # --- whole forward, across precision and compilation ----------------
    variants: list[tuple[str, object]] = [("eager", net)]
    if not quick:
        try:
            variants.append(
                ("compile", torch.compile(net, dynamic=False)))
            variants.append(
                ("reduce-overhead",
                 torch.compile(net, mode="reduce-overhead", dynamic=False))
            )
        except Exception as exc:  # pragma: no cover - backend availability
            results["compile_error"] = f"{type(exc).__name__}: {exc}"

    for label, module in variants:
        for dtype_name, ctx in (
            ("fp32", torch.autocast("cuda", enabled=False)),
            ("bf16", torch.autocast("cuda", dtype=torch.bfloat16)),
        ):
            key = f"{label}/{dtype_name}"
            try:
                with torch.no_grad():
                    def run() -> None:
                        with ctx:
                            module(obs, priv)

                    seconds, mem = timed(run)
            except Exception as exc:  # pragma: no cover
                results["variants"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            achieved = fwd_flops / seconds
            results["variants"][key] = {
                "ms": round(seconds * 1e3, 3),
                "decisions_per_s": round(batch / seconds, 1),
                "tflops": round(achieved / 1e12, 2),
                "mfu": round(mfu(fwd_flops, seconds, name, dtype_name) or 0.0, 4),
                "peak_alloc_mb": round(mem["cuda_max_allocated_bytes"] / 1e6, 1),
                "peak_reserved_mb": round(mem["cuda_max_reserved_bytes"] / 1e6, 1),
                # CUDA graphs allocate from a private pool that the
                # allocated counter does not see, so for those arms only
                # the reserved figure reflects what the device holds.
                "alloc_counter_trustworthy": label != "reduce-overhead",
            }

    # --- where the time goes, component by component --------------------
    hist = obs.history
    lens = obs.history_len
    with torch.no_grad():
        enc, keep = net.actor.history(hist, lens)

        pieces = {
            "history": lambda: net.actor.history(hist, lens),
            "query": lambda: net.actor.query(obs, enc, keep),
        }
        for label, fn in pieces.items():
            seconds, mem = timed(fn)
            share = flops.get(label, 0) / max(fwd_flops, 1)
            results["components"][label] = {
                "ms": round(seconds * 1e3, 3),
                "analytic_share": round(share, 3),
                "peak_alloc_mb": round(mem["cuda_max_allocated_bytes"] / 1e6, 1),
            }

    # --- the history cache ----------------------------------------------
    # The cache persists across a hand's decisions, so allocating one per
    # repetition would charge the cache for work the training loop never
    # does. Prime it once, then rewind only the length counter, which is
    # a scalar assignment.
    with torch.no_grad():
        cache = net.actor.history.new_cache(batch, device=dev)
        cache.tokens[:, :history] = hist[:, :history]
        net.actor.history.advance(cache, hist, history)
        prime = max(history - 4, 0)

        def full() -> None:
            net.actor.history(hist, lens)

        def advance_four() -> None:
            cache.length = prime
            net.actor.history.advance(cache, hist, history)

        # The cache must return what a full encode returns, or a speedup
        # is just a wrong answer computed quickly. `advance` takes one
        # scalar length for the whole batch, so the reference encode is
        # given uniform lengths; comparing it against the ragged lengths
        # `dummy_batch` produces would be comparing two different
        # computations and would report a drift that means nothing.
        uniform = torch.full_like(lens, history)
        cache.length = prime
        got = net.actor.history.advance(cache, hist, history)
        ref, _keep = net.actor.history(hist, uniform)
        drift = float((got - ref[:, :history]).abs().max().item())
        scale = float(ref.abs().max().item())

        # A persistent cache is not free: per-layer keys and values for
        # every position are held for the life of the hand.
        cache_bytes = sum(
            t.numel() * t.element_size()
            for t in (list(cache.keys) + list(cache.values)
                      + [cache.encoded, cache.tokens])
        )

        def advance_named() -> None:
            cache.length = prime
            net.actor.history.advance(cache, hist, history,
                                      recompute_from=prime)

        f_s, f_mem = timed(full)
        a_s, a_mem = timed(advance_four)
        named_s, _ = timed(advance_named)
        cached = analytic_flops(cfg, batch=batch, n_candidates=candidates,
                                history_len=4)
        results["cache"] = {
            "full_ms": round(f_s * 1e3, 3),
            "advance_ms": round(a_s * 1e3, 3),
            "advance_named_ms": round(named_s * 1e3, 3),
            "measured_speedup": round(f_s / max(a_s, 1e-9), 2),
            "measured_speedup_no_sync": round(
                f_s / max(named_s, 1e-9), 2),
            "sync_cost_ms": round((a_s - named_s) * 1e3, 3),
            # Deliberately not a ratio. The arithmetic model prices
            # four tokens attending over four keys, while the timed
            # path attends over the whole prefix, so the two are not
            # the same computation and dividing them would invite a
            # comparison that means nothing.
            "analytic_history_gflop_full": round(
                flops["history"] / 1e9, 3),
            "analytic_history_gflop_4tok": round(
                cached["history"] / 1e9, 3),
            "full_peak_mb": round(f_mem["cuda_max_allocated_bytes"] / 1e6, 1),
            "cache_peak_mb": round(a_mem["cuda_max_allocated_bytes"] / 1e6, 1),
            "max_abs_drift": round(drift, 6),
            "activation_scale": round(scale, 4),
            "relative_drift": round(drift / max(scale, 1e-9), 8),
            "cache_resident_mb": round(cache_bytes / 1e6, 1),
        }

    # Python has no block scope: `enc`, `ref` and the primed `cache`
    # are still live and would be charged to every peak measured
    # after this point, roughly 460 MB at batch 256.
    del enc, keep, ref, got, cache
    torch.cuda.empty_cache()

    # --- is the peak in the MFU denominator the real one? ---------------
    # Every MFU above divides by a table value. A measured large GEMM is
    # the only thing that says whether that value is the device's.
    def gemm(dtype, n: int = 8192) -> float:
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            a @ b
        stop.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(stop) / 1e3 / 10
        del a, b
        torch.cuda.empty_cache()
        return 2.0 * n ** 3 / sec / 1e12

    was_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32_strict = gemm(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    tf32_measured = gemm(torch.float32)
    bf16_measured = gemm(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = was_tf32
    results["gemm_tflops"] = {
        "fp32_no_tf32": round(fp32_strict, 2),
        "fp32_tf32": round(tf32_measured, 2),
        "bf16": round(bf16_measured, 2),
        "table_fp32": results["peak_tflops_fp32"],
        "table_bf16": results["peak_tflops_bf16"],
    }

    # --- does the telemetry clock agree with the wall clock? ------------
    from tractor.telemetry import Spans

    spans = Spans(enabled=True, device="cuda")
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            with spans.span("forward"):
                net(obs, priv)
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    recorded = spans.summary().get("forward", {}).get("total", 0.0)
    for key, arm in results["variants"].items():
        if "tflops" not in arm:
            continue
        ceiling = (bf16_measured if key.endswith("bf16")
                   else tf32_measured)
        arm["mfu_vs_measured_gemm"] = round(arm["tflops"] / ceiling, 4)
    results["gemm_tflops"]["note"] = (
        "mfu divides by the datasheet peak; mfu_vs_measured_gemm "
        "divides by the dense GEMM this container actually reached"
    )

    results["span_crosscheck"] = {
        "spans_total_s": round(recorded, 5),
        "wall_synchronized_s": round(wall, 5),
        "ratio": round(recorded / max(wall, 1e-9), 4),
    }

    # --- where does the GPU saturate? ------------------------------------
    sweep: dict = {}
    if not quick:
        # One compile per batch size: with dynamic=False each shape
        # gets its own graph, which is what the sweep is measuring.
        best = None
        for b in (32, 64, 128, 256, 512):
            try:
                ob, pv = dummy_batch(cfg, batch=b, n_candidates=candidates,
                                     history_len=history)
                ob, pv = ob.to(dev), pv.to(dev)
                bf = analytic_flops(cfg, batch=b, n_candidates=candidates,
                                    history_len=history)["total"]
                # Compiled where the backend allows it, eager where it
                # does not. A sweep that reports nothing at all because
                # one backend refused is worse than a sweep that says
                # which mode each row ran in.
                torch._dynamo.reset()
                try:
                    best = torch.compile(net, dynamic=False)
                    with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        best(ob, pv)
                    mode = "compiled"
                except Exception:
                    best, mode = net, "eager"
                with torch.no_grad():
                    def run_b() -> None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            best(ob, pv)

                    sec, mem = timed(run_b)
                sweep[str(b)] = {
                    "mode": mode,
                    "ms": round(sec * 1e3, 3),
                    "decisions_per_s": round(b / sec, 1),
                    "mfu": round(mfu(bf, sec, name, "bf16") or 0.0, 4),
                    "peak_alloc_mb": round(
                        mem["cuda_max_allocated_bytes"] / 1e6, 1),
                }
                del ob, pv
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover
                sweep[str(b)] = {"error": f"{type(exc).__name__}: {exc}"}
    results["batch_sweep"] = sweep

    # --- a training step, across the same arms as the forward ----------
    # The learner's rate, not the actor's, sets how fast experience can
    # be consumed, so it is measured under compilation and reduced
    # precision rather than extrapolated from the eager figure.
    # Fixed targets, drawn once: a step whose targets changed between
    # repetitions would be timing the target generation too.
    gen = torch.Generator(device="cpu").manual_seed(0)
    legal_counts = obs.candidate_mask.sum(dim=1).clamp(min=1).cpu()
    act = (torch.rand(batch, generator=gen) * legal_counts).long()
    act = act.clamp(max=candidates - 1).unsqueeze(1).to(dev)
    adv = torch.randn(batch, generator=gen).to(dev)
    ret = torch.randn(batch, generator=gen).to(dev)
    belief_t = torch.rand(batch, 54, 5, generator=gen)
    belief_t = (belief_t / belief_t.sum(dim=-1, keepdim=True)).to(dev)
    n_out = int(net.actor.heads.outcome[-1].out_features)
    outcome_t = torch.randint(
        0, n_out, (batch,), generator=gen).to(dev)

    train_arms: dict = {}
    train_specs = [("eager", "fp32"), ("eager", "bf16")]
    if not quick:
        train_specs += [("compile", "fp32"), ("compile", "bf16")]

    for label, dtype_name in train_specs:
        key = f"{label}/{dtype_name}"
        try:
            train = ActorCritic(cfg).to(dev).train()
            module = (torch.compile(train, dynamic=False)
                      if label == "compile" else train)
            opt = torch.optim.AdamW(train.parameters(), lr=3e-4, fused=True)
            use_bf16 = dtype_name == "bf16"

            def step() -> None:
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    out, value = module(obs, priv, detach_critic=False)
                    # Shaped like the real objective. A sum over
                    # `log_probs.exp()` is the batch size by
                    # construction and its gradient is analytically
                    # zero, so a loss built from it times a graph that
                    # trains nothing. These terms all carry signal and
                    # together reach the policy, both value paths and
                    # both auxiliary heads.
                    logp = out.log_probs.gather(1, act).squeeze(1)
                    loss = (
                        -(logp * adv).mean()
                        + (out.value - ret).pow(2).mean()
                        + (value - ret).pow(2).mean()
                        - (out.belief_log_probs * belief_t).sum(-1).mean()
                        + torch.nn.functional.cross_entropy(
                            out.outcome_logits.float(), outcome_t)
                    )
                loss.float().backward()
                opt.step()

            s_s, s_mem = timed(step, warmup=3)
            train_arms[key] = {
                "ms": round(s_s * 1e3, 3),
                "decisions_per_s": round(batch / s_s, 1),
                "peak_alloc_mb": round(
                    s_mem["cuda_max_allocated_bytes"] / 1e6, 1),
                "peak_reserved_mb": round(
                    s_mem["cuda_max_reserved_bytes"] / 1e6, 1),
                # Forward plus backward is conventionally three forwards.
                "implied_tflops": round(3 * fwd_flops / s_s / 1e12, 2),
                "implied_mfu": round(
                    mfu(3 * fwd_flops, s_s, name, dtype_name) or 0.0, 4),
            }
            del train, module, opt
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover
            train_arms[key] = {"error": f"{type(exc).__name__}: {exc}"}

    results["train_step"] = train_arms

    return results


@app.function(image=image, gpu="A10G", timeout=3600)
def bench_a10g(batch: int, candidates: int, history: int, quick: bool) -> dict:
    return _bench_body(batch, candidates, history, quick)


@app.function(image=image, gpu="A100-40GB", timeout=3600)
def bench_a100(batch: int, candidates: int, history: int, quick: bool) -> dict:
    return _bench_body(batch, candidates, history, quick)


@app.function(image=image, gpu="L4", timeout=3600)
def bench_l4(batch: int, candidates: int, history: int, quick: bool) -> dict:
    return _bench_body(batch, candidates, history, quick)


@app.local_entrypoint()
def main(
    gpu: str = "a10g",
    batch: int = 256,
    candidates: int = 128,
    history: int = 99,
    quick: bool = False,
) -> None:
    import json

    print(json.dumps(probe.remote(), indent=2))
    fn = {"a10g": bench_a10g, "a100": bench_a100, "l4": bench_l4}[gpu.lower()]
    out = fn.remote(batch, candidates, history, quick)
    print(json.dumps(out, indent=2))


@app.function(image=image, cpu=8.0, memory=8192, timeout=3600)
def bench_cpu(hands: int = 60) -> dict:
    """The simulator side, on the same hardware as the GPU arm.

    The actor's throughput is set by whichever side is slower, so the
    two must be measured under the same conditions before they can be
    divided.
    """
    import os
    import time

    import numpy as np

    from tractor.bench.throughput import bench_encoder, bench_simulator

    rows = bench_simulator(hands=hands, seed=20260921, cap=4096)
    enc = bench_encoder(hands=max(hands // 3, 5), seed=7, cap=4096)
    out = {r.name: round(r.measured, 4) for r in rows + enc}
    out["cpu_count"] = os.cpu_count()
    return out


@app.local_entrypoint()
def cpu_only(hands: int = 60) -> None:
    import json

    print(json.dumps(bench_cpu.remote(hands), indent=2))


@app.function(image=image, gpu="A10G", timeout=1800)
def diagnose() -> str:
    """Full traceback for the inductor failure, which the matrix swallows.

    The benchmark records compile errors as a string per arm so that one
    broken backend does not cost the whole run; that is right for the
    matrix and useless for fixing the cause.
    """
    import io
    import os
    import traceback

    os.environ["TORCHDYNAMO_VERBOSE"] = "1"

    import torch

    from tractor.nn.model import ActorCritic, ModelConfig, dummy_batch

    cfg = ModelConfig()
    obs, priv = dummy_batch(cfg, batch=8, n_candidates=16, history_len=32)
    obs, priv = obs.to("cuda"), priv.to("cuda")
    net = ActorCritic(cfg).to("cuda").eval()

    log = io.StringIO()
    for label, grad, detach in (
        ("no_grad + detach=True", False, True),
        ("no_grad + detach=False", False, False),
        ("grad + detach=False", True, False),
        ("grad + detach=True", True, True),
    ):
        torch._dynamo.reset()
        compiled = torch.compile(net)
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                compiled(obs, priv, detach_critic=detach)
            log.write(f"OK    {label}\n")
        except Exception:
            log.write(f"FAIL  {label}\n")
            log.write(traceback.format_exc()[-2500:] + "\n")
    # The actor alone compiled before the critic was added; separating
    # the towers says which one the backend objects to.
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            torch.compile(net.actor)(obs)
        log.write("OK    actor alone, no_grad\n")
    except Exception:
        log.write("FAIL  actor alone, no_grad\n")
        log.write(traceback.format_exc()[-1500:] + "\n")
    return log.getvalue()


@app.local_entrypoint()
def diag() -> None:
    print(diagnose.remote())
