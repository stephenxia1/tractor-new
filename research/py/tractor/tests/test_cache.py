"""The incremental history cache against the batched encoder.

Within one hand the encoding at play t depends only on plays up to t,
so appending is exact rather than approximate. These tests hold the two
paths to the same output under the increment patterns a real hand
produces, and check that the cache cannot silently serve stale or
over-long state.
"""

from __future__ import annotations

import numpy as np
import torch

from tractor.nn.features import HISTORY_DIM
from tractor.nn.model import HistoryEncoder, ModelConfig

TOL = 2e-5


def _encoder(seed: int = 0) -> tuple[HistoryEncoder, ModelConfig]:
    torch.manual_seed(seed)
    cfg = ModelConfig()
    enc = HistoryEncoder(cfg).eval()
    return enc, cfg


def _history(cfg: ModelConfig, batch: int, length: int) -> torch.Tensor:
    torch.manual_seed(1234)
    hist = torch.randn(batch, cfg.max_history, HISTORY_DIM)
    hist[:, length:] = 0.0
    return hist


@torch.no_grad()
def test_append_matches_the_batched_encoder() -> None:
    """Fed in the increments a hand produces, one play then four."""
    enc, cfg = _encoder()
    batch, length = 3, 99
    hist = _history(cfg, batch, length)
    lens = torch.full((batch,), length, dtype=torch.long)
    batched, _keep = enc(hist, lens)

    cache = enc.new_cache(batch)
    pos = 0
    steps = [1, 1, 1, 1] + [4] * 30
    for step in steps:
        if pos >= length:
            break
        take = min(step, length - pos)
        enc.append(cache, hist[:, pos : pos + take])
        pos += take
    assert pos == length
    assert cache.length == length

    diff = (cache.encoded[:, :length] - batched[:, :length]).abs().max().item()
    print(f"    {length} plays appended in increments: max abs diff {diff:.2e}")
    assert diff < TOL


@torch.no_grad()
def test_any_increment_pattern_gives_the_same_answer() -> None:
    """One at a time, all at once, and a ragged mixture agree."""
    enc, cfg = _encoder(7)
    batch, length = 2, 40
    hist = _history(cfg, batch, length)

    results = []
    for pattern in ([1] * length, [length], [3, 1, 7, 12, 2, 15]):
        cache = enc.new_cache(batch)
        pos = 0
        for step in pattern:
            if pos >= length:
                break
            take = min(step, length - pos)
            enc.append(cache, hist[:, pos : pos + take])
            pos += take
        assert pos == length
        results.append(cache.encoded[:, :length].clone())

    for other in results[1:]:
        assert (results[0] - other).abs().max().item() < TOL
    print(f"    three increment patterns agree over {length} plays")


@torch.no_grad()
def test_prefix_is_stable_as_the_hand_grows() -> None:
    """A position's encoding never changes once written.

    This is the property that makes the cache sound; if it failed, the
    cache would be serving a stale prefix rather than a correct one.
    """
    enc, cfg = _encoder(11)
    batch, length = 2, 60
    hist = _history(cfg, batch, length)

    cache = enc.new_cache(batch)
    snapshots = []
    for pos in range(0, length, 5):
        enc.append(cache, hist[:, pos : pos + 5])
        snapshots.append((pos + 5, cache.encoded[:, : pos + 5].clone()))

    final = cache.encoded[:, :length]
    for upto, snap in snapshots:
        diff = (snap - final[:, :upto]).abs().max().item()
        assert diff < TOL, f"prefix of {upto} moved after later appends"
    print(f"    {len(snapshots)} prefixes all stable to the end of the hand")


@torch.no_grad()
def test_cache_refuses_to_overrun() -> None:
    enc, cfg = _encoder()
    cache = enc.new_cache(1)
    enc.append(cache, torch.zeros(1, cfg.max_history, HISTORY_DIM))
    assert cache.length == cfg.max_history
    try:
        enc.append(cache, torch.zeros(1, 1, HISTORY_DIM))
    except ValueError:
        print(f"    overrun past {cfg.max_history} slots raised")
        return
    raise AssertionError("appending past the cache size must raise")


@torch.no_grad()
def test_reset_starts_a_new_hand() -> None:
    enc, cfg = _encoder(3)
    batch, length = 2, 20
    hist = _history(cfg, batch, length)

    cache = enc.new_cache(batch)
    enc.append(cache, hist[:, :length])
    first = cache.encoded[:, :length].clone()

    cache.reset()
    assert cache.length == 0
    enc.append(cache, hist[:, :length])
    diff = (cache.encoded[:, :length] - first).abs().max().item()
    assert diff == 0.0, "a reset cache must re-encode identically"
    print("    reset then re-encode is bit-identical")


@torch.no_grad()
def test_the_cache_is_worth_its_complexity() -> None:
    """Appending a trick's worth of plays beats re-encoding the hand."""
    import time

    enc, cfg = _encoder()
    batch = 32
    hist = _history(cfg, batch, 99)
    lens = torch.full((batch,), 99, dtype=torch.long)

    def timed(fn, reps: int = 5) -> float:
        fn()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - start) / reps

    def full() -> None:
        enc(hist, lens)

    def append_four() -> None:
        cache = enc.new_cache(batch)
        cache.length = 95
        enc.append(cache, hist[:, 95:99])

    ratio = timed(full) / timed(append_four)
    print(f"    full encode vs four-play append: {ratio:.1f}x")
    # The arithmetic ratio is about 25x. A realised figure below 5x
    # would mean the append path is dominated by overhead rather than
    # by the work it avoids, and the cache would not be earning its
    # complexity.
    assert ratio > 5.0, f"cache gave only {ratio:.1f}x"


def main() -> int:
    import time

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
