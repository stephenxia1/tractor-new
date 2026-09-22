"""Self-play training on Modal.

A thin wrapper: everything that is actually training lives in
``tractor.train.loop`` so the same code runs under pytest locally. This
file supplies a GPU, a volume for checkpoints, and the wandb secret.

    modal run modal_train.py::smoke                 # a few steps, cheap
    modal run modal_train.py::main --steps 2000
"""

from __future__ import annotations

import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent
REPO = "/repo"
RESEARCH = f"{REPO}/research"
VOLUME = "/vol"

app = modal.App("tractor-train")
volume = modal.Volume.from_name("tractor-rl", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "torch==2.6.0", "wandb>=0.17")
    .add_local_dir(
        LOCAL_REPO / "research/py",
        f"{RESEARCH}/py",
        copy=True,
        ignore=["**/__pycache__", "**/*.pt", "**/fixtures/games.jsonl"],
    )
    .env({"PYTHONPATH": f"{RESEARCH}/py", "PYTHONUNBUFFERED": "1"})
    .workdir(RESEARCH)
)

# `Secret.from_name` is lazy, so a wrong name raises at deploy time and
# takes down every function in the app, not only the ones that wanted
# it. Hydrating here turns that into a local check with a clear answer.
# Absent, the run still trains and still writes JSONL to the volume:
# `Run` degrades rather than failing, so a missing secret costs charts
# and not the run.
def _wandb_secret() -> list:
    for candidate in ("wandb-secret", "wandb"):
        try:
            secret = modal.Secret.from_name(candidate)
            secret.hydrate()
            return [secret]
        except Exception:  # pragma: no cover - depends on the workspace
            continue
    print("modal_train: no wandb secret found; metrics go to JSONL only")
    return []


WANDB = _wandb_secret()


def _train(steps: int, hands: int, name: str, gpu_label: str,
           wandb_on: bool, resume: str, overrides: dict) -> dict:
    import os

    import torch

    from dataclasses import asdict

    from tractor.telemetry import Run
    from tractor.train.loop import TrainConfig, train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = f"{VOLUME}/{name}/checkpoints"
    os.makedirs(ckpt, exist_ok=True)

    cfg = TrainConfig(
        total_steps=steps,
        hands_per_step=hands,
        checkpoint_dir=ckpt,
        **overrides,
    )
    enabled = wandb_on and bool(os.environ.get("WANDB_API_KEY"))
    with Run(
        project="tractor-rl",
        name=name,
        config={**asdict(cfg), "device": device, "gpu": gpu_label},
        enabled=enabled,
        jsonl=f"{VOLUME}/{name}/metrics.jsonl",
    ) as run:
        out = train(cfg, device=device, run=run,
                    resume=resume or None)
    volume.commit()
    out["device"] = device
    return out


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              secrets=WANDB, timeout=86400, cpu=8.0, memory=32768)
def train_a10g(steps: int, hands: int, name: str, wandb_on: bool,
               resume: str, overrides: dict) -> dict:
    return _train(steps, hands, name, "A10G", wandb_on, resume, overrides)


@app.function(image=image, gpu="A100-40GB", volumes={VOLUME: volume},
              secrets=WANDB, timeout=86400, cpu=16.0, memory=65536)
def train_a100(steps: int, hands: int, name: str, wandb_on: bool,
               resume: str, overrides: dict) -> dict:
    return _train(steps, hands, name, "A100-40GB", wandb_on, resume, overrides)


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              secrets=WANDB, timeout=3600, cpu=8.0, memory=32768)
def train_smoke() -> dict:
    """Three steps on a handful of hands: proves the loop runs on a GPU."""
    return _train(3, 4, "smoke", "A10G", False, "",
                  {"buffer_capacity": 2048, "checkpoint_every": 0,
                   "ppo": {"epochs": 1, "minibatch": 64}})


@app.local_entrypoint()
def smoke() -> None:
    import json

    print(json.dumps(train_smoke.remote(), indent=2, default=str))


@app.local_entrypoint()
def main(steps: int = 1000, hands: int = 64, name: str = "run",
         gpu: str = "a10g", wandb_on: bool = True, resume: str = "",
         eval_every: int = 0, eval_deals: int = 64,
         checkpoint_every: int = 50, keep_last: int = 3,
         anneal_steps: int = 400) -> None:
    import json

    fn = {"a10g": train_a10g, "a100": train_a100}[gpu.lower()]
    overrides = {
        "eval_every": eval_every,
        "eval_deals": eval_deals,
        "checkpoint_every": checkpoint_every,
        "keep_last": keep_last,
        "anneal_steps": anneal_steps,
    }
    print(json.dumps(
        fn.remote(steps, hands, name, wandb_on, resume, overrides),
        indent=2, default=str))
