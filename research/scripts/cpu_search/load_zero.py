"""ActorCritic with zero2-dst2 step 5000's actor and critic, from the app's export."""
import json
from pathlib import Path
import numpy as np, torch
from tractor.nn.model import ActorCritic, ModelConfig

# The app's float16 export, at the repository root.
W = Path(__file__).resolve().parents[3] / "packages/engine/src/rl/zero"

def load_zero(report: bool = False) -> ActorCritic:
    m = json.loads((W / "weights-zero.json").read_text())
    flat = np.fromfile(W / "weights-zero.bin", dtype="<f2").astype(np.float32)
    assert flat.size == m["elements"], (flat.size, m["elements"])
    net = ActorCritic(ModelConfig(**m["model_config"]))
    sd = net.state_dict()
    loaded = {}
    for name, t in m["tensors"].items():
        n = int(np.prod(t["shape"]))
        loaded[name] = torch.from_numpy(flat[t["offset"]: t["offset"] + n].copy()).reshape(t["shape"])
    missing = [k for k in sd if k not in loaded]
    extra = [k for k in loaded if k not in sd]
    net.load_state_dict(loaded, strict=False)
    if report:
        print("loaded", len(loaded), "missing", missing, "extra", extra)
    return net.eval()

if __name__ == "__main__":
    load_zero(report=True)
