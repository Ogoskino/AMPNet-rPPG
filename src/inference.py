"""Small, label-free inference API for the released 128-frame checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = json.loads((Path(__file__).with_name("checkpoints.json")).read_text(encoding="utf-8"))


def resolve_device(device="cpu"):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda and auto.")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    return result


def validate_input(array, channels):
    """Validate normalized N,C,T,H,W face crops without guessing their layout."""
    array = np.asarray(array)
    expected = (channels, 128, 64, 64)
    if array.ndim != 5 or tuple(array.shape[1:]) != expected or array.shape[0] < 1:
        raise ValueError(f"Expected nonempty input (N, {channels}, 128, 64, 64); got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Input must contain floating-point normalized face crops, not integers.")
    if not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinity.")
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Input must already be normalized to [0, 1]; no automatic normalization is applied.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _checkpoint_paths(name, checkpoint_dir=None):
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}. Choose from {', '.join(MODEL_SPECS)}.")
    folder = Path(checkpoint_dir) if checkpoint_dir is not None else ROOT / "model_paths"
    paths = []
    for entry in MODEL_SPECS[name]["checkpoints"]:
        path = folder / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}. Obtain the complete release including model_paths.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {path.name}. Use the compatible bundled checkpoint.")
        paths.append(path)
    return paths


def _load(module, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state, strict=True)
    return module.to(device).eval().requires_grad_(False)


class _RegisteredAMPNet(nn.Module):
    """Register branches while keeping the existing fusion-only weight format."""
    def __init__(self, rgb, thermal, head):
        super().__init__()
        self.rgb = rgb
        self.thermal = thermal
        # The legacy head refers to these same modules through ordinary lists.
        self.head = head

    def forward(self, frames):
        return self.head(frames[:, :3], frames[:, 3:4])


def load_model(name, device="cpu", checkpoint_dir=None):
    device = resolve_device(device)
    paths = _checkpoint_paths(name, checkpoint_dir)
    from src.EDSAN import EDSAN

    if name == "ampnet":
        from src.AMPNET import AMPNet
        rgb = _load(EDSAN(frames=128), paths[0], device)
        thermal = _load(EDSAN(frames=128, n_channels=1, model="thermal"), paths[1], device)
        head = _load(AMPNet([rgb], [thermal]), paths[2], device)
        return _RegisteredAMPNet(rgb, thermal, head).to(device).eval().requires_grad_(False)
    if name == "r3edsan":
        model = EDSAN(frames=128)
    elif name.startswith("t3"):
        model = EDSAN(frames=128, n_channels=1, model="thermal",
                      is_cbam=name in {"t3edsan-cbam", "t3edsan-cs"},
                      is_tam=name in {"t3edsan-tam", "t3edsan-cs"})
    elif name == "physnet":
        from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    elif name == "ibvpnet":
        from src.iBVPNet import iBVPNet
        model = iBVPNet(frames=128, in_channels=3, debug=False)
    elif name == "rtrppg":
        from src.RTrPPG import N3DED64
        model = N3DED64(frames=128)
    else:
        raise ValueError(f"Unsupported model {name!r}")
    return _load(model, paths[0], device)


def predict_array(model_name, frames, batch_size=1, device="cpu", checkpoint_dir=None):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_name!r}.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    frames = validate_input(frames, MODEL_SPECS[model_name]["channels"])
    device = resolve_device(device)
    model = load_model(model_name, device=device, checkpoint_dir=checkpoint_dir)
    keys = ("waveform", "rgb_waveform", "thermal_waveform") if model_name == "ampnet" else ("waveform",)
    collected = {key: [] for key in keys}
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[start:start + batch_size]).to(device)
            output = model(batch)
            outputs = output if isinstance(output, tuple) else (output,)
            for key, tensor in zip(keys, outputs):
                values = tensor.detach().cpu().numpy()
                if values.shape != (len(batch), 128) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid {key} output: expected finite (N, 128), got {values.shape}.")
                collected[key].append(values)
    return {key: np.concatenate(chunks, axis=0) for key, chunks in collected.items()}
