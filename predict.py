"""Predict rPPG waveforms from prepared face-crop tensors; no labels needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from src.inference import MODEL_SPECS, predict_array


def read_input(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "frames" not in archive:
                raise ValueError("NPZ input must contain an array named 'frames'.")
            return archive["frames"]
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError("Torch input must be a single tensor, not a checkpoint or dictionary.")
        return value.detach().cpu().numpy()
    raise ValueError("Input must be .npy, .npz (key 'frames'), .pt or .pth (a single tensor).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="ampnet")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, help="Normalized N,C,128,64,64 face crops.")
    group.add_argument("--demo", action="store_true", help="Run a synthetic smoke test, NOT a physiological example.")
    parser.add_argument("--output", type=Path, help="Output .npz file containing waveforms and provenance.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=28.0, help="Input frame rate metadata; does not resample inputs.")
    parser.add_argument("--threads", type=int, default=2, help="CPU worker threads.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models:
        for name, spec in MODEL_SPECS.items():
            print(f"{name:14s} C={spec['channels']}  T=128  H=W=64")
        return 0
    if args.input is None and not args.demo:
        parser.error("provide --input PATH or --demo")
    if args.output is None:
        parser.error("--output PATH.npz is required")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must have a .npz extension")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite only to replace predictions")
    if args.input is not None:
        same_path = args.input.resolve() == args.output.resolve()
        same_file = (args.input.exists() and args.output.exists()
                     and args.input.samefile(args.output))
        if same_path or same_file:
            parser.error("output cannot overwrite the input")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    torch.set_num_threads(args.threads)

    try:
        if args.demo:
            rng = np.random.default_rng(0)
            frames = rng.random((1, MODEL_SPECS[args.model]["channels"], 128, 64, 64), dtype=np.float32)
        else:
            frames = read_input(args.input)
        outputs = predict_array(args.model, frames, args.batch_size, args.device)
        metadata = {
            "model": args.model,
            "sampling_rate_hz": args.fps,
            "input_shape": list(frames.shape),
            "synthetic_demo": args.demo,
            "input_name": None if args.input is None else args.input.name,
            "checkpoint_manifest": MODEL_SPECS[args.model],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "waveform_units": "model output, arbitrary units; not BPM or a probability",
            "window_samples": 128,
            "time_axis": "time_seconds is relative to the start of each input segment",
        }
        outputs["time_seconds"] = np.arange(128, dtype=np.float64) / args.fps
        outputs["segment_index"] = np.arange(len(frames))
        outputs["metadata_json"] = np.asarray(json.dumps(metadata))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against another process creating it meanwhile.
        with args.output.open("wb" if args.overwrite else "xb") as handle:
            np.savez_compressed(handle, **outputs)
    except (OSError, ValueError, TypeError, RuntimeError, EOFError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(frames)} segment(s), 128 waveform samples each: {args.output}")
    if args.demo:
        print("Synthetic smoke test only. These waveforms are not a heart-rate measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
