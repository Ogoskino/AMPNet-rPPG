# Prediction guide

`predict.py` runs bundled, trained models on prepared face-crop tensors. It needs no ground-truth waveform or heart-rate label. It does not read raw videos or perform face detection, tracking, cropping, synchronization, resizing, or normalization.

## Installation

Use Python 3.10+; Python 3.12 is recommended and was tested for this release. In an activated virtual environment, from the release checkout:

```bash
python -m pip install -r requirements-predict.txt
python predict.py --list-models
```

Prediction requires NumPy and PyTorch, not the full training dependency stack or an MLflow server. CPU inference is supported. For GPU inference, use a PyTorch installation compatible with your CUDA environment; `--device cuda` requests CUDA explicitly, while `--device auto` chooses an available device.

## Input contract

Supply finite floating-point values normalized to `[0, 1]` in a five-dimensional array:

```text
(N, C, 128, 64, 64)
 windows, channels, frames, height, width
```

- `N` is the number of independent windows, not the CLI batch size.
- RGB-only models require `C = 3` in **RGB**, not OpenCV BGR, order.
- Thermal-only models require `C = 1`.
- AMPNet requires `C = 4`: RGB first, then thermal. The two modalities must represent the same time window and aligned face crops.
- The temporal length and spatial dimensions are fixed at 128 and 64 by 64 for this release.
- Floating-point inputs are cast to float32. Integer inputs, non-finite values, and values outside `[0, 1]` are not accepted.
- Prepare normalized crops before inference. Do not feed raw pixels or arbitrary temperature units and expect automatic normalization. Historical preprocessing scales RGB and thermal separately by their maxima; matching the preprocessing and acquisition conditions of the checkpoint matters. Guard against a zero divisor when preparing your own data.

The CLI does not infer axis order, reorder BGR channels, fill missing frames, split sessions, or resample time. To convert a channels-last array, **transpose axes**, not just reshape the same memory:

```python
import numpy as np

# prepared_crops is already normalized, shaped [N, 128, 64, 64, C].
frames = np.ascontiguousarray(
    prepared_crops.transpose(0, 4, 1, 2, 3), dtype=np.float32
)
np.savez_compressed("crops.npz", frames=frames)
```

Supported containers:

| Extension | Required contents |
|---|---|
| `.npy` | The input array itself |
| `.npz` | An array named `frames` |
| `.pt`, `.pth` | A plain PyTorch tensor, loaded with `weights_only=True` |

A training checkpoint/state dictionary is not a prediction input. File suffixes alone do not identify tensor contents. Use only input files from sources you trust.

## Commands

Run the multimodal model:

```bash
python predict.py --model ampnet --input crops.npz --output prediction.npz --device cpu --batch-size 1 --fps 28
```

Run an RGB model on a three-channel input:

```bash
python predict.py --model r3edsan --input rgb_crops.npy --output rgb_prediction.npz --device auto
```

Run a thermal model on a one-channel input:

```bash
python predict.py --model t3edsan-cs --input thermal_crops.npy --output thermal_prediction.npz --device cpu
```

List supported models or inspect all options:

```bash
python predict.py --list-models
python predict.py --help
```

Supported model keys are `ampnet`, `r3edsan`, `physnet`, `ibvpnet`, `rtrppg`, `t3ed`, `t3edsan-tam`, `t3edsan-cbam`, and `t3edsan-cs`. Each key selects its packaged checkpoint through [src/checkpoints.json](src/checkpoints.json); it is not a request to train or choose among models using your data. Older 192-frame checkpoints are not interchangeable with these manifest-selected 128-frame weights.

Use `--demo` **instead of** `--input` for a random-input smoke test:

```bash
python predict.py --model ampnet --demo --output demo_waveform.npz --device cpu
```

Random input has no physiological interpretation. A successful demo establishes execution, not heart-rate accuracy.

`--batch-size` controls the number of windows processed together; start with 1 if memory is limited. `--threads` controls CPU threads (default 2). `--fps` supplies sampling-rate metadata and the relative `time_seconds` axis: it changes neither model input values nor output length, and does not invoke heart-rate estimation. Replacing an existing output requires `--overwrite`.

Bundled checkpoints are found relative to `predict.py`, independent of the current working directory. User-supplied relative input/output paths still refer to the current working directory. Quote paths containing spaces, for example:

```powershell
python "C:\path with spaces\AMPNet-rPPG\predict.py" --model ampnet --input "C:\my data\crops.npz" --output "C:\my data\prediction.npz" --device cpu
```

## Output contract

The output is an `.npz` archive:

| Key | Shape/type | Meaning |
|---|---|---|
| `waveform` | `(N, 128)` | Predicted rPPG waveform for each input window |
| `rgb_waveform` | `(N, 128)` | RGB branch waveform; AMPNet only |
| `thermal_waveform` | `(N, 128)` | Thermal branch waveform; AMPNet only |
| `time_seconds` | `(128,)` | `0, 1/fps, ..., 127/fps`, relative to each window's start |
| `segment_index` | `(N,)` | Input-window indices, starting at zero |
| `metadata_json` | JSON string | Model, sampling-rate metadata, checkpoint identity/hashes, and run information |

Read the result without pickle:

```python
import json
import numpy as np

with np.load("prediction.npz", allow_pickle=False) as result:
    waveform = result["waveform"]
    metadata = json.loads(result["metadata_json"].item())

print(waveform.shape)
print(metadata)
```

These values are **not class probabilities or beats per minute**. At 28 fps, one 128-frame window spans approximately 4.57 seconds. Heart-rate estimation needs an appropriately long recording, known sampling times, and separate detrending/filtering/spectral or peak analysis. Do not concatenate unrelated windows or sessions to invent a continuous recording. The historical evaluation utilities are a separate workflow and are not automatically run by this CLI.

## Common problems

| Symptom | Check |
|---|---|
| Wrong input shape | Use `N,C,T,H,W`, with 128 frames and 64 by 64 pixels; preserve an explicit batch dimension even for one window. |
| Wrong number of channels | Use 4 for AMPNet, 3 for RGB models, and 1 for thermal models. |
| `.npz` rejected | Save the array under the `frames` key. |
| Torch input rejected | Save a plain tensor, not a model, state dictionary, or arbitrary Python object. |
| Invalid dtype/range | Supply finite floating-point values in `[0, 1]`; inspect normalization and missing/corrupt frames. |
| Output already exists | Choose a new path or explicitly pass `--overwrite`. |
| CUDA unavailable/out of memory | Use `--device cpu`, or reduce `--batch-size`. |
| Missing checkpoint | Keep the manifest-listed `model_paths` files with this release; changing the working directory should not affect their resolution. |
| Unexpected waveform | Verify RGB order, modality alignment, preprocessing, frame rate, and acquisition/domain compatibility before interpreting the output. |

## Scope of validation

See [Release validation](README.md#release-validation) for the checks performed and their scope.
