# AMPNet: Attentive Multimodal Pulse Network for Robust & Fair rPPG

Official implementation of **AMPNet**, a remote photoplethysmography (rPPG) model using RGB and thermal facial video. This release's prediction entry point runs bundled checkpoints on prepared face-crop tensors, without ground-truth labels or an MLflow server.

See [PREDICTION.md](PREDICTION.md) for the full input/output contract and [RELEASE_NOTES.md](RELEASE_NOTES.md) for release scope and checkpoint caveats.

## Quick start: predict a waveform

Use Python 3.10 or newer; Python 3.12 is recommended and was tested for this release. From the repository directory containing `predict.py`:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS, then install the prediction dependencies:

```bash
python -m pip install -r requirements-predict.txt
python predict.py --list-models
python predict.py --model ampnet --demo --output demo_waveform.npz --device cpu
```

The demo uses random tensors: it is an execution smoke test, not a physiological example or an accuracy test. Bundled checkpoints require no additional download.

For real, already prepared crops:

```bash
python predict.py --model ampnet --input crops.npz --output predicted_waveform.npz --device auto --batch-size 1 --fps 28
```

Existing output files are protected; add `--overwrite` only when replacement is intended. Input and output paths are relative to your working directory, while bundled checkpoint paths are resolved relative to the code. You can invoke `predict.py` by its full path from another directory.

For a GitHub clone, the usual commands are `git clone https://github.com/Ogoskino/AMPNet-rPPG.git` and `cd AMPNet-rPPG`; the checkout must contain this release's `predict.py`.

## What the predictor accepts and returns

Input is a finite floating-point tensor normalized to `[0, 1]`, shaped **`(N, C, 128, 64, 64)`**: independent windows, channels, time, height, width. Inputs are converted to float32; normalization is not automatic.

| Model key | Channels | Input |
|---|---:|---|
| `ampnet` | 4 | RGB channels followed by one synchronized thermal channel |
| `r3edsan`, `physnet`, `ibvpnet`, `rtrppg` | 3 | RGB face crops, in RGB channel order |
| `t3ed`, `t3edsan-tam`, `t3edsan-cbam`, `t3edsan-cs` | 1 | Thermal face crops |

Supported containers are `.npy`, `.npz` with a `frames` array, and `.pt`/`.pth` containing a plain tensor. This is **not a raw MP4/webcam predictor**: face detection, cropping, alignment, synchronization, resizing, normalization, and segmentation must happen beforehand.

The output `.npz` contains `waveform` shaped `(N, 128)` and a `metadata_json` string. AMPNet also returns `rgb_waveform` and `thermal_waveform`. These are predicted **rPPG waveforms**, not probabilities, diagnoses, or heart-rate values. `--fps` records metadata only; it does not resample the input. Meaningful heart-rate estimation requires suitable recording duration and separate signal processing.

## Checkpoints

[src/checkpoints.json](src/checkpoints.json) declares the prediction models, channel counts, checkpoint filenames, and SHA-256 hashes. The CLI uses these compatible 128-frame retained weights; no retraining is performed. Older, legacy-named 192-frame checkpoints are not interchangeable with this interface. Do not choose weights by filename resemblance alone.

## Project structure

```text
predict.py                Tensor-to-waveform prediction CLI
requirements-predict.txt  Minimal prediction dependencies
PREDICTION.md              Input/output details and examples
model_paths/              Bundled trained checkpoints
src/                      Architectures and checkpoint manifest
train.py, test.py         Legacy dataset training/evaluation entry points
config.py                 Legacy experiment configuration
preprocessing/            Historical dataset preprocessing
evaluation/, signals/     Metrics and signal-processing utilities
```

## Training and dataset evaluation

Prediction does not require labels. The `train.py` and `test.py` workflows do: they expect prepared feature and waveform-label tensors, appropriate dataset ordering, and experiment-specific configuration.

Before using those workflows:

- Install the broader dependencies in `requirements.txt` and inspect the dataset loader's format requirements.
- Put `ibvp_train_features.pth`, `ibvp_train_labels.pth`, `ibvp_test_features.pth`, and `ibvp_test_labels.pth` under the repository's `datasets/` directory, or set `AMPNET_DATA_DIR` to your prepared-data directory.
- Confirm model names, checkpoint filenames, modality, split construction, sampling rate, and full-session length for your dataset.
- MLflow defaults to the repository's local `mlruns/` directory; no tracking server is required. Set `MLFLOW_TRACKING_URI` only if you want a different backend.

For training/evaluation, the loader reads channels-last session tensors `(N, T, 64, 64, C)` and labels `(N, T)`, then splits and transposes them into model inputs `(segments, C, 128, 64, 64)`. This differs from the ready-segment input accepted by `predict.py`. The historical iBVP configuration uses `SESSION_LENGTH = 1792`, `SEGMENT_LENGTH = 128`, and `SAMPLING_RATE = 28`, giving 14 segments per session. Session reconstruction requires correct chronological order and session boundaries. These settings are dataset-specific.

Only after those prerequisites are satisfied:

```bash
python train.py
python test.py
```

## Paper and results

**AMPNet: An Attentive Multimodal Pulse Network for Equitable and Robust Remote Photoplethysmography (rPPG)**, IEEE Sensors Journal, 2026. DOI: `10.1109/JSEN.2026.3706851`.

AMPNet combines RGB and thermal modalities, a 3D CNN encoder-decoder, spatiotemporal attention, and decision-level fusion to address illumination, motion, and skin-tone variability. The table and figures below present the project's existing reported results.

### Reported iBVP results

Lower MAE/RMSE is better; higher correlation (`r`), SNR, and MACC is better.

| Model | MAE | RMSE | r | SNR | MACC |
|---|---:|---:|---:|---:|---:|
| PhysNet | 2.717 | 5.957 | 0.716 | 7.433 | 0.624 |
| iBVPNet | 3.264 | 6.438 | 0.643 | 5.522 | 0.555 |
| RTrPPG | 2.666 | 5.877 | 0.734 | 5.680 | 0.504 |
| 3EDSAN | 1.504 | 3.033 | 0.933 | 7.095 | 0.628 |
| AMPNet | 1.248 | 2.458 | 0.958 | 7.098 | 0.696 |

### Architecture and example outputs

![AMPNet architecture](ampnet_architecture.png)

![Heart-rate example](hr_plot.png)

![BVP waveform examples](bvp_signals.png)

The work evaluates demographic robustness and resolution/temporal perturbations. Cross-dataset validation, thermal noise, temporal sensitivity, and real-time deployment remain directions for further work.

## Release validation

Validation passed with Python 3.12.7, PyTorch 2.6.0+cpu, and NumPy 1.26.4: 17 unit tests and two pretrained smoke tests covering all nine models, finite waveforms, and repeatable AMPNet inference. On fixed-seed, two-window synthetic batches, all nine released models exactly matched the corresponding local implementations with the selected retained weights (maximum absolute difference 0, including all three AMPNet outputs).

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_pretrained.py
```

These are software/checkpoint checks, not a new dataset-accuracy evaluation; the reported paper results above are retained unchanged. Complete training reproduction is outside this release's validation scope.

## Citation

```bibtex
@ARTICLE{11589532,
  author={Okafor, Ogonna and Adama, David Ada and Dangana, Muhammad and Vinkemeier, Doratha},
  journal={IEEE Sensors Journal},
  title={AMPNet: An Attentive Multimodal Pulse Network for Equitable and Robust Remote Photoplethysmography (rPPG)},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Heart rate;Videos;Training;Estimation;Measurement;Skin;Modules (abstract algebra);Attention mechanisms;Signal to noise ratio;Blood Volume Pulse (BVP) estimation;Convolutional Block Attention Module (CBAM);Multimodal fusion;Remote photoplethysmography (rPPG);RGB-thermal imaging;Spatiotemporal attention;Temporal Attention Module (TAM)},
  doi={10.1109/JSEN.2026.3706851}
}
```

## Author

Ogonna Okafor, Nottingham Trent University.
