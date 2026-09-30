# Prediction release notes

## 2026-09-30 release preparation

This release adds a direct, label-free tensor-to-waveform prediction workflow to the AMPNet research code. Release preparation takes place in a separate checkout; it does not change the original local working copy or imply that a release has already been pushed to GitHub.

### Prediction workflow

- A standalone `predict.py` command accepts prepared `.npy`, `.npz`, and plain-tensor `.pt`/`.pth` inputs.
- Nine model keys cover AMPNet, RGB models, and thermal models.
- Inputs use a fixed 128-frame, 64-by-64 crop contract; outputs contain predicted waveforms and JSON metadata, including checkpoint hashes.
- Checkpoint lookup is independent of the caller's working directory.
- CPU/CUDA device selection, explicit output replacement, model listing, and a random-input demo are exposed through the CLI.
- `requirements-predict.txt` separates minimal NumPy/PyTorch inference dependencies from the broader research environment.

### Checkpoints and reproducibility

The prediction models use bundled, compatible **128-frame retained checkpoints** selected by [src/checkpoints.json](src/checkpoints.json). The manifest records model keys, channel counts, checkpoint filenames, and SHA-256 hashes. These are existing weights, not models retrained for this release. Checkpoint compatibility must be interpreted together with the model architecture and preprocessing; a filename alone is not sufficient provenance.

Do not substitute older checkpoints merely because they have similar names. Legacy-named 192-frame weights and modified local architectures are not interchangeable with the new CLI's manifest-selected 128-frame weights; temporal dimensions or convolution groups may differ. Runtime checkpoint hashes identify the exact weights used for a prediction.

### Documentation

The README now distinguishes ready tensor prediction from legacy training/dataset evaluation, corrects the usual GitHub clone directory name, and links the detailed [prediction guide](PREDICTION.md). Existing paper citation, numerical results, and figures are retained and explicitly identified as historical rather than newly measured results.

### Configuration and validation

- No raw-video/webcam inference, automatic face preprocessing, modality synchronization, or heart-rate estimator is included in this prediction interface.
- `--fps` is metadata only. Predicted waveforms are not probabilities, clinical measurements, or heart-rate values.
- A random-input demo and synthetic tests do not measure physiological accuracy or fairness.
- Dataset paths now default to the repository's `datasets/` directory, configurable through `AMPNET_DATA_DIR`. MLflow defaults to local `mlruns/`, configurable through `MLFLOW_TRACKING_URI`; a server is not mandatory.
- Legacy training/testing still requires source data, correct tensor layout/order and splits, and the broader dependencies.

Validation passed with Python 3.12.7, PyTorch 2.6.0+cpu, and NumPy 1.26.4: 17 unit tests and two pretrained smoke tests covering all nine models. Fixed-seed two-window inputs also produced exact parity with the corresponding local implementations and selected retained checkpoints: maximum absolute difference 0 for every model and all three AMPNet outputs. See [Release validation](README.md#release-validation) for commands. This validation concerns software operation, not a new dataset-level performance study or complete training reproduction.

The missing `data` source package is restored and its Python files are no longer excluded by Git. Tracked Python bytecode caches are removed from this release copy; the prior versions remain recoverable from repository history. A GitHub Actions workflow is included for Windows and Linux prediction checks but has not yet run remotely.
