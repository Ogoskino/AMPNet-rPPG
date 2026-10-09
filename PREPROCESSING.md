# Preparing iBVP data for AMPNet

[prepare_ibvp.py](prepare_ibvp.py) converts raw iBVP RGB frames, thermal frames and contact-BVP recordings into the four tensors used by AMPNet. You can run it from this repository or copy the file elsewhere. Dependency installation, model download and environment setup are built into the script.

The default preparation retains **5,376 frames per recording**, packed into **three chronological rows of 1,792 frames**. AMPNet divides each row into 14 non-overlapping 128-frame clips, giving **42 clips per recording**. This packing preserves all selected frames and keeps recording boundaries intact.

## Set up the converter

Install **64-bit Python 3.12.x**, then run:

```powershell
py -3.12 prepare_ibvp.py --setup
```

On Linux or macOS, use `python3.12` instead of `py -3.12`. If the Windows launcher is unavailable, use the full path to your Python 3.12 executable. Python must already be installed; no separate requirements file is needed.

Setup creates `.ibvp-venv` beside the script and installs these direct dependencies:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| PyTorch | 2.6.0 |
| OpenCV-contrib | 4.11.0.86 |
| MediaPipe | 0.10.32 |

Windows and Linux use the [official PyTorch CPU index](https://pytorch.org/get-started/previous-versions/#v260); macOS uses PyPI. A GPU is not required. Setup also downloads Google's [BlazeFace short-range model](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) and checks its SHA256 against the value embedded in the script.

The first setup needs internet access. Later runs automatically use the managed environment and downloaded model without shell activation. Global Python packages are left unchanged. The environment's `.ibvp-managed.json` records the installation checks and resolved dependency versions; transitive dependencies are recorded rather than all pinned in advance.

Use `--env-dir "D:\environments\ibvp"` during both setup and conversion to choose another location. Keep it outside the raw-data and output directories. Unrelated nonempty directories are refused. An interrupted installation can be retried with the same `--setup` command.

For an existing environment, `--use-current-env` disables automatic relaunch. Supply compatible dependencies yourself and, for MediaPipe Tasks, provide `--face-model`. This option cannot be combined with `--setup`.

The pinned wheels support Windows/Linux x86-64 and Apple Silicon macOS; Linux requires glibc 2.28 or newer. Setup has been tested on Windows x86-64. `--help` works before dependencies are installed.

## Arrange the input

Point `--input-dir` at the iBVP dataset root. Recording directories are discovered recursively and should contain:

```text
p32_d/
    p32_d_bvp.csv
    p32_d_rgb.zip       # or an extracted p32_d_rgb/ directory
    p32_d_t.zip         # or an extracted p32_d_t/ directory
```

RGB frames may be BMP, PNG, JPG or JPEG; thermal frames must be `.raw` files. ZIPs are read directly. Filenames are sorted naturally, so `2` precedes `10`, regardless of archive order. Duplicate recording names or frame identifiers are rejected.

The CSV must contain a named `BVP` column with finite numeric values. Use `--bvp-column` for another column name. The converter requires one BVP sample per RGB frame and rejects unequal lengths; it does not resample an independently timed contact signal.

Thermal frames default to 640 by 512 pixels, stored as little-endian unsigned 16-bit values. They are converted to Celsius using `raw * 0.04 - 273.15`.

## Inspect, convert and verify

Replace these example paths with your own. Start by checking the roster and disk estimate:

```powershell
py -3.12 prepare_ibvp.py --input-dir "C:\data\iBVP_Dataset" --output-dir "D:\ibvp_prepared" --inspect
```

A short two-recording check can help confirm that your files and environment work together:

```powershell
py -3.12 prepare_ibvp.py --input-dir "C:\data\iBVP_Dataset" --output-dir "D:\ibvp_check" --sessions p32_d p22_a --train-subjects p32 --test-subjects p22 --frames 256 --row-frames 128
py -3.12 prepare_ibvp.py --output-dir "D:\ibvp_check" --verify-only
```

To process the default participant roster with the full 5,376-frame selection:

```powershell
py -3.12 prepare_ibvp.py --input-dir "C:\data\iBVP_Dataset" --output-dir "D:\ibvp_prepared"
py -3.12 prepare_ibvp.py --output-dir "D:\ibvp_prepared" --verify-only
```

The same arguments work with Unix paths, for example `--input-dir /data/iBVP_Dataset --output-dir /data/ibvp_prepared`. You can combine `--setup` with conversion arguments for a single first-run command.

Output must be outside the raw dataset tree. Existing nonempty output directories are refused. For the 88-recording roster, the estimated video output is **28.9 GiB**, with a conservative **59.3 GiB peak disk requirement** including temporary files. Inspection calculates the requirement for your selection; leave additional free space for the environment and other files.

## Saved data

| Output | Contents |
| --- | --- |
| `ibvp_train_features.pth` / `ibvp_test_features.pth` | Float32 tensors shaped `(N, row_frames, 64, 64, 4)` |
| `ibvp_train_labels.pth` / `ibvp_test_labels.pth` | Float32 BVP tensors shaped `(N, row_frames)` |
| `manifest.json` | Settings, split membership, row mapping, normalization statistics and hashes |
| `frame_indices/<session>_frames.json` | Source indices, filenames, available timestamps and crop boxes |
| `verification.json` | Saved-output verification results |

Channels are ordered **R, G, B, thermal**. `N` counts packed rows, not participants. The AMPNet loader segments those rows and transposes them to `(batch, channels, time, height, width)`. Set `AMPNET_DATA_DIR` to the output directory before running training or evaluation; see the [README](README.md).

Verification checks saved shapes, ranges, hashes and recorded alignment indices. It does not establish that a detected box contains the correct face or that video and contact pulse are physiologically synchronized.

## Sampling, timing and normalization

| Option | Default | Alternative or purpose |
| --- | --- | --- |
| `--frames` | `5376` | Selected positions per recording |
| `--row-frames` | `1792` | Packing length; must be a multiple of 128 |
| `--sampling` | `legacy-random` | `uniform` spaces selections across source positions |
| `--seed` | `42` | Combined with a stable recording-name hash |
| `--alignment` | `index` | `timestamp` pairs thermal frames to RGB times |
| `--max-pair-offset-ms` | `50` | Maximum offset for timestamp pairing |
| `--analysis-rate` | `28` | Metadata only; does not resample |
| `--normalization` | `split` | `recording` scales recordings separately |

The selected frame count must divide into complete rows. Short recordings are rejected rather than padded. `legacy-random` samples without replacement and sorts the chosen indices, following the archived preprocessing approach with an explicit seed. It preserves chronology but does not create a uniformly sampled time grid.

Index alignment uses the same positions in RGB, thermal and BVP. Different dropped frames can therefore produce timing drift. Available filename timestamps and pairing offsets are recorded for review.

Timestamp alignment requires filenames whose stems are increasing integer Unix timestamps in milliseconds in both image streams. Each selected RGB frame is paired with the nearest thermal frame within the configured tolerance; thermal frames may be reused. BVP still follows the **RGB index**, assuming the CSV is already aligned one-to-one with RGB. Neither mode estimates contact-sensor delay.

The [AMPNet paper](https://doi.org/10.1109/JSEN.2026.3706851) describes 30 FPS acquisition and a 28-Hz reference/analysis rate without fully specifying their conversion. `--analysis-rate 28` follows the release configuration but does not resolve this discrepancy.

RGB and thermal values are divided by their respective normalization-group maxima. BVP uses `(value - minimum) / (maximum - minimum)`. Default `split` normalization computes statistics separately over the whole training split and whole test split. It therefore uses test-video extrema and the test reference-BVP range; development folds also share development-split statistics. This is an offline labeled-data convention, not preprocessing fitted only on training folds. `recording` computes the same quantities independently for each recording. Neither scope is fully specified by the paper.

## Face detection and missing crops

The managed environment uses MediaPipe Tasks with the short-range BlazeFace model. `--face-backend auto` instead selects legacy full-range `FaceDetection(model_selection=1)` when that API is available in an explicitly supplied environment. The two detectors are not interchangeable reproduction settings. `--face-backend` and `--face-model` allow explicit choices, which are recorded in the manifest. Google's [Face Detector guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python) describes the Tasks interface.

The first detected face is cropped, clamped to image bounds and resized to 64 by 64 using bilinear interpolation. Thermal detection uses a normalized JET visualization; the crop is taken from the Celsius array before the specified normalization. The converter uses bounding boxes, not FaceMesh alignment.

By default, missing detections become zero crops without dropping temporal positions. `--missing-face error` stops immediately; `--max-missing-face-fraction` sets a per-recording, per-modality limit. Its permissive default is `1.0`, although entirely missing modalities and constant BVP signals are rejected for each recording.

Detection quality needs attention: in the full `p32_d` check, **1,694 of 5,376 thermal frames were missed (31.5%)** and retained as zeros. Inspect these counts before training. Raising or lowering an acceptance threshold changes the dataset used by the experiment.

## Participant split

The default follows the paper's explicit table: **18 training participants and four test participants**. Its prose instead reports 19 training participants; the converter does not infer an additional subject.

Training IDs:

```text
p02 p03 p04 p06 p11 p12 p13 p14 p15 p17 p18 p21 p23 p24 p25 p27 p30 p32
```

Test recordings are grouped in this order:

| Subject | Paper's demographic description |
| --- | --- |
| p22 | Asian |
| p26 | Black |
| p28 | Caucasian |
| p05 | Mixed |

Custom `--train-subjects` and `--test-subjects` lists retain their order; overlap and missing requested subjects are errors. Sessions remain grouped by participant and ordered by recording name.

The existing demographic evaluator assumes four equal contiguous groups in the order above. A subset or unequal recording counts can invalidate its labels; it does not read membership from the manifest automatically. The two-recording check is therefore unsuitable for four-group demographic evaluation.

## Validation and scope

The release passed **66 tests** covering conversion, setup and prediction interfaces. Two complete recordings, `p32_d` and `p22_a`, were converted with 5,376 frames each and checked with the AMPNet loader and a forward pass. A fresh managed CPU environment also processed 256 frames per recording and reproduced the corresponding verified tensors exactly, while leaving its host environment unchanged.

The full 88-recording dataset has **not** been processed with this converter. These checks establish software compatibility, not reproduced heart-rate accuracy. The paper leaves the original random seed, frame selection, normalization scope and some crop/timing details unspecified; those choices are explicit here and recorded with each export.

Raw files are unchanged. Output tensors contain derived face pixels, while audit files contain numerical metadata rather than preview images. Dataset confidentiality and media-use restrictions continue to apply. Use the saved provenance, missing-face counts and timing summaries when deciding whether a prepared dataset is suitable for your experiment.
