"""Prediction contracts and isolated dependency check; no training or real inference."""

from pathlib import Path
import os
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inference


MODEL_NAMES = {
    "ampnet", "r3edsan", "physnet", "ibvpnet", "rtrppg",
    "t3ed", "t3edsan-tam", "t3edsan-cbam", "t3edsan-cs",
}


class InputValidationTests(unittest.TestCase):
    def test_all_nine_public_models_are_listed(self):
        self.assertEqual(set(inference.MODEL_SPECS), MODEL_NAMES)

    def test_supported_float_types_and_channel_counts(self):
        for dtype in (np.float16, np.float32, np.float64):
            for channels in (1, 3, 4):
                with self.subTest(dtype=dtype, channels=channels):
                    frames = np.full((1, channels, 128, 64, 64), .5, dtype=dtype)
                    result = inference.validate_input(frames, channels=channels)
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(result.flags.c_contiguous)
                    np.testing.assert_array_equal(result, frames)

    def test_zero_and_one_are_valid_endpoints(self):
        for value in (0., 1.):
            with self.subTest(value=value):
                inference.validate_input(
                    np.full((1, 1, 128, 64, 64), value, dtype=np.float32),
                    channels=1,
                )

    def test_wrong_rank_dimensions_channels_and_empty_batch(self):
        shapes = [
            (1, 128, 64, 64),
            (1, 1, 1, 128, 64, 64),
            (1, 3, 128, 64, 64),
            (1, 1, 127, 64, 64),
            (1, 1, 128, 63, 64),
            (1, 1, 128, 64, 63),
            (0, 1, 128, 64, 64),
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                with self.assertRaises((ValueError, TypeError)):
                    inference.validate_input(np.zeros(shape, np.float32), channels=1)

    def test_nonfinite_and_out_of_range_are_rejected(self):
        frames = np.zeros((1, 1, 128, 64, 64), np.float32)
        for value in (np.nan, np.inf, -np.inf, -.001, 1.001):
            with self.subTest(value=value):
                frames.flat[0] = value
                with self.assertRaises((ValueError, TypeError)):
                    inference.validate_input(frames, channels=1)

    def test_nonfloating_input_is_not_silently_normalized(self):
        for dtype in (np.uint8, np.int32, np.bool_, np.complex64, "U1"):
            with self.subTest(dtype=dtype):
                frames = np.zeros((1, 1, 128, 64, 64), dtype=dtype)
                with self.assertRaises((ValueError, TypeError)):
                    inference.validate_input(frames, channels=1)

    def test_invalid_frames_are_rejected_before_checkpoint_loading(self):
        with mock.patch.object(inference, "load_model", side_effect=AssertionError("checkpoint loaded")) as load:
            with self.assertRaises((ValueError, TypeError)):
                inference.predict_array("ampnet", np.zeros((1, 4, 2, 4, 4), np.float32))
            load.assert_not_called()

    def test_invalid_batch_sizes_are_rejected_before_checkpoint_loading(self):
        frames = np.zeros((1, 1, 128, 64, 64), np.float32)
        for size in (0, -1, 1.5, True, "1"):
            with self.subTest(size=size):
                with mock.patch.object(inference, "load_model") as load:
                    with self.assertRaises(ValueError):
                        inference.predict_array("t3ed", frames, batch_size=size)
                    load.assert_not_called()


class PredictionCliTests(unittest.TestCase):
    def run_cli(self, directory, *args):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "predict.py"), *map(str, args)],
            cwd=directory, env=environment, capture_output=True, text=True,
            timeout=60, check=False,
        )

    def test_list_models_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, "--list-models")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in MODEL_NAMES:
                self.assertIn(name, result.stdout.lower())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_model_is_a_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, "--model", "not-a-model", "--demo",
                                  "--output", Path(directory) / "out.npz")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.npz"
            result = self.run_cli(directory, "--model", "ampnet", "--input",
                                  Path(directory) / "missing.npy", "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_wrong_shape_is_a_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "frames.npy", Path(directory) / "out.npz"
            np.save(source, np.zeros((1, 4, 4, 4, 4), np.float32))
            result = self.run_cli(directory, "--input", source, "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.npz"
            sentinel = b"existing-user-output"
            output.write_bytes(sentinel)
            result = self.run_cli(directory, "--demo", "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), sentinel)

    def test_existing_output_is_rejected_before_input_or_inference(self):
        import predict
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.npz"
            sentinel = b"existing-user-output"
            output.write_bytes(sentinel)
            with mock.patch.object(predict, "predict_array") as infer, \
                    mock.patch.object(predict, "read_input") as read, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    predict.main(["--input", "not-read.npy", "--output", str(output)])
                self.assertEqual(error.exception.code, 2)
                infer.assert_not_called()
                read.assert_not_called()
            self.assertEqual(output.read_bytes(), sentinel)

    def test_output_hard_link_cannot_overwrite_input(self):
        import predict
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.npz", Path(directory) / "out.npz"
            source.write_bytes(b"protected-input")
            try:
                os.link(source, output)
            except OSError:
                self.skipTest("Hard links unavailable on this filesystem")
            with mock.patch.object(predict, "predict_array") as infer, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    predict.main(["--input", str(source), "--output", str(output), "--overwrite"])
                self.assertEqual(error.exception.code, 2)
                infer.assert_not_called()
            self.assertEqual(source.read_bytes(), b"protected-input")

    def test_npz_requires_explicit_frames_key(self):
        import predict
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frames.npz"
            np.savez(source, other=np.zeros((1,), np.float32))
            with self.assertRaisesRegex(ValueError, "frames"):
                predict.read_input(source)

    def test_inference_does_not_require_training_or_preprocessing_dependencies(self):
        code = """
import importlib.abc
import sys
from pathlib import Path
blocked = {'mediapipe', 'mlflow', 'cv2', 'sklearn', 'matplotlib'}
class DenyTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked test dependency: ' + fullname)
        return None
sys.meta_path.insert(0, DenyTrainingDependencies())
sys.path.insert(0, sys.argv[1])
import predict
from src.inference import load_model
model = load_model('ampnet', device='cpu')
assert not model.training
assert not any(name.split('.')[0] in blocked for name in sys.modules)
assert not list(Path.cwd().iterdir()), 'Import created files in the working directory'
print('Isolated AMPNet checkpoint loading passed')
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code, str(ROOT)],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=60, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("checkpoint loading passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
