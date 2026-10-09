"""Synthetic raw-conversion contracts; no participant data or detector downloads."""

from pathlib import Path
import copy
import os
import random
import sys
import tempfile
import unittest
import zipfile

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_ibvp as prep


class FrameSourceTests(unittest.TestCase):
    def test_directory_natural_order_filters_extensions_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, payload in [("1000.bmp", b"last"), ("99.bmp", b"first"),
                                  ("100.bmp", b"middle"), ("README.txt", b"ignore")]:
                (directory / name).write_bytes(payload)
            with prep.FrameSource(directory, "rgb") as source:
                self.assertEqual([Path(str(n)).name for n in source.names],
                                 ["99.bmp", "100.bmp", "1000.bmp"])
                self.assertEqual([source.read(i) for i in range(3)],
                                 [b"first", b"middle", b"last"])

    def test_zip_natural_order_ignores_nonframes_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "recording.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("recording/", b"")
                z.writestr("recording/10.raw", b"ten")
                z.writestr("recording/2.raw", b"two")
                z.writestr("recording/1.raw", b"one")
                z.writestr("recording/README.txt", b"ignored")
                z.writestr("recording/3.bmp", b"other modality")
            with prep.FrameSource(archive, "thermal") as source:
                self.assertEqual([Path(str(n)).name for n in source.names],
                                 ["1.raw", "2.raw", "10.raw"])
                self.assertEqual([source.read(i) for i in range(3)],
                                 [b"one", b"two", b"ten"])
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             ["recording.zip"])
            # A closed source must release the ZIP so it can be renamed on Windows.
            archive.rename(directory / "renamed.zip")

    def test_archive_and_directory_agree_on_same_frame_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            unpacked = directory / "frames"
            unpacked.mkdir()
            archive = directory / "frames.zip"
            payloads = {"frame_11.png": b"eleven", "frame_3.png": b"three",
                        "frame_2.png": b"two"}
            with zipfile.ZipFile(archive, "w") as z:
                for name, payload in payloads.items():
                    (unpacked / name).write_bytes(payload)
                    z.writestr(name, payload)
            with prep.FrameSource(unpacked, "rgb") as a, prep.FrameSource(archive, "rgb") as b:
                self.assertEqual([a.read(i) for i in range(len(a.names))],
                                 [b.read(i) for i in range(len(b.names))])


class SamplingTests(unittest.TestCase):
    def test_legacy_random_matches_seeded_sorted_without_replacement(self):
        expected = sorted(random.Random(23).sample(range(100), 19))
        actual = prep.sample_indices(100, 19, "legacy-random", 23)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual,
                                      prep.sample_indices(100, 19, "legacy-random", 23))
        self.assertTrue(np.issubdtype(np.asarray(actual).dtype, np.integer))
        self.assertEqual(len(np.unique(actual)), 19)
        self.assertTrue(np.all(np.diff(actual) > 0))

    def test_uniform_covers_interval_with_balanced_spacing(self):
        actual = np.asarray(prep.sample_indices(12, 4, "uniform", 5))
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[-1], 11)
        self.assertEqual(len(actual), 4)
        self.assertTrue(np.all(np.diff(actual) > 0))
        self.assertLessEqual(np.ptp(np.diff(actual)), 1)
        np.testing.assert_array_equal(actual, prep.sample_indices(12, 4, "uniform", 88))

    def test_full_length_is_identity_for_both_modes(self):
        for mode in ("uniform", "legacy-random"):
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(prep.sample_indices(8, 8, mode, 2),
                                              np.arange(8))

    def test_shared_indices_preserve_rgb_thermal_bvp_correspondence(self):
        origin = np.arange(20)
        rgb = np.stack((origin, origin + 20, origin + 40), axis=1)
        thermal = origin + 100
        bvp = origin * 0.125
        selected = prep.sample_indices(20, 9, "legacy-random", 79)
        np.testing.assert_array_equal(rgb[selected, 0], thermal[selected] - 100)
        np.testing.assert_allclose(rgb[selected, 0] * 0.125, bvp[selected])
        # Selected positions retain elapsed-order information, not a shuffled clip.
        self.assertTrue(np.all(np.diff(rgb[selected, 0]) > 0))

    def test_invalid_lengths_and_mode_fail_instead_of_padding(self):
        for length, target in [(0, 1), (-1, 1), (3, 0), (3, -1), (3, 4)]:
            for mode in ("uniform", "legacy-random"):
                with self.subTest(length=length, target=target, mode=mode):
                    with self.assertRaises(ValueError):
                        prep.sample_indices(length, target, mode, 1)
        with self.assertRaises(ValueError):
            prep.sample_indices(8, 4, "not-a-mode", 1)


class BvpAndThermalTests(unittest.TestCase):
    def test_bvp_column_is_named_not_positionally_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bvp.csv"
            path.write_text("time,BVP,unused\n1000,-2,8\n1033,0,9\n1066,3,10\n",
                            encoding="utf-8")
            result = prep.read_bvp(path)
            self.assertEqual(result.shape, (3,))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, [-2, 0, 3])

    def test_explicit_bvp_column_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bvp.csv"
            path.write_text("time,pulse\n0,1.5\n1,2.5\n", encoding="utf-8")
            np.testing.assert_array_equal(prep.read_bvp(path, column="pulse"), [1.5, 2.5])

    def test_absent_header_empty_non_numeric_and_nonfinite_bvp_are_rejected(self):
        bad = ["0,1\n1,2\n", "time,pulse\n0,1\n", "BVP\n",
               "BVP\nhello\n", "BVP\nnan\n", "BVP\ninf\n", "BVP\n-inf\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for content in bad:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        prep.read_bvp(path)

    def test_thermal_little_endian_conversion_and_spatial_order(self):
        # Explicit bytes make the endian requirement independent of host byte order.
        payload = bytes([0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
                         0xAC, 0x1B, 0x10, 0x27, 0xFF, 0xFF])
        actual = prep.thermal_celsius(payload, width=3, height=2)
        expected = np.array([[0, 1, 256], [7084, 10000, 65535]], dtype=np.float32)
        expected = expected * 0.04 - 273.15
        self.assertEqual(actual.shape, (2, 3))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=3e-5)

    def test_thermal_payload_must_have_exact_size(self):
        for payload in (b"", b"\0" * 11, b"\0" * 13, b"\0" * 24):
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    prep.thermal_celsius(payload, width=3, height=2)


class CropTests(unittest.TestCase):
    def test_rgb_crop_preserves_spatial_and_channel_identity(self):
        plane = np.arange(16, dtype=np.float32).reshape(4, 4)
        rgb = np.stack([plane, plane + 100, plane + 200], axis=-1)
        actual = prep.crop_resize(rgb, (1, 1, 2, 2), size=2)
        np.testing.assert_array_equal(actual, rgb[1:3, 1:3])
        np.testing.assert_array_equal(actual[..., 1] - actual[..., 0], 100)
        np.testing.assert_array_equal(actual[..., 2] - actual[..., 0], 200)

    def test_negative_box_is_clamped_instead_of_numpy_wraparound(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        actual = prep.crop_resize(image, (-1, -1, 3, 3), size=2)
        np.testing.assert_array_equal(actual, image[:2, :2])

    def test_box_past_bottom_right_is_clamped_before_resize(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        actual = prep.crop_resize(image, (2, 2, 5, 5), size=3)
        np.testing.assert_array_equal(actual, np.full((3, 3), 8))

    def test_missing_detection_returns_blank_without_dropping_time_position(self):
        for shape in ((5, 7), (5, 7, 3)):
            with self.subTest(shape=shape):
                image = np.full(shape, 17, np.float32)
                actual = prep.crop_resize(image, None, size=4)
                self.assertEqual(actual.shape, (4, 4) + shape[2:])
                np.testing.assert_array_equal(actual, 0)

    def test_empty_or_nonoverlapping_boxes_are_errors(self):
        image = np.ones((4, 4, 3), np.uint8)
        for box in [(0, 0, 0, 3), (0, 0, 3, 0), (0, 0, -1, 3),
                    (5, 0, 2, 2), (0, 5, 2, 2), (-5, -5, 2, 2)]:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    prep.crop_resize(image, box, size=2)


class NormalizationTests(unittest.TestCase):
    @staticmethod
    def fixture():
        features = np.array([[[[[10., 20., 30., 40.]]],
                              [[[20., 40., 60., 80.]]]]], dtype=np.float32)
        labels = np.array([[-5., 15.]], dtype=np.float32)
        stats = dict(rgb_max=60., thermal_max=80., bvp_min=-5., bvp_max=15.)
        return features, labels, stats

    def test_explicit_modality_scaling_is_inplace_and_preserves_channel_layout(self):
        features, labels, stats = self.fixture()
        original = features.copy()
        original_stats = dict(stats)
        feature_identity, label_identity = id(features), id(labels)
        result = prep.normalize_inplace(features, labels, stats)
        self.assertIsNone(result)
        self.assertEqual(id(features), feature_identity)
        self.assertEqual(id(labels), label_identity)
        self.assertEqual(stats, original_stats)
        np.testing.assert_allclose(features[..., :3], original[..., :3] / 60.)
        np.testing.assert_allclose(features[..., 3], original[..., 3] / 80.)
        np.testing.assert_array_equal(labels, [[0., 1.]])
        # Per-frame normalization would incorrectly erase this intensity ratio.
        np.testing.assert_allclose(features[:, 1], features[:, 0] * 2.)

    def test_supplied_stats_are_not_recomputed_from_the_current_recording(self):
        features, labels, stats = self.fixture()
        stats.update(rgb_max=120., thermal_max=160., bvp_min=-25., bvp_max=55.)
        prep.normalize_inplace(features, labels, stats)
        np.testing.assert_allclose(features.max(axis=(0, 1, 2, 3)),
                                   [1 / 6, 1 / 3, 1 / 2, 1 / 2])
        np.testing.assert_allclose(labels, [[.25, .5]])

    def test_zero_negative_or_nonfinite_maxima_and_constant_bvp_fail(self):
        replacements = [("rgb_max", 0.), ("thermal_max", 0.),
                        ("rgb_max", -1.), ("thermal_max", -1.),
                        ("rgb_max", np.nan), ("thermal_max", np.inf),
                        ("bvp_min", np.nan), ("bvp_max", np.inf),
                        ("bvp_max", -5.), ("bvp_max", -6.)]
        for key, value in replacements:
            with self.subTest(key=key, value=value):
                features, labels, stats = self.fixture()
                stats[key] = value
                with self.assertRaises(ValueError):
                    prep.normalize_inplace(features, labels, stats)


class SubjectSplitTests(unittest.TestCase):
    @staticmethod
    def sessions(subjects):
        return [dict(subject=subject, session=f"{subject}_{session}",
                     path=Path("unused") / f"{subject}_{session}")
                for subject in subjects for session in ("b", "a")]

    def test_explicit_subject_order_then_recording_order_and_no_input_mutation(self):
        sessions = self.sessions(["p05", "p02", "p28", "p26", "p22", "p03", "p99"])
        original = copy.deepcopy(sessions)
        result = prep.resolve_split(sessions, ["p03", "p02"], ["p22", "p26", "p28", "p05"])
        self.assertEqual(sessions, original)
        expected_subjects = [p for p in ["p03", "p02", "p22", "p26", "p28", "p05"] for _ in range(2)]
        self.assertEqual([row["subject"] for row in result], expected_subjects)
        self.assertEqual([row["session"] for row in result],
                         [f"{p}_{s}" for p in ["p03", "p02", "p22", "p26", "p28", "p05"]
                          for s in ("a", "b")])
        self.assertEqual([row["split"] for row in result], ["train"] * 4 + ["test"] * 8)
        self.assertNotIn("p99", {row["subject"] for row in result})

    def test_every_session_of_a_subject_stays_in_the_same_split(self):
        result = prep.resolve_split(self.sessions(["p02", "p22"]), ["p02"], ["p22"])
        for subject in ("p02", "p22"):
            selected = [row for row in result if row["subject"] == subject]
            self.assertEqual(len(selected), 2)
            self.assertEqual(len({row["split"] for row in selected}), 1)
        self.assertFalse({r["subject"] for r in result if r["split"] == "train"}
                         & {r["subject"] for r in result if r["split"] == "test"})

    def test_subject_overlap_is_rejected_even_for_different_sessions(self):
        with self.assertRaises(ValueError):
            prep.resolve_split(self.sessions(["p02", "p22"]), ["p02"], ["p02", "p22"])

    def test_requested_missing_subject_is_not_silently_removed(self):
        sessions = self.sessions(["p02", "p22"])
        for train, test in [(["p02", "p99"], ["p22"]), (["p02"], ["p22", "p99"])]:
            with self.subTest(train=train, test=test):
                with self.assertRaises(ValueError):
                    prep.resolve_split(sessions, train, test)




class SavedOutputGuardTests(unittest.TestCase):
    def test_verifier_rejects_negative_thermal_input_even_with_valid_file_hash(self):
        import hashlib
        import json
        import torch
        import prepare_ibvp
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            features = torch.zeros((1, 128, 64, 64, 4), dtype=torch.float32)
            features[0, 0, 0, 0, 3] = -0.1
            labels = torch.zeros((1, 128), dtype=torch.float32)
            fp, lp = folder/'features.pth', folder/'labels.pth'
            torch.save(features, fp)
            torch.save(labels, lp)
            metadata = {'features':fp.name, 'labels':lp.name,
                        'feature_shape':list(features.shape), 'label_shape':list(labels.shape),
                        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (fp,lp)}}
            (folder/'manifest.json').write_text(json.dumps({'outputs':{'train':metadata}}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'normalized video range'):
                prepare_ibvp.verify_output(folder)

if __name__ == "__main__":
    unittest.main()
