"""Run real checkpoint-loading/forward checks without participant data."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import MODEL_SPECS, load_model, predict_array

torch.set_num_threads(2)


class PretrainedSmokeTests(unittest.TestCase):
    def test_all_bundled_models(self):
        rng = np.random.default_rng(7)
        for name, spec in MODEL_SPECS.items():
            with self.subTest(model=name):
                frames = rng.random((1, spec["channels"], 128, 64, 64), dtype=np.float32)
                output = predict_array(name, frames)
                self.assertEqual(output["waveform"].shape, (1, 128))
                self.assertTrue(np.isfinite(output["waveform"]).all())
                if name == "ampnet":
                    self.assertEqual(set(output), {"waveform", "rgb_waveform", "thermal_waveform"})

    def test_ampnet_registered_frozen_and_repeatable(self):
        model = load_model("ampnet")
        self.assertTrue(all(not child.training for child in model.modules()))
        self.assertTrue(all(not param.requires_grad for param in model.parameters()))
        self.assertTrue(any(key.startswith("rgb.") for key in model.state_dict()))
        frames = torch.rand(1, 4, 128, 64, 64)
        with torch.inference_mode():
            a, b = model(frames), model(frames)
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
