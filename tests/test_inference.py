"""Automated integration tests for model loading and end-to-end inference."""

import os
import unittest
import numpy as np
import yaml
from PIL import Image

from src.inference import RoadVisionSegmentor
from src.visualization import color_mask, overlay_mask

class TestInference(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load configuration
        config_path = "configs/default_config.yaml"
        with open(config_path, "r") as f:
            cls.config = yaml.safe_load(f)
            
        # Initialize segmentor
        cls.segmentor = RoadVisionSegmentor(
            model_name=cls.config["model"]["name"],
            architecture=cls.config["model"]["architecture"],
            device="cpu"
        )
        
    def test_configuration_loading(self):
        """Verify configuration file exists and has essential keys."""
        self.assertIn("model", self.config)
        self.assertIn("classes", self.config)
        self.assertIn("corruptions", self.config)
        self.assertEqual(len(self.config["classes"]), 19)
        
    def test_synthetic_image_inference(self):
        """Run inference on synthetic RGB image and verify dimensions and keys."""
        H, W = 256, 512
        synth_img = np.zeros((H, W, 3), dtype=np.uint8)
        # Top half sky-like, bottom half road-like
        synth_img[:H//2, :] = [70, 130, 180]
        synth_img[H//2:, :] = [128, 64, 128]
        
        res = self.segmentor.predict(synth_img)
        
        self.assertIn("pred_mask", res)
        self.assertIn("colored_mask", res)
        self.assertIn("overlay", res)
        self.assertIn("latency_ms", res)
        self.assertIn("fps", res)
        
        # Verify dimension preservation
        self.assertEqual(res["pred_mask"].shape, (H, W))
        self.assertEqual(res["colored_mask"].shape, (H, W, 3))
        self.assertEqual(res["overlay"].shape, (H, W, 3))
        self.assertEqual(res["orig_shape"], (H, W))
        self.assertGreater(res["latency_ms"], 0)
        self.assertGreater(res["fps"], 0)
        
    def test_nonexistent_image_error_handling(self):
        """Segmentor should raise FileNotFoundError for missing image path."""
        with self.assertRaises(FileNotFoundError):
            self.segmentor.predict("nonexistent_path/fake_road.jpg")
            
    def test_color_mask_and_overlay_functions(self):
        """Test standalone mask colorization and alpha overlay."""
        mask = np.array([[0, 1], [10, 13]], dtype=np.uint8)
        colored = color_mask(mask)
        self.assertEqual(colored.shape, (2, 2, 3))
        self.assertEqual(colored.dtype, np.uint8)
        
        img = np.ones((2, 2, 3), dtype=np.uint8) * 100
        blended = overlay_mask(img, colored, alpha=0.5)
        self.assertEqual(blended.shape, (2, 2, 3))
        self.assertEqual(blended.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
