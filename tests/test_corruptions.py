"""Automated tests for environmental corruptions and determinism."""

import unittest
import numpy as np
from src.corruptions import (
    apply_corruption,
    CORRUPTION_REGISTRY,
    corrupt_low_light,
    corrupt_fog,
    corrupt_rain,
    corrupt_gaussian_blur,
    corrupt_gaussian_noise
)

class TestCorruptions(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic 100x100 RGB road image
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
    def test_all_corruptions_registered(self):
        """Verify that all five required corruptions are registered."""
        expected = {"low_light", "fog", "rain", "gaussian_blur", "gaussian_noise"}
        self.assertEqual(set(CORRUPTION_REGISTRY.keys()), expected)
        
    def test_output_shape_and_dtype(self):
        """All corruptions must preserve shape (H, W, 3) and uint8 dtype."""
        for c_type in CORRUPTION_REGISTRY.keys():
            for sev in [1, 3, 5]:
                out = apply_corruption(self.image, c_type, severity=sev, seed=42)
                self.assertEqual(out.shape, self.image.shape, f"Failed for {c_type} sev {sev}")
                self.assertEqual(out.dtype, np.uint8, f"Failed dtype for {c_type}")
                self.assertTrue(np.all(out >= 0) and np.all(out <= 255), f"Pixel out of range for {c_type}")
                
    def test_deterministic_reproducibility(self):
        """With fixed random seed, transformations must yield identical outputs."""
        for c_type in CORRUPTION_REGISTRY.keys():
            out1 = apply_corruption(self.image, c_type, severity=3, seed=1234)
            out2 = apply_corruption(self.image, c_type, severity=3, seed=1234)
            self.assertTrue(np.array_equal(out1, out2), f"Corruption {c_type} is not deterministic with fixed seed!")
            
    def test_low_light_monotonic_dimming(self):
        """Higher low_light severity must result in lower average intensity."""
        means = []
        for sev in range(1, 6):
            out = corrupt_low_light(self.image, severity=sev)
            means.append(np.mean(out))
            
        # Check strict monotonic decrease
        for i in range(len(means) - 1):
            self.assertGreater(means[i], means[i + 1], f"Severity {i+1} mean {means[i]} not greater than severity {i+2} mean {means[i+1]}")
            
    def test_gaussian_noise_monotonic_variance(self):
        """Higher gaussian noise severity must increase difference variance."""
        var_diffs = []
        for sev in range(1, 6):
            out = corrupt_gaussian_noise(self.image, severity=sev, seed=42)
            diff = out.astype(np.float32) - self.image.astype(np.float32)
            var_diffs.append(np.var(diff))
            
        for i in range(len(var_diffs) - 1):
            self.assertGreater(var_diffs[i + 1], var_diffs[i], f"Noise variance not increasing at severity {i+2}")
            
    def test_invalid_corruption_type(self):
        """Unsupported corruption type must raise ValueError."""
        with self.assertRaises(ValueError):
            apply_corruption(self.image, "tornado", severity=1)

if __name__ == "__main__":
    unittest.main()
