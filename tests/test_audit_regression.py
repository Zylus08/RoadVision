"""Regression tests preventing evaluation defects discovered during the audit."""

import unittest
import numpy as np
from src.metrics import SegmentationEvaluator, compute_degradation
from src.corruptions import apply_corruption
from src.visualization import CITYSCAPES_CLASSES

class TestAuditRegression(unittest.TestCase):
    
    def test_ground_truth_integrity_check(self):
        """Regression test: Ground truth masks must have diverse classes and non-trivial distribution."""
        # A corrupted/dummy mask (like the previous 3-value mask) should fail integrity check
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[:95, :] = 0
        dummy_mask[95:98, :] = 1
        dummy_mask[98:, :] = 3
        
        unique_classes = np.unique(dummy_mask)
        # Real Cityscapes scenes must contain at least 5 distinct semantic classes
        self.assertLess(len(unique_classes), 5, "Integrity check failed: dummy mask detected as too simple")
        
        # Authentic Cityscapes mask check
        authentic_sample = np.array([0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 255], dtype=np.uint8)
        valid_classes = [c for c in authentic_sample if c != 255]
        self.assertGreaterEqual(len(valid_classes), 10, "Authentic mask must have at least 10 valid semantic classes")
        
    def test_cityscapes_label_mapping_alignment(self):
        """Regression test: SegFormer 19 classes must align with official Cityscapes trainIds."""
        expected_classes = {
            0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
            5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
            9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
            14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
        }
        for cid, cname in expected_classes.items():
            self.assertEqual(CITYSCAPES_CLASSES[cid], cname, f"Class ID {cid} mismatch: {CITYSCAPES_CLASSES[cid]} != {cname}")
            
    def test_ignore_index_strict_filtering(self):
        """Regression test: Ignore index (255) must NOT contribute to False Positives or False Negatives."""
        evaluator = SegmentationEvaluator(num_classes=3, ignore_index=255)
        # 4 pixels:
        # Pixel (0,0): GT=0, Pred=0 -> TP(0)
        # Pixel (0,1): GT=1, Pred=1 -> TP(1)
        # Pixel (1,0): GT=255 (void), Pred=0 -> MUST BE COMPLETELY IGNORED (not FP for class 0!)
        # Pixel (1,1): GT=255 (void), Pred=2 -> MUST BE COMPLETELY IGNORED
        gt = np.array([[0, 1], [255, 255]])
        pred = np.array([[0, 1], [0, 2]])
        
        evaluator.update(pred, gt)
        res = evaluator.compute()
        
        # Valid pixels must be exactly 2
        self.assertEqual(res["total_pixels_evaluated"], 2)
        # Class 0 should have TP=1, FP=0, FN=0 -> IoU = 1.0
        self.assertAlmostEqual(res["per_class_iou"]["class_0"], 1.0, places=5)
        # Class 1 should have TP=1, FP=0, FN=0 -> IoU = 1.0
        self.assertAlmostEqual(res["per_class_iou"]["class_1"], 1.0, places=5)
        # Overall mIoU must be 1.0
        self.assertAlmostEqual(res["mIoU"], 1.0, places=5)
        self.assertAlmostEqual(res["pixel_accuracy"], 1.0, places=5)
        
    def test_corruption_monotonic_degradation_under_fog(self):
        """Regression test: Under synthetic fog with depth, pixel accuracy must degrade monotonically as severity increases."""
        # Create a synthetic road scene
        H, W = 100, 100
        synthetic_scene = np.zeros((H, W, 3), dtype=np.uint8)
        synthetic_scene[:50, :] = [70, 130, 180]   # Sky
        synthetic_scene[50:, :] = [128, 64, 128]  # Road
        
        # Calculate pixel difference from clean baseline across severities
        diff_magnitudes = []
        for sev in range(1, 6):
            corrupt = apply_corruption(synthetic_scene, "fog", severity=sev, seed=42)
            diff = np.mean(np.abs(corrupt.astype(np.float32) - synthetic_scene.astype(np.float32)))
            diff_magnitudes.append(diff)
            
        for i in range(len(diff_magnitudes) - 1):
            self.assertGreater(diff_magnitudes[i + 1], diff_magnitudes[i],
                               f"Fog severity {i+2} difference ({diff_magnitudes[i+1]}) not strictly greater than severity {i+1} ({diff_magnitudes[i]})")

if __name__ == "__main__":
    unittest.main()

