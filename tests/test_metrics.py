"""Automated unit tests for quantitative segmentation metrics."""

import unittest
import numpy as np
from src.metrics import SegmentationEvaluator, compute_degradation

class TestSegmentationMetrics(unittest.TestCase):
    
    def test_perfect_prediction(self):
        """When prediction exactly matches ground truth, mIoU and Pixel Acc must be 1.0."""
        evaluator = SegmentationEvaluator(num_classes=3, ignore_index=255)
        # Ground truth: 2x2 grid with classes 0, 1, 2
        gt = np.array([[0, 1], [2, 0]])
        pred = np.array([[0, 1], [2, 0]])
        
        evaluator.update(pred, gt)
        res = evaluator.compute()
        
        self.assertAlmostEqual(res["mIoU"], 1.0, places=5)
        self.assertAlmostEqual(res["pixel_accuracy"], 1.0, places=5)
        for c in range(3):
            self.assertAlmostEqual(res["per_class_iou"][f"class_{c}"], 1.0, places=5)
            
    def test_manual_confusion_matrix(self):
        """Test with manually calculated 3-class confusion matrix values."""
        evaluator = SegmentationEvaluator(num_classes=3, ignore_index=255)
        # 4 pixels:
        # Pixel 1: gt=0, pred=0 -> TP(0)
        # Pixel 2: gt=0, pred=1 -> FN(0), FP(1)
        # Pixel 3: gt=1, pred=1 -> TP(1)
        # Pixel 4: gt=2, pred=2 -> TP(2)
        gt = np.array([[0, 0], [1, 2]])
        pred = np.array([[0, 1], [1, 2]])
        
        evaluator.update(pred, gt)
        res = evaluator.compute()
        
        # Class 0: TP=1, FP=0, FN=1 -> IoU = 1 / (1 + 0 + 1) = 0.5
        self.assertAlmostEqual(res["per_class_iou"]["class_0"], 0.5, places=5)
        # Class 1: TP=1, FP=1, FN=0 -> IoU = 1 / (1 + 1 + 0) = 0.5
        self.assertAlmostEqual(res["per_class_iou"]["class_1"], 0.5, places=5)
        # Class 2: TP=1, FP=0, FN=0 -> IoU = 1 / (1 + 0 + 0) = 1.0
        self.assertAlmostEqual(res["per_class_iou"]["class_2"], 1.0, places=5)
        
        # Mean IoU: (0.5 + 0.5 + 1.0) / 3 = 0.66667
        self.assertAlmostEqual(res["mIoU"], 2.0 / 3.0, places=4)
        # Pixel accuracy: 3 correct / 4 total = 0.75
        self.assertAlmostEqual(res["pixel_accuracy"], 0.75, places=5)
        
    def test_ignore_index_exclusion(self):
        """Ignore index pixels must be completely excluded from metrics."""
        evaluator = SegmentationEvaluator(num_classes=2, ignore_index=255)
        # 4 pixels, but 2 are ignore_index (255)
        gt = np.array([[0, 255], [1, 255]])
        pred = np.array([[0, 1], [1, 0]]) # Second column has arbitrary predictions
        
        evaluator.update(pred, gt)
        res = evaluator.compute()
        
        # Only pixels (0,0) [gt=0, pred=0] and (1,0) [gt=1, pred=1] should be evaluated
        self.assertEqual(res["total_pixels_evaluated"], 2)
        self.assertAlmostEqual(res["mIoU"], 1.0, places=5)
        self.assertAlmostEqual(res["pixel_accuracy"], 1.0, places=5)
        
    def test_shape_mismatch_error(self):
        """Mismatched prediction and ground truth shapes must raise ValueError."""
        evaluator = SegmentationEvaluator(num_classes=2)
        pred = np.zeros((10, 10))
        gt = np.zeros((10, 12))
        
        with self.assertRaises(ValueError):
            evaluator.update(pred, gt)
            
    def test_degradation_calculation(self):
        """Test calculation of absolute and relative degradation drops."""
        clean = {"mIoU": 0.80, "pixel_accuracy": 0.90}
        corrupt = {"mIoU": 0.60, "pixel_accuracy": 0.75}
        
        deg = compute_degradation(clean, corrupt)
        self.assertAlmostEqual(deg["absolute_drop"], 0.20, places=5)
        self.assertAlmostEqual(deg["relative_drop_pct"], 25.0, places=5)
        self.assertAlmostEqual(deg["pixel_accuracy_drop"], 0.15, places=5)

if __name__ == "__main__":
    unittest.main()
