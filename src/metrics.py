"""Quantitative evaluation metrics for semantic segmentation and robustness testing.

Provides confusion-matrix-based evaluation:
- Per-class Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)
- Overall Pixel Accuracy
- Robustness degradation metrics (Absolute & Relative drops)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union, Tuple

class SegmentationEvaluator:
    """Confusion matrix evaluator for multi-class semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        class_names: Optional[Dict[int, str]] = None,
        strict_mode: bool = False
    ):
        """Initialize segmentation evaluator.

        Args:
            num_classes: Number of distinct semantic classes.
            ignore_index: Ground truth class ID to ignore (void / boundary).
            class_names: Mapping from class ID to human-readable string.
            strict_mode: If True, raise ValueError when predictions are clipped.
                         If False (default), issue a warning and clip silently.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or {i: f"class_{i}" for i in range(num_classes)}
        self.strict_mode = strict_mode
        self.reset()
        
    def reset(self):
        """Reset the internal accumulated confusion matrix."""
        # Row index: ground truth class, Column index: predicted class
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
        
    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update confusion matrix with a single prediction-ground truth pair.

        Args:
            pred: 2D array of predicted class indices (H, W).
            target: 2D array of ground truth class indices (H, W).
        """
        pred = pred.astype(np.int64)
        target = target.astype(np.int64)

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} vs target shape {target.shape}")

        # Filter out ignored pixels
        valid_mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

        # Check for out-of-range predictions before clipping
        out_of_range_mask = (valid_pred < 0) | (valid_pred >= self.num_classes)
        if np.any(out_of_range_mask):
            num_clipped = int(np.sum(out_of_range_mask))
            msg = (f"Clipping {num_clipped} prediction(s) out of range [0, {self.num_classes - 1}]. "
                   f"Min pred: {valid_pred.min()}, Max pred: {valid_pred.max()}. "
                   f"This may indicate model output issues or label mismatch.")
            if self.strict_mode:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Ensure predicted classes are in range [0, num_classes - 1]
        valid_pred = np.clip(valid_pred, 0, self.num_classes - 1)

        # Compute flat 1D indices for fast 2D bincount: idx = target * num_classes + pred
        indices = valid_target * self.num_classes + valid_pred
        counts = np.bincount(indices, minlength=self.num_classes ** 2)

        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))
        self.total_samples += 1
        
    def compute(self, only_present_classes: bool = True) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute mIoU, per-class IoU, and pixel accuracy from the confusion matrix.
        
        IoU_i = TP_i / (TP_i + FP_i + FN_i)
        where:
            TP_i = C[i, i]
            FP_i = sum_j(C[j, i]) - C[i, i] (column sum minus diagonal)
            FN_i = sum_j(C[i, j]) - C[i, i] (row sum minus diagonal)
            
        Args:
            only_present_classes: If True, mIoU averages only over classes that appeared in either
                                 ground truth or predictions (denominator > 0).
                                 
        Returns:
            Dictionary containing 'mIoU', 'pixel_accuracy', 'per_class_iou', and 'confusion_matrix'.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = np.sum(cm, axis=0).astype(np.float64) - tp
        fn = np.sum(cm, axis=1).astype(np.float64) - tp
        
        denominator = tp + fp + fn
        
        per_class_iou = {}
        valid_ious = []
        
        for i in range(self.num_classes):
            name = self.class_names.get(i, f"class_{i}")
            if denominator[i] > 0:
                iou_i = float(tp[i] / denominator[i])
                per_class_iou[name] = iou_i
                valid_ious.append(iou_i)
            else:
                per_class_iou[name] = float('nan')
                if not only_present_classes:
                    valid_ious.append(0.0)
                    
        mIoU = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
        
        total_valid_pixels = np.sum(cm)
        pixel_accuracy = float(np.sum(tp) / total_valid_pixels) if total_valid_pixels > 0 else 0.0
        
        return {
            "mIoU": mIoU,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": per_class_iou,
            "evaluated_classes_count": len(valid_ious),
            "total_pixels_evaluated": int(total_valid_pixels),
            "total_samples": self.total_samples
        }

def compute_degradation(clean_metrics: Dict, corrupted_metrics: Dict) -> Dict[str, float]:
    """Compute absolute and relative performance degradation between clean and corrupted conditions.
    
    Args:
        clean_metrics: Output from SegmentationEvaluator.compute() on clean images.
        corrupted_metrics: Output from SegmentationEvaluator.compute() on corrupted images.
        
    Returns:
        Dict with 'clean_mIoU', 'corrupted_mIoU', 'absolute_drop', 'relative_drop_pct'.
    """
    clean_miou = clean_metrics["mIoU"]
    corrupt_miou = corrupted_metrics["mIoU"]
    
    abs_drop = clean_miou - corrupt_miou
    rel_drop_pct = (abs_drop / clean_miou * 100.0) if clean_miou > 0 else 0.0
    
    clean_acc = clean_metrics["pixel_accuracy"]
    corrupt_acc = corrupted_metrics["pixel_accuracy"]
    acc_drop = clean_acc - corrupt_acc
    
    return {
        "clean_mIoU": clean_miou,
        "corrupted_mIoU": corrupt_miou,
        "absolute_drop": abs_drop,
        "relative_drop_pct": rel_drop_pct,
        "clean_pixel_accuracy": clean_acc,
        "corrupted_pixel_accuracy": corrupt_acc,
        "pixel_accuracy_drop": acc_drop
    }
