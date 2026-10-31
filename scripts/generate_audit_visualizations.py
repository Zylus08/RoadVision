"""Script to generate comprehensive audit visualizations: Original, GT, Pred, Overlay, and Side-by-Side."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from src.inference import RoadVisionSegmentor
from src.visualization import color_mask, overlay_mask, CITYSCAPES_CLASSES, CITYSCAPES_PALETTE
from src.metrics import SegmentationEvaluator

def color_gt_mask(gt_mask: np.ndarray) -> np.ndarray:
    """Colorize ground truth mask, rendering ignore_index 255 as black."""
    H, W = gt_mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in CITYSCAPES_PALETTE.items():
        colored[gt_mask == cid] = color
    # Ignore index 255 remains [0, 0, 0] (black / void)
    return colored

def main():
    out_dir = "outputs/plots/audit_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    segmentor = RoadVisionSegmentor(device="cpu")
    
    samples = [
        "frankfurt_000000_000294",
        "frankfurt_000000_000576",
        "frankfurt_000000_001016",
        "frankfurt_000000_001236",
        "frankfurt_000000_001751"
    ]
    
    overall_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=segmentor.class_names)
    sample_reports = []
    
    for stem in samples:
        img_path = f"data/samples/authentic/{stem}.png"
        gt_path = f"data/labels/authentic/{stem}_label.png"
        
        raw_img = Image.open(img_path).convert("RGB")
        img_np = np.array(raw_img)
        gt_np = np.array(Image.open(gt_path))
        
        # Predict
        res = segmentor.predict(img_np)
        pred_np = res["pred_mask"]
        
        # Single sample evaluation
        sample_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=segmentor.class_names)
        sample_evaluator.update(pred=pred_np, target=gt_np)
        overall_evaluator.update(pred=pred_np, target=gt_np)
        s_metrics = sample_evaluator.compute()
        
        # Colorize
        gt_colored = color_gt_mask(gt_np)
        pred_colored = color_mask(pred_np)
        overlay = overlay_mask(img_np, pred_colored, alpha=0.45)
        
        # Pixel-level error heatmap (ignoring 255)
        valid_mask = (gt_np != 255)
        error_map = np.zeros((gt_np.shape[0], gt_np.shape[1], 3), dtype=np.uint8)
        error_map[valid_mask & (pred_np != gt_np)] = [255, 50, 50]  # Red: misclassified
        error_map[valid_mask & (pred_np == gt_np)] = [50, 200, 50]  # Green: correctly classified
        # Ignore pixels stay black
        
        # Save individual assets
        Image.fromarray(img_np).save(f"{out_dir}/{stem}_original.png")
        Image.fromarray(gt_colored).save(f"{out_dir}/{stem}_ground_truth.png")
        Image.fromarray(pred_colored).save(f"{out_dir}/{stem}_predicted_mask.png")
        Image.fromarray(overlay).save(f"{out_dir}/{stem}_overlay.png")
        Image.fromarray(error_map).save(f"{out_dir}/{stem}_accuracy_map.png")
        
        # Create 4-panel side-by-side comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f"Original Input ({stem})", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title("Ground Truth (Cityscapes 19 Classes, Void=Black)", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")
        
        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title(f"SegFormer Prediction (Sample mIoU: {s_metrics['mIoU']*100:.1f}%, Acc: {s_metrics['pixel_accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Blended Semantic Overlay", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        sbs_path = f"{out_dir}/{stem}_side_by_side.png"
        plt.savefig(sbs_path, dpi=200)
        plt.close()
        
        rep = {
            "sample": stem,
            "mIoU": s_metrics["mIoU"],
            "pixel_accuracy": s_metrics["pixel_accuracy"],
            "per_class_iou": {k: v for k, v in s_metrics["per_class_iou"].items() if not np.isnan(v)}
        }
        sample_reports.append(rep)
        print(f"Processed {stem}: mIoU = {s_metrics['mIoU']*100:.2f}%, Pixel Accuracy = {s_metrics['pixel_accuracy']*100:.2f}%")
        
    overall_metrics = overall_evaluator.compute()
    print("\n=======================================================")
    print("OVERALL VERIFIED CLEAN VALIDATION METRICS (5 SAMPLES):")
    print(f"Mean IoU (mIoU): {overall_metrics['mIoU']*100:.2f}%")
    print(f"Pixel Accuracy: {overall_metrics['pixel_accuracy']*100:.2f}%")
    print("=======================================================")
    
    # Save audit metrics
    with open(f"{out_dir}/audit_clean_evaluation.json", "w") as f:
        json.dump({
            "overall": overall_metrics,
            "samples": sample_reports
        }, f, indent=2)

if __name__ == "__main__":
    main()

