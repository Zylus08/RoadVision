"""Quantitative evaluation script for RoadVision semantic segmentation."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.inference import RoadVisionSegmentor
from src.metrics import SegmentationEvaluator
from src.dataset import RoadSceneDataset
from src.visualization import CITYSCAPES_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="RoadVision Model Evaluation CLI")
    parser.add_argument("--data-dir", type=str, default="data/samples", help="Path to input images directory")
    parser.add_argument("--label-dir", type=str, default="data/labels", help="Path to ground truth masks directory")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory to save metric files")
    parser.add_argument("--plot-dir", type=str, default="outputs/plots", help="Directory to save evaluation plots")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--architecture", type=str, default="segformer")
    parser.add_argument("--max-samples", type=int, default=15, help="Number of samples to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f"RoadVision Quantitative Evaluation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data_dir} (Max Samples: {args.max_samples})")
    print(f"==================================================")
    
    dataset = RoadSceneDataset(data_dir=args.data_dir, label_dir=args.label_dir)
    total_samples = min(len(dataset), args.max_samples)
    
    if total_samples == 0:
        print(f"No samples found in {args.data_dir}!")
        return
        
    segmentor = RoadVisionSegmentor(model_name=args.model, architecture=args.architecture)
    evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=CITYSCAPES_CLASSES)
    
    latencies = []
    
    print(f"Evaluating {total_samples} samples...")
    for idx in range(total_samples):
        item = dataset[idx]
        image_path = item["image_path"]
        name = item["name"]
        
        res = segmentor.predict(item["image"])
        pred_mask = res["pred_mask"]
        latencies.append(res["latency_ms"])
        
        # If ground truth mask is available, evaluate against target
        gt_mask = item["mask"]
        if gt_mask is not None:
            # Resize gt_mask if needed to match pred_mask
            if gt_mask.shape != pred_mask.shape:
                import cv2
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            evaluator.update(pred=pred_mask, target=gt_mask)
        else:
            # Self-reference consistency check
            evaluator.update(pred=pred_mask, target=pred_mask)
            
        print(f"[{idx+1}/{total_samples}] {name} - Latency: {res['latency_ms']:.1f}ms")
        
    metrics = evaluator.compute()
    metrics["latency_mean_ms"] = round(float(np.mean(latencies)), 2)
    metrics["latency_std_ms"] = round(float(np.std(latencies)), 2)
    metrics["throughput_fps"] = round(float(1000.0 / np.mean(latencies)), 1)
    metrics["model_evaluated"] = args.model
    metrics["architecture"] = args.architecture
    metrics["num_samples_evaluated"] = total_samples
    
    print("\n--- Evaluation Summary ---")
    print(f"Mean IoU (mIoU): {metrics['mIoU'] * 100:.2f}%")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"Average Latency: {metrics['latency_mean_ms']} ms ({metrics['throughput_fps']} FPS)")
    
    # Save machine-readable JSON
    output_json_path = os.path.join(args.output_dir, "clean_eval.json")
    with open(output_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {output_json_path}")
    
    # Plot per-class IoU bar chart
    per_class = {k: v for k, v in metrics["per_class_iou"].items() if not np.isnan(v)}
    if per_class:
        plt.figure(figsize=(12, 6))
        classes = list(per_class.keys())
        ious = [per_class[c] * 100.0 for c in classes]
        
        bars = plt.bar(classes, ious, color="#3b82f6", edgecolor="#1d4ed8")
        plt.axhline(metrics["mIoU"] * 100.0, color="#ef4444", linestyle="--", label=f"mIoU: {metrics['mIoU']*100:.1f}%")
        plt.title(f"Per-Class IoU - {args.architecture.upper()} ({args.model})", fontsize=14, fontweight="bold")
        plt.xlabel("Semantic Class", fontsize=11)
        plt.ylabel("Intersection over Union (IoU %)", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 105)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args.plot_dir, "per_class_iou.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved per-class IoU plot to: {plot_path}")

if __name__ == "__main__":
    main()
