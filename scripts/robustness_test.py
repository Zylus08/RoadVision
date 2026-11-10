"""Robustness evaluation benchmark script testing models across environmental corruptions."""

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
from src.corruptions import apply_corruption, CORRUPTION_REGISTRY
from src.metrics import SegmentationEvaluator, compute_degradation
from src.dataset import RoadSceneDataset
from src.visualization import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, color_mask, overlay_mask


def parse_args():
    parser = argparse.ArgumentParser(description="RoadVision Environmental Robustness Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/samples", help="Path to input images directory")
    parser.add_argument("--label-dir", type=str, default="data/labels", help="Path to ground truth masks directory")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory to save metric files")
    parser.add_argument("--plot-dir", type=str, default="outputs/plots", help="Directory to save evaluation plots")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--architecture", type=str, default="segformer")
    parser.add_argument("--samples", type=int, default=10, help="Number of road scenes to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print("==================================================")
    print("RoadVision Environmental Robustness Benchmark")
    print(f"Model: {args.model} ({args.architecture})")
    print(f"Corruptions: {list(CORRUPTION_REGISTRY.keys())}")
    print(f"Severities: [1, 2, 3, 4, 5]")
    print(f"Samples: {args.samples} | Seed: {args.seed}")
    print("==================================================")

    dataset = RoadSceneDataset(data_dir=args.data_dir, label_dir=args.label_dir)
    num_eval = min(len(dataset), args.samples)
    if num_eval == 0:
        print(f"No samples found in {args.data_dir}!")
        return

    segmentor = RoadVisionSegmentor(model_name=args.model, architecture=args.architecture)

    # Pre-compute clean baseline predictions and targets
    clean_images = []
    clean_preds = []
    ground_truths = []
    names = []

    clean_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=CITYSCAPES_CLASSES)

    print(f"\n[Phase 1] Computing Clean Baseline Predictions ({num_eval} samples)...")
    has_ground_truth = False
    for i in range(num_eval):
        item = dataset[i]
        img_np = np.array(item["image"])
        clean_images.append(img_np)
        names.append(item["name"])

        res = segmentor.predict(img_np)
        pred_mask = res["pred_mask"]
        clean_preds.append(pred_mask)

        gt = item["mask"]
        if gt is not None:
            if gt.shape != pred_mask.shape:
                import cv2
                gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            ground_truths.append(gt)
            clean_evaluator.update(pred=pred_mask, target=gt)
            has_ground_truth = True
        else:
            # No ground truth available - skip evaluation (self-reference gives 100% mIoU by definition)
            ground_truths.append(None)  # Use None to indicate no GT
            has_ground_truth = False

    clean_metrics = clean_evaluator.compute()
    if has_ground_truth:
        print(f"Clean Baseline mIoU: {clean_metrics['mIoU'] * 100:.2f}% | Pixel Accuracy: {clean_metrics['pixel_accuracy'] * 100:.2f}%")
    else:
        print("No ground truth available - skipping mIoU and Pixel Accuracy (self-reference would be 100%)")

    # Phase 2: Corrupted benchmark matrix (5 types x 5 severities)
    print("\n[Phase 2] Running Robustness Grid Evaluation (5 types x 5 severities = 25 conditions)...")
    results_grid = {}

    severities = [1, 2, 3, 4, 5]
    corruption_types = list(CORRUPTION_REGISTRY.keys())

    # Store for visualization
    sample_visuals = {}

    for c_type in corruption_types:
        results_grid[c_type] = {}
        print(f"\nEvaluating Corruption: '{c_type}'")

        for sev in severities:
            sev_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=CITYSCAPES_CLASSES)
            sev_latencies = []

            for i in range(num_eval):
                img = clean_images[i]
                target = ground_truths[i]

                # Apply deterministic corruption
                corrupted_img = apply_corruption(img, corruption_type=c_type, severity=sev, seed=args.seed + i)

                # Save visual sample for demo (sample 0, severity 3)
                if i == 0 and sev == 3 and c_type not in sample_visuals:
                    sample_visuals[c_type] = {
                        "clean_img": img,
                        "corrupt_img": corrupted_img,
                        "clean_pred": clean_preds[0]
                    }

                # Predict
                res = segmentor.predict(corrupted_img)
                pred = res["pred_mask"]
                sev_latencies.append(res["latency_ms"])

                # Only update evaluator if ground truth is available
                if target is not None:
                    sev_evaluator.update(pred=pred, target=target)

                if i == 0 and sev == 3:
                    sample_visuals[c_type]["corrupt_pred"] = pred

            c_metrics = sev_evaluator.compute()
            if has_ground_truth:
                deg = compute_degradation(clean_metrics, c_metrics)
                deg["latency_ms"] = round(float(np.mean(sev_latencies)), 2)
                deg["evaluated_samples"] = num_eval
                deg["per_class_iou"] = c_metrics["per_class_iou"]

                results_grid[c_type][sev] = deg
                print(f"  Severity {sev}: mIoU={deg['corrupted_mIoU']*100:.2f}% (Drop: {deg['absolute_drop']*100:.2f}%, Rel: {deg['relative_drop_pct']:.1f}%) | Latency: {deg['latency_ms']:.1f}ms")
            else:
                # No ground truth - only store latency
                deg = {
                    "latency_ms": round(float(np.mean(sev_latencies)), 2),
                    "evaluated_samples": num_eval,
                    "corrupted_mIoU": None,
                    "absolute_drop": None,
                    "relative_drop_pct": None,
                    "clean_mIoU": None,
                    "clean_pixel_accuracy": None,
                    "corrupted_pixel_accuracy": None,
                    "pixel_accuracy_drop": None,
                    "per_class_iou": c_metrics["per_class_iou"]
                }
                results_grid[c_type][sev] = deg
                print(f"  Severity {sev}: Latency: {deg['latency_ms']:.1f}ms (no ground truth for mIoU)")

    # Save benchmark JSON
    benchmark_data = {
        "model": args.model,
        "architecture": args.architecture,
        "clean_baseline": clean_metrics,
        "corruptions": results_grid,
        "num_samples": num_eval,
        "seed": args.seed
    }

    benchmark_json_path = os.path.join(args.output_dir, "robustness_benchmark.json")
    with open(benchmark_json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved benchmark metrics to: {benchmark_json_path}")

    # Phase 3: Generate visual plots
    print("\n[Phase 3] Generating Evaluation Plots...")

    # 1. Degradation Curves Plot (only if ground truth available)
    if has_ground_truth:
        plt.figure(figsize=(10, 6))
        colors = {"low_light": "#eab308", "fog": "#94a3b8", "rain": "#3b82f6", "gaussian_blur": "#8b5cf6", "gaussian_noise": "#ef4444"}

        plt.axhline(clean_metrics["mIoU"] * 100.0, color="#10b981", linestyle="--", linewidth=2, label=f"Clean Baseline ({clean_metrics['mIoU']*100:.1f}%)")

        for c_type in corruption_types:
            mious = [results_grid[c_type][s]["corrupted_mIoU"] * 100.0 for s in severities]
            label = c_type.replace("_", " ").title()
            plt.plot(severities, mious, marker="o", linewidth=2.2, color=colors.get(c_type, "#000000"), label=label)

        plt.title(f"Segmentation Robustness Degradation Curves - {args.architecture.upper()}", fontsize=14, fontweight="bold")
        plt.xlabel("Corruption Severity Level", fontsize=12)
        plt.ylabel("mIoU (%)", fontsize=12)
        plt.xticks(severities)
        plt.ylim(0, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        curves_plot_path = os.path.join(args.plot_dir, "robustness_degradation_curves.png")
        plt.savefig(curves_plot_path, dpi=300)
        plt.close()
        print(f"Saved degradation curves to: {curves_plot_path}")

        # 2. Maximum Severity Comparison Bar Chart (Severity 5)
        plt.figure(figsize=(9, 5))
        c_labels = [c.replace("_", " ").title() for c in corruption_types]
        drops = [results_grid[c][5]["absolute_drop"] * 100.0 for c in corruption_types]
        bar_colors = [colors.get(c, "#3b82f6") for c in corruption_types]

        bars = plt.bar(c_labels, drops, color=bar_colors, edgecolor="#1e293b")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"-{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.title("Maximum Performance Degradation at Severity 5 (Absolute mIoU Drop)", fontsize=13, fontweight="bold")
        plt.ylabel("mIoU Drop (%)", fontsize=11)
        plt.ylim(0, max(drops) + 15)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        bar_plot_path = os.path.join(args.plot_dir, "robustness_max_severity_drop.png")
        plt.savefig(bar_plot_path, dpi=300)
        plt.close()
        print(f"Saved severity drop bar plot to: {bar_plot_path}")
    else:
        print("Skipping mIoU degradation curves and bar chart (no ground truth available)")

    # 3. Visual Grid of Corruptions
    fig, axes = plt.subplots(len(corruption_types), 4, figsize=(16, 3.5 * len(corruption_types)))
    for row, c_type in enumerate(corruption_types):
        v = sample_visuals[c_type]
        clean_cmask = color_mask(v["clean_pred"])
        corrupt_cmask = color_mask(v["corrupt_pred"])

        axes[row, 0].imshow(v["clean_img"])
        axes[row, 0].set_title(f"Clean Original", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(clean_cmask)
        axes[row, 1].set_title(f"Clean Prediction", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(v["corrupt_img"])
        axes[row, 2].set_title(f"{c_type.replace('_', ' ').title()} (Sev 3)", fontsize=11)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(corrupt_cmask)
        axes[row, 3].set_title(f"Corrupted Prediction", fontsize=11)
        axes[row, 3].axis("off")

    plt.tight_layout()
    grid_plot_path = os.path.join(args.plot_dir, "robustness_visual_comparison_grid.png")
    plt.savefig(grid_plot_path, dpi=200)
    plt.close()
    print(f"Saved visual comparison grid to: {grid_plot_path}")
    print("\nRobustness benchmark successfully completed!")


if __name__ == "__main__":
    main()