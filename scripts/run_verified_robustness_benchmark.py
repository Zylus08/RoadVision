"""Verified Robustness Benchmark evaluating models across 25 environmental conditions on authentic Cityscapes."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.inference import RoadVisionSegmentor
from src.corruptions import apply_corruption, CORRUPTION_REGISTRY
from src.metrics import SegmentationEvaluator, compute_degradation
from src.visualization import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, color_mask, overlay_mask

def main():
    print("================================================================")
    print("ROADVISION AUDIT: VERIFIED ADVERSE WEATHER ROBUSTNESS BENCHMARK")
    print("================================================================")
    
    samples = [
        "frankfurt_000000_000294",
        "frankfurt_000000_000576",
        "frankfurt_000000_001016",
        "frankfurt_000000_001236",
        "frankfurt_000000_001751"
    ]
    
    data_dir = "data/samples/authentic"
    label_dir = "data/labels/authentic"
    out_dir = "outputs/metrics"
    plot_dir = "outputs/plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    segmentor = RoadVisionSegmentor(device="cpu")
    
    # Load clean images and ground truth masks
    clean_images = []
    clean_preds = []
    ground_truths = []
    
    clean_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=segmentor.class_names)
    
    print("\n[Phase 1] Evaluating Clean Baseline on 5 Authentic Cityscapes Scenes...")
    for s in samples:
        img_np = np.array(Image.open(f"{data_dir}/{s}.png").convert("RGB"))
        gt_np = np.array(Image.open(f"{label_dir}/{s}_label.png"))
        
        clean_images.append(img_np)
        ground_truths.append(gt_np)
        
        res = segmentor.predict(img_np)
        pred = res["pred_mask"]
        clean_preds.append(pred)
        
        clean_evaluator.update(pred=pred, target=gt_np)
        
    clean_metrics = clean_evaluator.compute()
    print(f"Verified Clean Baseline mIoU: {clean_metrics['mIoU']*100:.2f}%")
    print(f"Verified Clean Pixel Accuracy: {clean_metrics['pixel_accuracy']*100:.2f}%")
    
    # Benchmark grid across 25 conditions
    print("\n[Phase 2] Running 25-Condition Environmental Robustness Matrix...")
    severities = [1, 2, 3, 4, 5]
    corruption_types = list(CORRUPTION_REGISTRY.keys())
    
    results_grid = {}
    
    for c_type in corruption_types:
        results_grid[c_type] = {}
        print(f"\nCorruption: '{c_type}'")
        
        for sev in severities:
            sev_evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255, class_names=segmentor.class_names)
            sev_latencies = []
            
            for i in range(len(samples)):
                img = clean_images[i]
                target = ground_truths[i]
                
                # Apply deterministic corruption with fixed seed
                corrupted_img = apply_corruption(img, corruption_type=c_type, severity=sev, seed=42 + i)
                
                # Predict
                res = segmentor.predict(corrupted_img)
                pred = res["pred_mask"]
                sev_latencies.append(res["latency_ms"])
                
                sev_evaluator.update(pred=pred, target=target)
                
            c_metrics = sev_evaluator.compute()
            deg = compute_degradation(clean_metrics, c_metrics)
            deg["latency_ms"] = round(float(np.mean(sev_latencies)), 2)
            deg["evaluated_samples"] = len(samples)
            deg["per_class_iou"] = c_metrics["per_class_iou"]
            
            results_grid[c_type][sev] = deg
            print(f"  Severity {sev}: mIoU={deg['corrupted_mIoU']*100:.2f}% | Pixel Acc={c_metrics['pixel_accuracy']*100:.2f}% | Abs Drop={deg['absolute_drop']*100:.2f}% (Rel: {deg['relative_drop_pct']:.1f}%) | Latency: {deg['latency_ms']:.1f}ms")
            
    # Save verified benchmark JSON
    benchmark_payload = {
        "status": "VERIFIED_AUTHENTIC",
        "description": "Robustness benchmark evaluated against authentic Cityscapes validation labels",
        "model": segmentor.model_name,
        "clean_baseline": clean_metrics,
        "corruptions": results_grid,
        "num_samples": len(samples),
        "seed": 42
    }
    
    bench_file = os.path.join(out_dir, "verified_robustness_benchmark.json")
    with open(bench_file, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\nSaved verified benchmark metrics to: {bench_file}")
    
    # Save clean baseline JSON
    clean_file = os.path.join(out_dir, "verified_clean_eval.json")
    with open(clean_file, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"Saved verified clean evaluation to: {clean_file}")
    
    # Phase 3: Generate verified plots
    print("\n[Phase 3] Generating Verified Evaluation Plots...")
    
    # 1. Verified Degradation Curves
    plt.figure(figsize=(10, 6))
    colors = {"low_light": "#eab308", "fog": "#94a3b8", "rain": "#3b82f6", "gaussian_blur": "#8b5cf6", "gaussian_noise": "#ef4444"}
    
    plt.axhline(clean_metrics["mIoU"] * 100.0, color="#10b981", linestyle="--", linewidth=2, label=f"Clean Baseline ({clean_metrics['mIoU']*100:.1f}%)")
    
    for c_type in corruption_types:
        mious = [results_grid[c_type][s]["corrupted_mIoU"] * 100.0 for s in severities]
        label = c_type.replace("_", " ").title()
        plt.plot(severities, mious, marker="o", linewidth=2.2, color=colors.get(c_type, "#000000"), label=label)
        
    plt.title("Verified Robustness Degradation Curves on Authentic Cityscapes", fontsize=14, fontweight="bold")
    plt.xlabel("Corruption Severity Level", fontsize=12)
    plt.ylabel("Mean IoU (mIoU %)", fontsize=12)
    plt.xticks(severities)
    plt.ylim(0, 50)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    curves_plot_path = os.path.join(plot_dir, "verified_robustness_degradation_curves.png")
    plt.savefig(curves_plot_path, dpi=300)
    plt.close()
    print(f"Saved verified degradation curves to: {curves_plot_path}")
    
    # 2. Maximum Severity Drop Bar Chart
    plt.figure(figsize=(9, 5))
    c_labels = [c.replace("_", " ").title() for c in corruption_types]
    drops = [results_grid[c][5]["absolute_drop"] * 100.0 for c in corruption_types]
    bar_colors = [colors.get(c, "#3b82f6") for c in corruption_types]
    
    bars = plt.bar(c_labels, drops, color=bar_colors, edgecolor="#1e293b")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"-{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.title("Verified Maximum Degradation at Severity 5 (Absolute mIoU Drop)", fontsize=13, fontweight="bold")
    plt.ylabel("mIoU Drop (%)", fontsize=11)
    plt.ylim(0, max(drops) + 5)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    bar_plot_path = os.path.join(plot_dir, "verified_robustness_max_severity_drop.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"Saved verified severity drop bar plot to: {bar_plot_path}")
    
    # 3. Verified Per-Class IoU Bar Chart
    per_class = {k: v for k, v in clean_metrics["per_class_iou"].items() if not np.isnan(v)}
    plt.figure(figsize=(12, 6))
    classes = list(per_class.keys())
    ious = [per_class[c] * 100.0 for c in classes]
    
    bars = plt.bar(classes, ious, color="#3b82f6", edgecolor="#1d4ed8")
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.axhline(clean_metrics["mIoU"] * 100.0, color="#ef4444", linestyle="--", label=f"mIoU: {clean_metrics['mIoU']*100:.1f}%")
    plt.title("Verified Per-Class IoU on Authentic Cityscapes Validation Data", fontsize=14, fontweight="bold")
    plt.xlabel("Semantic Class", fontsize=11)
    plt.ylabel("Intersection over Union (IoU %)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    pclass_path = os.path.join(plot_dir, "verified_per_class_iou.png")
    plt.savefig(pclass_path, dpi=300)
    plt.close()
    print(f"Saved verified per-class IoU plot to: {pclass_path}")

if __name__ == "__main__":
    main()

