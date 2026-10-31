"""Command-line script for RoadVision image and video segmentation."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import cv2
from PIL import Image
from src.inference import RoadVisionSegmentor
from src.video_processor import VideoProcessor
from src.visualization import create_legend_image

def parse_args():
    parser = argparse.ArgumentParser(description="RoadVision Semantic Segmentation CLI")
    parser.add_argument("--input", type=str, default="data/samples/road_scene_001.png", help="Path to input image or video")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", help="Model checkpoint identifier")
    parser.add_argument("--architecture", type=str, default="segformer", choices=["segformer", "deeplabv3"], help="Model architecture")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, or cuda")
    parser.add_argument("--frame-skip", type=int, default=1, help="Video frame skip factor")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f"RoadVision Inference Engine")
    print(f"Model: {args.model} ({args.architecture})")
    print(f"Input: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print(f"==================================================")
    
    segmentor = RoadVisionSegmentor(
        model_name=args.model,
        architecture=args.architecture,
        device=args.device
    )
    
    input_path = args.input
    stem = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1].lower()
    
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    
    if ext in video_exts:
        print(f"Processing video: {input_path}")
        output_video = os.path.join(args.output_dir, f"{stem}_segmented.mp4")
        processor = VideoProcessor(segmentor, frame_skip=args.frame_skip)
        
        def on_progress(curr, total, fps):
            if curr % 10 == 0 or curr == total:
                print(f"Frame {curr}/{total} - Throughput: {fps:.1f} FPS")
                
        stats = processor.process_video(input_path, output_video, progress_callback=on_progress)
        print("\n--- Video Processing Complete ---")
        print(json.dumps(stats, indent=2))
        
        stats_path = os.path.join(args.output_dir, f"{stem}_video_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to: {stats_path}")
        print(f"Saved segmented video to: {output_video}")
        
    else:
        print(f"Processing image: {input_path}")
        res = segmentor.predict(input_path)
        
        # Save output artifacts
        mask_path = os.path.join(args.output_dir, f"{stem}_mask.png")
        overlay_path = os.path.join(args.output_dir, f"{stem}_overlay.png")
        legend_path = os.path.join(args.output_dir, f"{stem}_legend.png")
        meta_path = os.path.join(args.output_dir, f"{stem}_telemetry.json")
        
        Image.fromarray(res["colored_mask"]).save(mask_path)
        Image.fromarray(res["overlay"]).save(overlay_path)
        
        legend_img = create_legend_image(res["present_classes"])
        Image.fromarray(legend_img).save(legend_path)
        
        telemetry = {
            "image": input_path,
            "architecture": args.architecture,
            "model": args.model,
            "device": res["device"],
            "resolution": res["orig_shape"],
            "latency_ms": res["latency_ms"],
            "throughput_fps": res["fps"],
            "class_proportions": res["class_proportions"]
        }
        with open(meta_path, "w") as f:
            json.dump(telemetry, f, indent=2)
            
        print("\n--- Image Inference Complete ---")
        print(f"Resolution: {res['orig_shape']}")
        print(f"Latency: {res['latency_ms']} ms ({res['fps']} FPS)")
        print(f"Detected classes: {list(res['class_proportions'].keys())}")
        print(f"Saved mask: {mask_path}")
        print(f"Saved overlay: {overlay_path}")
        print(f"Saved telemetry: {meta_path}")

if __name__ == "__main__":
    main()
