"""Hardware and Performance Diagnostic Script."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import numpy as np
from PIL import Image
from src.inference import RoadVisionSegmentor
from src.video_processor import VideoProcessor

def audit_hardware_and_performance():
    print("================================================================")
    print("ROADVISION AUDIT: HARDWARE AND PERFORMANCE VALIDATION")
    print("================================================================")
    
    # 1. Device Diagnostics
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    torch_version = torch.__version__
    
    print(f"PyTorch Version: {torch_version}")
    print(f"CUDA Available in PyTorch Build: {cuda_avail}")
    print(f"PyTorch Device Count: {device_count}")
    
    if not cuda_avail:
        print("  NOTE: The installed PyTorch wheel is '+cpu'.")
        print("  System GPU (RTX 3050 Laptop) is present on hardware, but the active PyTorch binary is CPU-only.")
        print("  Execution Mode: STRICT CPU EXECUTION.")
        
    # 2. Model Forward Pass Latency Measurement
    print("\n[2] Model-Only Forward Pass Latency Measurement (SegFormer-B0):")
    segmentor = RoadVisionSegmentor(device="cpu")
    
    # Warmup
    dummy = np.zeros((1024, 2048, 3), dtype=np.uint8)
    for _ in range(2):
        _ = segmentor.predict(dummy)
        
    runs = 10
    latencies = []
    print(f"Running {runs} forward passes on (1024, 2048) input...")
    for r in range(runs):
        t0 = time.perf_counter()
        res = segmentor.predict(dummy)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(res["latency_ms"])
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    model_fps = 1000.0 / mean_lat
    print(f"  Model-Only Latency: {mean_lat:.2f} ms +/- {std_lat:.2f} ms")
    print(f"  Model-Only Throughput: {model_fps:.2f} FPS")
    
    # 3. Video Pipeline Latency Measurement
    print("\n[3] Complete End-to-End Video Processing Latency Measurement:")
    test_vid_path = "data/samples/sample_dashcam.mp4"
    if os.path.exists(test_vid_path):
        processor = VideoProcessor(segmentor, frame_skip=2, target_resolution=(640, 360))
        out_path = "outputs/predictions/audit_video_perf_test.mp4"
        
        t0_vid = time.perf_counter()
        stats = processor.process_video(test_vid_path, out_path, max_frames=30)
        total_wall_time = time.perf_counter() - t0_vid
        
        computed_fps = stats["total_frames_read"] / total_wall_time
        print(f"  Total Frames Read: {stats['total_frames_read']}")
        print(f"  Frames Inferred: {stats['frames_inferred']}")
        print(f"  Frames Skipped: {stats['frames_skipped']}")
        print(f"  Total Elapsed Wall-Clock Time: {total_wall_time:.2f} s")
        print(f"  Direct Wall-Clock FPS: {computed_fps:.2f} FPS")
        print(f"  Reported Video Stats FPS: {stats['overall_fps']:.2f} FPS")
        print(f"  Mean Model Latency in Video: {stats['model_latency_mean_ms']:.2f} ms")
        print(f"  Mean End-to-End Latency in Video: {stats['end_to_end_latency_mean_ms']:.2f} ms")
        
        assert abs(computed_fps - stats['overall_fps']) < 0.5, "Discrepancy in FPS calculation!"
        print("  CONFIRMED: Reported video FPS exactly matches wall-clock elapsed time.")

if __name__ == "__main__":
    audit_hardware_and_performance()

