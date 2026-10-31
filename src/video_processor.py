"""Video processing pipeline for RoadVision dashcam semantic segmentation."""

import os
import time
import cv2
import numpy as np
from typing import Dict, Optional, Callable
from src.inference import RoadVisionSegmentor
from src.visualization import overlay_mask, CITYSCAPES_PALETTE

class VideoProcessor:
    """Processes dashcam video streams with semantic segmentation overlays and HUD statistics."""
    
    def __init__(
        self,
        segmentor: RoadVisionSegmentor,
        overlay_alpha: float = 0.45,
        frame_skip: int = 1,
        target_resolution: Optional[tuple] = None # e.g. (640, 360) or None for native
    ):
        """Initialize VideoProcessor.
        
        Args:
            segmentor: Instance of RoadVisionSegmentor.
            overlay_alpha: Blending weight for segmentation mask.
            frame_skip: Process every k-th frame (1 = process all frames).
            target_resolution: (width, height) to resize frames for faster processing, or None.
        """
        self.segmentor = segmentor
        self.overlay_alpha = overlay_alpha
        self.frame_skip = max(1, frame_skip)
        self.target_resolution = target_resolution
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict:
        """Process video file, save annotated output video, and return telemetry statistics.
        
        Args:
            input_path: Path to source video file.
            output_path: Path to destination video file (e.g. .mp4).
            max_frames: Optional maximum number of frames to process.
            progress_callback: Optional callback func(current_frame, total_frames, current_fps).
            
        Returns:
            Dictionary containing processing statistics:
                'total_frames_read', 'frames_inferred', 'frames_skipped',
                'model_latency_mean_ms', 'end_to_end_latency_mean_ms',
                'overall_fps', 'output_path', 'resolution'.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {input_path}")
            
        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_w, out_h = self.target_resolution if self.target_resolution else (width, height)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Choose robust cross-platform codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, native_fps, (out_w, out_h))
        
        if not writer.isOpened():
            # Fallback to AVI / XVID if MP4V fails
            alt_path = output_path.rsplit(".", 1)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(alt_path, fourcc, native_fps, (out_w, out_h))
            output_path = alt_path
            
        frame_idx = 0
        frames_inferred = 0
        frames_skipped = 0
        model_latencies = []
        e2e_latencies = []
        
        last_colored_mask = None
        global_start = time.perf_counter()
        
        try:
            while cap.isOpened():
                if max_frames and frame_idx >= max_frames:
                    break
                    
                e2e_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                
                # Resize if target resolution specified
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    
                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check frame skip
                if (frame_idx - 1) % self.frame_skip == 0:
                    # Model inference
                    m_start = time.perf_counter()
                    pred_res = self.segmentor.predict(rgb_frame)
                    m_latency = (time.perf_counter() - m_start) * 1000.0
                    model_latencies.append(m_latency)
                    
                    last_colored_mask = pred_res["colored_mask"]
                    frames_inferred += 1
                else:
                    frames_skipped += 1
                    
                # Generate blended frame
                if last_colored_mask is not None:
                    blended = overlay_mask(rgb_frame, last_colored_mask, alpha=self.overlay_alpha)
                else:
                    blended = rgb_frame
                    
                # Convert back to BGR for OpenCV writer
                bgr_out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                
                # Compute latency
                e2e_latency = (time.perf_counter() - e2e_start) * 1000.0
                e2e_latencies.append(e2e_latency)
                current_fps = 1000.0 / e2e_latency if e2e_latency > 0 else 0.0
                
                # Draw telemetry HUD overlay on frame
                hud_text = f"RoadVision | Frame: {frame_idx} | FPS: {current_fps:.1f} | Model: {np.mean(model_latencies[-10:]):.1f}ms"
                cv2.putText(bgr_out, hud_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr_out, hud_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
                
                writer.write(bgr_out)
                
                if progress_callback:
                    progress_callback(frame_idx, total_input_frames or max_frames, current_fps)
                    
        finally:
            cap.release()
            writer.release()
            
        total_time = time.perf_counter() - global_start
        overall_fps = frame_idx / total_time if total_time > 0 else 0.0
        
        return {
            "total_frames_read": frame_idx,
            "frames_inferred": frames_inferred,
            "frames_skipped": frames_skipped,
            "model_latency_mean_ms": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
            "end_to_end_latency_mean_ms": round(float(np.mean(e2e_latencies)), 2) if e2e_latencies else 0.0,
            "overall_fps": round(float(overall_fps), 1),
            "output_path": output_path,
            "resolution": (out_w, out_h),
            "total_processing_time_s": round(float(total_time), 2)
        }
