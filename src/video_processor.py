"""Video processing pipeline for RoadVision dashcam semantic segmentation."""

import os
import time
import cv2
import numpy as np
import uuid
import subprocess
import shutil
from typing import Dict, Optional, Callable
from src.inference import RoadVisionSegmentor
from src.visualization import overlay_mask, CITYSCAPES_PALETTE

# Try to import imageio-ffmpeg for bundled FFmpeg support
try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_AVAILABLE = True
except ImportError:
    IMAGEIO_FFMPEG_AVAILABLE = False


def processed_video_path(input_path: str, output_dir: str) -> str:
    """Build an MP4 output path using the source video's filename stem and a UUID to prevent race conditions."""
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    return os.path.join(output_dir, f"processed_{input_stem}_{unique_id}.mp4")


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

    @staticmethod
    def _draw_text_with_outline(
        image: np.ndarray,
        text: str,
        position: tuple,
        font: int,
        font_scale: float,
        text_color: tuple,
        outline_color: tuple,
        outline_thickness: int,
        text_thickness: int
    ) -> None:
        """Draw text with an outline (background) for better visibility.

        Args:
            image: Image to draw on (modified in place).
            text: Text to draw.
            position: (x, y) position for text.
            font: OpenCV font type.
            font_scale: Font scale factor.
            text_color: Text color (BGR).
            outline_color: Outline color (BGR).
            outline_thickness: Outline thickness.
            text_thickness: Text thickness.
        """
        cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(image, text, position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

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
            output_path: Path to destination MP4 video file (H.264 MP4).
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

        # Use a temporary file for initial encoding (MJPG is widely supported)
        temp_output_path = output_path.replace('.mp4', '_temp.avi')

        # Use MJPG codec which is widely supported in OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            output_path.replace('.mp4', '_temp.avi'),
            cv2.VideoWriter_fourcc(*"MJPG"),
            native_fps,
            (out_w, out_h)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Failed to initialize video writer for output: {output_path}. "
                "OpenCV MJPG writer is unavailable."
            )

        frame_idx = 0
        frames_inferred = 0
        frames_skipped = 0
        model_latencies = []
        e2e_latencies = []

        last_colored_mask = None
        last_hud_text = ""  # Cache last HUD text
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
                is_inference_frame = (frame_idx - 1) % self.frame_skip == 0

                if is_inference_frame:
                    # Model inference
                    m_start = time.perf_counter()
                    pred_res = self.segmentor.predict(rgb_frame)
                    m_latency = (time.perf_counter() - m_start) * 1000.0
                    model_latencies.append(m_latency)

                    last_colored_mask = pred_res["colored_mask"]
                    frames_inferred += 1

                    # Update HUD text on inference frames
                    current_fps = 1000.0 / m_latency if m_latency > 0 else 0.0
                    if model_latencies:
                        avg_model_latency = np.mean(model_latencies[-10:])
                    else:
                        avg_model_latency = 0.0
                    last_hud_text = f"RoadVision | Frame: {frame_idx} | FPS: {current_fps:.1f} | Model: {avg_model_latency:.1f}ms"
                else:
                    frames_skipped += 1

                # Reuse the last prediction on the current frame to preserve motion while skipping inference.
                if last_colored_mask is not None:
                    blended = overlay_mask(rgb_frame, last_colored_mask, alpha=self.overlay_alpha)
                else:
                    blended = rgb_frame

                # Convert back to BGR for OpenCV writer
                bgr_out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

                # Compute latency
                e2e_latency = (time.perf_counter() - e2e_start) * 1000.0
                e2e_latencies.append(e2e_latency)

                # Use cached HUD text
                if is_inference_frame:
                    self._draw_text_with_outline(
                        bgr_out, last_hud_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), (0, 0, 0), 3, 1
                    )
                elif last_hud_text:
                    self._draw_text_with_outline(
                        bgr_out, last_hud_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), (0, 0, 0), 3, 1
                    )

                writer.write(bgr_out)

                if progress_callback:
                    progress_callback(frame_idx, total_input_frames or max_frames, current_fps)

        finally:
            # Ensure both resources are released even if one fails
            try:
                cap.release()
            except Exception:
                pass
            try:
                writer.release()
            except Exception:
                pass

        # Transcode to H.264 MP4 using FFmpeg for browser compatibility (with fallback)
        final_output_path = output_path
        temp_avi_path = output_path.replace('.mp4', '_temp.avi')

        # Try to transcode using FFmpeg for H.264 (browser compatible)
        # If FFmpeg is not available, fall back to mp4v codec
        transcode_success = False
        try:
            self._transcode_to_h264_mp4(temp_avi_path, final_output_path)
            # Validate the output file
            self._validate_video_file(final_output_path)
            transcode_success = True
        except Exception as e:
            # FFmpeg not available or transcoding failed - fall back to mp4v codec
            # which is natively supported by OpenCV and has reasonable browser support
            if os.path.exists(temp_avi_path):
                try:
                    # Re-encode with mp4v codec which is natively supported
                    self._reencode_with_mp4v(temp_avi_path, final_output_path)
                    self._validate_video_file(final_output_path)
                    transcode_success = True
                except Exception as fallback_e:
                    # If fallback also fails, raise the original error
                    raise RuntimeError(f"Video transcoding failed: {e}. Fallback also failed: {fallback_e}")

        # Clean up temp file
        if os.path.exists(temp_avi_path):
            try:
                os.unlink(temp_avi_path)
            except Exception:
                pass

        if not transcode_success:
            raise RuntimeError("All video encoding attempts failed")

        total_time = time.perf_counter() - global_start
        overall_fps = frame_idx / total_time if total_time > 0 else 0.0

        return {
            "total_frames_read": frame_idx,
            "frames_inferred": frames_inferred,
            "frames_skipped": frames_skipped,
            "model_latency_mean_ms": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
            "end_to_end_latency_mean_ms": round(float(np.mean(e2e_latencies)), 2) if e2e_latencies else 0.0,
            "overall_fps": round(float(overall_fps), 1),
            "output_path": final_output_path,
            "resolution": (out_w, out_h),
            "total_processing_time_s": round(float(total_time), 2)
        }

    @staticmethod
    def _transcode_to_h264_mp4(input_path: str, output_path: str) -> None:
        """Transcode video to H.264 MP4 using FFmpeg for browser compatibility.

        Tries system FFmpeg first, then falls back to imageio-ffmpeg (bundled FFmpeg).

        Args:
            input_path: Path to input video file (e.g., AVI with MJPG).
            output_path: Path to output MP4 file with H.264 codec.
        """
        # Try system FFmpeg first
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path and IMAGEIO_FFMPEG_AVAILABLE:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or install imageio-ffmpeg package for H.264 video encoding.")

        # FFmpeg command for H.264 encoding with browser-compatible settings
        # -c:v libx264: Use H.264 codec
        # -preset fast: Fast encoding preset
        # -crf 23: Constant Rate Factor for quality (lower = better quality, 18-28 typical)
        # -pix_fmt yuv420p: Pixel format for broad browser compatibility
        # -movflags +faststart: Move moov atom to start for fast streaming
        # -vf "format=yuv420p": Ensure YUV420P pixel format for browser compatibility
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output file
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vf", "format=yuv420p",
            output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg transcoding failed: {result.stderr}")

    @staticmethod
    def _reencode_with_mp4v(input_path: str, output_path: str) -> None:
        """Re-encode video with H.264 codec using OpenCV's native support (if available).

        This is a fallback when FFmpeg is not available. Tries to use H.264 via OpenCV,
        falls back to mp4v (MPEG-4 Part 2) which has limited browser support.

        Args:
            input_path: Path to input video file.
            output_path: Path to output MP4 file.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try H.264 first (if OpenCV was built with H.264 support)
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # If H.264 not supported, fall back to mp4v
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to initialize video writer for output: {output_path}")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
        finally:
            cap.release()
            writer.release()

    @staticmethod
    def _validate_video_file(file_path: str) -> None:
        """Validate that the video file is a valid MP4 with H.264 codec.

        Args:
            file_path: Path to the video file to validate.
        """
        if not os.path.exists(file_path):
            raise RuntimeError(f"Output video file not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise RuntimeError(f"Output video file is empty: {file_path}")

        # Check MP4 signature (ftyp box)
        with open(file_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[4:8] != b"ftyp":
                raise RuntimeError(f"Output file is not a valid MP4: {file_path}")

        # Verify with OpenCV that the video can be decoded
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Output video cannot be decoded by OpenCV: {file_path}")

        # Check that we can read at least one frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Output video has no decodable frames: {file_path}")

        # Check frame properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check codec - prefer H.264 for browser compatibility
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        if fourcc > 0:
            codec_chars = ''.join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            # Note: We don't raise an error for non-H.264 codecs, just warn
            # because some browsers may still play them

        cap.release()

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Output video has invalid dimensions: {width}x{height}")
        if frame_count <= 0:
            raise RuntimeError(f"Output video has no frames: {file_path}")
        if fps <= 0:
            raise RuntimeError(f"Output video has invalid FPS: {fps}")