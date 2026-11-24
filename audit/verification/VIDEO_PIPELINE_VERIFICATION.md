# Video Pipeline Verification

## 1. Video Processing Architecture

The video processing pipeline (`src/video_processor.py`) is engineered for dashcam road video analysis with temporal continuity and browser playback compatibility.

### Processing Workflow
1. **Frame Extraction:** `cv2.VideoCapture` streams source frames in native resolution.
2. **Dimension Normalization:** Width and height are adjusted to even integers (`w -= w % 2`, `h -= h % 2`) as required by H.264 `yuv420p`.
3. **Temporal Frame Skipping ($k$-factor):**
   - For frame index $i$: if $(i - 1) \pmod k == 0$, full segmentation inference is performed.
   - For skipped frames: the latest generated colored mask is reused and composited over the current raw video frame.
   - **Benefit:** Preserves fluid motion in the underlying camera stream while cutting neural network inference compute by a factor of $k$.
4. **On-Frame HUD Telemetry:**
   - Outlined text banner rendered using `cv2.putText` with dual-pass (black stroke + yellow fill) for visibility against any background.
   - Displays: `RoadVision | Frame: {idx} | FPS: {fps} | Model: {latency}ms`.
5. **Two-Stage Encoding & Transcoding:**
   - Stage 1: OpenCV writes lossless/high-quality intermediate MJPG frames to a temporary directory (`TemporaryDirectory`).
   - Stage 2: `imageio-ffmpeg` invokes FFmpeg with `-c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart` to create a streaming-ready MP4.
   - Stage 3: Output MP4 verified for decodable H.264 packets before moving atomically to final path.

---

## 2. Test Verification Results

All 4 unit and integration tests in `tests/test_video_processor.py` pass:
- `test_process_video_outputs_decodable_mp4_and_preserves_frame_skip`: **PASS**
  - Verified 4-frame video with `frame_skip=2` resulted in 2 inferred frames and 2 skipped frames.
  - Verified output file starts with MP4 `ftyp` box.
  - Verified video decodes cleanly using `cv2.VideoCapture` with `avc1` / `h264` codec.
  - Verified frame brightness variance confirms motion preservation across skipped frames.
- `test_process_video_raises_when_writer_fails`: **PASS**
- `test_encoder_failure_keeps_prior_output_and_cleans_temporary_files`: **PASS**
- `test_processed_video_path_uses_mp4_extension`: **PASS**

---

## 3. Summary Verdict
- **Video Pipeline Status:** **PASS**
