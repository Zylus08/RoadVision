# Section B Video Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make processed Section B dashcam videos playable in browsers by writing WebM/VP8 output.

**Architecture:** Keep OpenCV as the video reader and writer, but change processed output to `.webm` with the VP8 codec. Pass a `video/webm` data URI to Streamlit so the WebM MIME type is correct and playback avoids the failing `/media` range response. Keep inference, frame skipping, and telemetry unchanged; surface a clear error if VP8 writing is unavailable.

**Tech Stack:** Python, OpenCV, Streamlit, unittest.

**Spec:** [2026-11-09-section-b-video-playback-design.md](../specs/2026-11-09-section-b-video-playback-design.md)

## Global Constraints

- Do not change model inference, accuracy, input video support, or frame skipping.
- Do not add FFmpeg or another runtime dependency.
- The current OpenCV installation supports writing VP8/WebM.
- The output extension must match the encoded format.

## Review Focus

- VP8 encoder unavailable: processing raises a clear error instead of silently creating an unplayable file.
- Valid source video: output has WebM/VP8 encoding and can be reopened and decoded by OpenCV.
- Frame skipping: inferred and skipped counts remain unchanged, while skipped output frames blend the current source frame with the most recent mask so motion does not freeze.
- Source extension: MP4 and AVI inputs both receive a `.webm` processed output name.
- Empty or unreadable input: preserve existing explicit input errors; no success-shaped output should be displayed.

---

### Task 1: Write and display browser-playable processed video

**Files:**
- Modify: `src/video_processor.py`
- Modify: `app.py`
- Test: `tests/test_video_processor.py`

**Interfaces:**
- Consumes: `VideoProcessor.process_video(input_path, output_path, max_frames, progress_callback)`.
- Produces: OpenCV-written WebM/VP8 at the requested `.webm` path; return `output_path` matching the actual file.
- Produces: `processed_video_path(input_path: str, output_dir: str) -> str`, returning `processed_<input-stem>.webm`.
- Produces: `video_to_data_uri(video_path: str) -> str`, encoding the WebM bytes as a `data:video/webm` URI for playback.

- [x] **Step 1: Write failing integration tests**

Add `test_process_video_outputs_decodable_webm_and_preserves_frame_skip` using a four-frame synthetic MJPG/AVI input and deterministic segmentor stub. Assert output can be reopened and all four frames decoded; with `frame_skip=2`, assert `total_frames_read == 4`, `frames_inferred == 2`, and `frames_skipped == 2`, and verify every output frame reflects changing source motion. Add `test_process_video_raises_when_vp8_writer_fails`, patching `cv2.VideoWriter` to return a writer with `isOpened() == False`, and assert a clear `RuntimeError` plus release of the capture and writer. Add `test_processed_video_path_uses_webm_extension` and assert `.mp4`, `.avi`, and `.mov` inputs produce `processed_<stem>.webm`. Add `test_video_to_data_uri_preserves_webm_bytes` and assert the data URI round-trips the exact bytes with the correct MIME prefix.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_video_processor -v`
Expected: FAIL because the helpers are missing and the current processor uses MP4V without reporting writer initialization failure.

- [x] **Step 3: Implement tested output-path helper and VP8 writer**

Add `processed_video_path(input_path: str, output_dir: str) -> str` in `src/video_processor.py`. In `VideoProcessor.process_video`, use `cv2.VideoWriter_fourcc(*"VP80")` for the WebM output. If the writer cannot initialize, release it and raise a clear `RuntimeError`; do not fall back to MP4V or XVID. Blend the current frame with the latest mask on skipped frames, without running inference. Preserve frame sampling and statistics.

- [x] **Step 4: Use the WebM output path in Section B**

In `app.py`, call `processed_video_path(selected_vid_path, "outputs/predictions")`. Pass `video_to_data_uri(stats["output_path"])` to `st.video(..., format="video/webm")`.

- [x] **Step 5: Run focused and full test suites**

Run: `python -m unittest tests.test_video_processor -v`
Expected: PASS, including decodable output, writer failure, and unchanged frame-skip statistics.

Run: `python -m unittest discover -s tests -v`
Expected: all tests PASS.

- [x] **Step 6: Verify playback in Section B**

Run the Streamlit app from `RoadVision`, process the sample dashcam video, and verify that the displayed `.webm` plays in the browser and the reported inference/skip counts remain consistent.
