# Git Commit Change Inventory & History Analysis

## 1. Commit History Overview

| Commit SHA | Author | Date | Subject |
|------------|--------|------|---------|
| `81be5e2` | Abhay Kumar | 2026-11-21 15:08:55 +0530 | Merge pull request #2 from Zylus08/fix/video-playback-ui |
| `54ef5d7` | Abhay-coding | 2026-11-21 14:34:24 +0530 | fix: stabilize dashcam playback, reproduce benchmarks, and clean up tracked artifacts |
| `27baf17` | Saksham Mishra | 2026-11-18 11:19:12 +0530 | Merge pull request #1 from Zylus08/fix/video-playback-ui |
| `3807a2c` | Abhay-coding | 2026-11-10 17:02:16 +0530 | Fix video playback (H.264 via imageio-ffmpeg), UI rendering, and docstring issues |
| `8353d51` | Saksham Mishra | Prior | feat: initialize RoadVision project structure with core pipeline and documentation |

---

## 2. Detailed Change Analysis by Commit

### Commit `3807a2c` & PR #1 (`27baf17`)
- **Key Additions:**
  - Introduced `imageio-ffmpeg` dependency for transcoding dashcam video outputs to H.264 MP4.
  - Added video processing HUD with outline font rendering (`_draw_text_with_outline`).
  - Added frame skipping logic to `src/video_processor.py`.
  - Added Streamlit UI base64 data URI helper (`prepare_image_for_display`) in `app.py` to prevent broken browser image links.
  - Added unit test suite for video processor in `tests/test_video_processor.py`.
- **Issues Introduced:**
  - Tracked several untracked command artifact files with invalid filenames (`0`, `8`, `Dict[str`, `MJPG,-`, `np.ndarray`, `str`).
  - Eager top-level import of `imageio_ffmpeg` inside `src/__init__.py` which broke package imports when `imageio-ffmpeg` was not installed.

### Commit `54ef5d7` & PR #2 (`81be5e2`)
- **Key Remediations:**
  - Removed accidental untracked root-level files from the Git index.
  - Updated `.gitignore` with comprehensive exclusions for artifacts, caches, and prediction dumps.
  - Refactored `src/video_processor.py` to enforce `.mp4` destination extensions, check decodable frames via OpenCV, and use temporary directories during transcoding.
  - Updated `requirements.txt` to include `imageio-ffmpeg>=0.6.0` and `streamlit>=1.50.0`.
  - Harmonized docstrings and typing across `src/` modules.

---

## 3. Preservation of Developer Contributions
All architectural innovations from both developers are preserved:
- Saksham Mishra: Core SegFormer pipeline, Cityscapes authentic data curation, confusion-matrix math, 5-condition robustness corruptions.
- Abhay Kumar: H.264 video transcoding pipeline, outline HUD telemetry rendering, base64 data URI UI display, unit test coverage for video processor.
