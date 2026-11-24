# Root Cause Analysis: Critical Pipeline & Metric Findings

## 1. Executive Summary
This document records the definitive root cause analysis for the three major technical anomalies encountered across the lifecycle of the RoadVision project:
1. The historical 2.56% baseline clean mIoU anomaly.
2. The package-wide import failure triggered by optional video processing dependencies.
3. The mathematical discrepancy between image-level mean mIoU (46.79% / 58.52%) and dataset-level aggregate mIoU (38.98% / 55.11%).

---

## 2. Root Cause Analysis: The 2.56% Clean mIoU Failure

### The Symptom
Initial baseline evaluations on five road scenes reported an exceptionally poor mean Intersection-over-Union of 2.56%, despite visual inspections showing plausible segmentation masks for roads, cars, and buildings.

### Diagnostic Investigation
1. **Inspection of Ground-Truth Masks:**
   - The masks in `data/labels/` were loaded as NumPy arrays.
   - Distinct values detected: only `{0, 1, 3}` across the entire array.
   - All five mask files had identical MD5/SHA256 checksums, despite the five corresponding camera scenes being completely distinct geographical locations in Frankfurt.
   - The dummy masks came from an uncurated third-party HuggingFace repository (`Chris1/cityscapes_segmentation`) which had uploaded placeholder/dummy masks.
2. **Impact on Evaluation:**
   - Real Cityscapes images contain trees, sky, buildings, cars, poles, and traffic signs.
   - The model correctly predicted `10` (sky), `8` (vegetation), `2` (building), `13` (car).
   - In the dummy masks, sky and buildings were labeled as `0` (road) or `255` (void).
   - Every true prediction for building, sky, and vegetation was counted as a False Positive, and true road pixels were inundated with mismatches.
   - Result: 2.56% mIoU.

### Resolution & Proof
Authentic Frankfurt validation scenes from `tackhwa/cityscapes` were integrated:
- Real masks exhibit 10 to 16 distinct semantic classes per scene.
- Real masks have unique SHA256 checksums.
- Evaluated against authentic annotations, the SegFormer-B0 model immediately achieves **38.98% aggregate mIoU** (at 512x512 resolution) and **55.11% aggregate mIoU** (at native 1024x1024 resolution) with **94.47% - 96.62% pixel accuracy**.

---

## 3. Root Cause Analysis: Eager Import Dependency Cascade

### The Symptom
Running `pytest` or importing `src.metrics` resulted in:
`ModuleNotFoundError: No module named 'imageio_ffmpeg'`

### Diagnostic Investigation
1. `src/__init__.py` exposed public APIs by importing:
   ```python
   from src.video_processor import VideoProcessor, processed_video_path
   ```
2. `src/video_processor.py` had a top-level import:
   ```python
   import imageio_ffmpeg
   ```
3. Even when running pure arithmetic tests like `test_metrics.py` (which only imports `src.metrics`), Python executed `src/__init__.py`, which failed on the missing `imageio_ffmpeg`.

### Resolution & Proof
1. `imageio-ffmpeg` and `imageio` installed in the environment.
2. `src/__init__.py` refactored to make the video processor import lazy with graceful `try/except ImportError` fallback.
3. All 23 tests now execute and pass cleanly.

---

## 4. Root Cause Analysis: Mathematical Averaging Conventions & Input Resolutions

### The Symptom
Different runs and scripts reported 38.98%, 46.79%, 55.11%, or 58.52% clean baseline mIoU.

### Mathematical Explanation
Two independent dimensions dictate the reported metric:
1. **Input Resolution:**
   - SegFormer-B0 was trained at $1024 \times 1024$.
   - When downscaled to $(512, 512)$ before forward pass (to conserve CPU RAM and maintain high throughput), small objects (poles, traffic signs, riders) experience edge distortion, producing **38.98% aggregate mIoU**.
   - When evaluated at full native $(1024, 1024)$ resolution, the model achieves **55.11% aggregate mIoU**.
2. **Averaging Convention:**
   - **Convention A (Dataset-Level Aggregate):** Accumulate the $19 \times 19$ confusion matrix across all 5 scenes first, then compute per-class IoU from $\frac{\sum TP_i}{\sum (TP_i + FP_i + FN_i)}$ and average over valid classes.
     - At $(512, 512)$: **38.98% mIoU**, **94.47% pixel accuracy**.
     - At $(1024, 1024)$: **55.11% mIoU**, **96.62% pixel accuracy**.
   - **Convention B (Image-Level Macro Mean):** Compute mIoU on each individual image, then take the arithmetic mean across the 5 images.
     - At $(512, 512)$: **46.79% mIoU**.
     - At $(1024, 1024)$: **58.52% mIoU**.
   - **Conclusion:** Both conventions are mathematically legitimate and fully reproducible. RoadVision documents Convention A (38.98% / 55.11%) as the primary dataset benchmark per official Cityscapes convention.
