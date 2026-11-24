# RoadVision System Architecture Reconnaissance

## 1. System Overview

RoadVision is a modular, autonomous-driving road-scene understanding system designed for semantic segmentation, real-time video stream processing, and environmental robustness benchmarking under adverse weather conditions.

### Architectural Tiers

1. **Model & Inference Tier (`src/inference.py`, `src/constants.py`, `src/config.py`)**
   - **Primary Model:** SegFormer-B0 (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, 3.7M parameters). Transformer-based encoder-decoder trained on 19 Cityscapes evaluation classes.
   - **Secondary Model:** DeepLabV3-MobileNetV3 (`torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`) for PASCAL VOC segmentation.
   - **Resolution Handling:** Configurable input size (`(512, 512)` for memory efficiency or `(1024, 1024)` for full-resolution inference) with bilinear upsampling of logits back to original image dimensions before argmax.
   - **Input Validation:** Strict dimension checks (`MAX_IMAGE_DIMENSION=4096`, `MAX_IMAGE_PIXELS=8,000,000`), dtype validation (`uint8`, RGB 3-channel).

2. **Video Processing Tier (`src/video_processor.py`)**
   - **Engine:** OpenCV `cv2.VideoCapture` + temporary MJPG intermediate AVI + FFmpeg H.264 transcoding via `imageio-ffmpeg`.
   - **Performance Optimization:** Frame-skipping parameter $k$ (`frame_skip >= 1`), reusing prior colored mask on skipped frames to maintain temporal continuity while reducing inference cost by factor $k$.
   - **HUD Telemetry:** Real-time on-frame telemetry banner rendering frame index, inference FPS, and rolling average model latency.
   - **Output Format:** Browser-compatible H.264 MP4 (`yuv420p`, `+faststart`).

3. **Environmental Robustness Benchmarking Tier (`src/corruptions.py`)**
   - **5 Adverse Weather Conditions:**
     - `low_light`: Non-linear intensity scaling and gamma shadow compression ($I' = \alpha (I/255)^\gamma \cdot 255$).
     - `fog`: Koschmieder atmospheric scattering model ($I_{fog} = I \cdot t(x) + L \cdot (1 - t(x))$) using vertical gradient depth approximation.
     - `rain`: Directional synthetic rain streaks generated via motion-blur kernels on Poisson-distributed drip centers, with contrast attenuation.
     - `gaussian_blur`: Optical defocus and lens blur via Gaussian kernel filtering with configurable $(\text{ksize}, \sigma)$.
     - `gaussian_noise`: Sensor electronic shot noise and high-ISO grain artifacts via zero-mean normal distribution with variance $\sigma$.
   - **Severity Levels:** 1 (mild) through 5 (severe), parameterized via YAML configuration.
   - **Determinism:** Seeded random number generation (`np.random.RandomState(seed)`).

4. **Evaluation & Metrics Tier (`src/metrics.py`)**
   - **Confusion Matrix Evaluator:** Vectorized $19 \times 19$ contingency matrix using flat bincount (`idx = target * 19 + pred`).
   - **Metrics Computed:**
     - True Positives ($TP_i = C_{ii}$)
     - False Positives ($FP_i = \sum_j C_{ji} - C_{ii}$)
     - False Negatives ($FN_i = \sum_j C_{ij} - C_{ii}$)
     - Per-class Intersection over Union: $\text{IoU}_i = \frac{TP_i}{TP_i + FP_i + FN_i}$
     - Mean IoU (Macro Average over present classes)
     - Overall Pixel Accuracy: $\text{Acc} = \frac{\sum_i TP_i}{\sum C}$
     - Absolute and relative degradation drops.
   - **Ignore Index:** Strict filtering of void/boundary pixels (`255`), ensuring unannotated pixels do not penalize false positives or false negatives.

5. **Visualization & Presentation Tier (`src/visualization.py`, `app.py`)**
   - **Colorization:** Vectorized NumPy palette indexing mapping class IDs $0..18$ to official Cityscapes RGB colors.
   - **Overlays:** Alpha-blended compositing ($I_{blend} = (1 - \alpha) I + \alpha M$).
   - **Interactive Dashboard:** Streamlit application with four functional sections:
     - Section A: Image Semantic Segmentation
     - Section B: Dashcam Video Inference
     - Section C: Robustness Laboratory
     - Section D: Quantitative Model Evaluation & Benchmarks

---

## 2. Directory Structure

```
E:\RoadVision
├── app.py                     # Streamlit multi-page interactive dashboard
├── configs/
│   └── default_config.yaml    # System hyperparameters, classes, palette, corruptions
├── data/
│   ├── samples/authentic/     # 5 official Cityscapes Frankfurt validation images
│   └── labels/authentic/      # 5 official Cityscapes labelTrainIds ground-truth masks
├── docs/
│   ├── audit_report.md        # Prior audit documentation
│   ├── methodology.md         # Mathematical formulations and pipeline design
│   ├── results.md             # Quantitative benchmarks and degradation tables
│   └── viva_preparation.md    # Technical defense Q&A
├── outputs/
│   ├── metrics/               # Saved benchmark JSON files
│   ├── plots/                 # Per-class IoU and robustness degradation curves
│   └── predictions/           # Inference outputs, overlays, and video clips
├── scripts/
│   ├── evaluate.py            # Quantitative dataset evaluation CLI
│   ├── reproduce_clean_eval.py# Clean baseline reproduction script
│   ├── robustness_test.py     # Robustness testing CLI
│   └── run_inference.py       # Single-image inference CLI
├── src/
│   ├── __init__.py            # Package root with lazy VideoProcessor export
│   ├── config.py              # Configuration singleton with environment variable overrides
│   ├── constants.py           # Single source of truth for classes and palettes
│   ├── corruptions.py         # Environmental corruption simulation engine
│   ├── dataset.py             # Dataset loader with paired image-mask verification
│   ├── inference.py           # Segmentation predictor (SegFormer / DeepLabV3)
│   ├── metrics.py             # Confusion matrix evaluator and degradation computer
│   ├── video_processor.py     # Dashcam video inference with H.264 transcoding
│   └── visualization.py       # Colorization, alpha overlay, and legend rendering
└── tests/
    ├── test_audit_regression.py # Regression tests for discovered audit defects
    ├── test_corruptions.py      # Unit tests for corruption monotonicity and determinism
    ├── test_inference.py        # Integration tests for model prediction and shapes
    ├── test_metrics.py          # Unit tests for confusion matrix math and ignore index
    └── test_video_processor.py  # Tests for video encoding, frame skipping, and MP4 format
```
