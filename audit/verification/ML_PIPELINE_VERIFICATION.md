# ML Pipeline Verification

## 1. Model Architecture & Checkpoints

### Primary Model: SegFormer-B0
- **Identifier:** `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
- **Architecture:** Transformer-based Hierarchical Encoder-Decoder (`MiT-B0` + All-MLP Decoder).
- **Parameters:** ~3.7 Million parameters.
- **Pretrained Dataset:** Cityscapes (19 classes + void/ignore index 255).
- **Input Conditioning:** Default config sets `input_size: [1024, 1024]`, with flexibility to run at `[512, 512]` for memory-constrained environments.
- **Logit Upsampling:** Bilinear interpolation with `align_corners=False` scales model logits from output feature resolution back to the original image dimensions before taking `argmax(dim=1)`.
- **Latency on CPU:** ~1.4 - 1.8 seconds per frame on single-thread/multi-core x86 CPU.

### Secondary Model: DeepLabV3-MobileNetV3
- **Identifier:** `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`
- **Pretrained Dataset:** PASCAL VOC (21 classes including background).
- **Status:** Supported via unified `RoadVisionSegmentor(architecture="deeplabv3")` interface. Note: Cityscapes quantitative benchmarks use SegFormer.

---

## 2. Label Mapping Alignment Verification

The 19 classes configured in `src/constants.py` and `configs/default_config.yaml` match the official Cityscapes `trainIds` taxonomy:

| Class ID | Class Name | Model Output Label | Status |
|----------|------------|--------------------|--------|
| 0 | road | road | Verified Match |
| 1 | sidewalk | sidewalk | Verified Match |
| 2 | building | building | Verified Match |
| 3 | wall | wall | Verified Match |
| 4 | fence | fence | Verified Match |
| 5 | pole | pole | Verified Match |
| 6 | traffic light | traffic light | Verified Match |
| 7 | traffic sign | traffic sign | Verified Match |
| 8 | vegetation | vegetation | Verified Match |
| 9 | terrain | terrain | Verified Match |
| 10 | sky | sky | Verified Match |
| 11 | person | person | Verified Match |
| 12 | rider | rider | Verified Match |
| 13 | car | car | Verified Match |
| 14 | truck | truck | Verified Match |
| 15 | bus | bus | Verified Match |
| 16 | train | train | Verified Match |
| 17 | motorcycle | motorcycle | Verified Match |
| 18 | bicycle | bicycle | Verified Match |

Ignore Index: `255` is strictly excluded from confusion matrix updates.

---

## 3. Edge Case Handling Verification
- **Grayscale Inputs:** Automatically converted to 3-channel RGB via `pil_img.convert("RGB")`.
- **RGBA Inputs:** Alpha channel stripped and converted to RGB.
- **Arbitrary Aspect Ratios:** Bilinear upsampling restores the exact original `(height, width)` aspect ratio without geometric distortion.
- **Batch Processing:** `predict_batch` method supports list inputs and parallel tensor forward passes.

## 4. Summary Verdict
- **ML Pipeline Status:** **PASS**
