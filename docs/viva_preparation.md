# RoadVision: Semester Viva & Oral Examination Preparation Guide

This document contains concise, technically precise answers to the 15 standard viva questions for the RoadVision autonomous road-scene understanding project.

---

### Q1: What is semantic segmentation?
**Answer:** Semantic segmentation is a dense pixel-wise computer vision classification task where every pixel in an image is assigned a discrete semantic class label (e.g., road, sidewalk, vehicle, pedestrian, sky) without distinguishing between individual object instances.

### Q2: How does semantic segmentation differ from object detection?
**Answer:** Object detection identifies objects using axis-aligned 2D bounding boxes $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ along with class labels. In contrast, semantic segmentation classifies every individual pixel, providing exact boundary delineation for non-rectangular and continuous classes (such as roads, lane lines, terrain, and sidewalks) which cannot be accurately bounded by boxes.

### Q3: Why did you select this architecture (SegFormer)?
**Answer:** SegFormer (specifically SegFormer-B0) was selected because:
1. It replaces traditional fixed positional encodings with Mix-FFN (depthwise convolutions), allowing resolution-invariant inference.
2. It uses hierarchical multi-scale feature representations ($\frac{1}{4}$ to $\frac{1}{32}$), capturing both fine boundaries (poles, traffic signs) and large spatial contexts (roads, buildings).
3. It uses a lightweight All-MLP decoder with only 3.7 million parameters, achieving real-time inference speed with low memory consumption.

### Q4: What is the role of the backbone?
**Answer:** The backbone (e.g., MiT-B0 in SegFormer or MobileNetV3 in DeepLabV3) acts as the feature extractor. It processes raw RGB pixel values through sequential hierarchical layers to extract low-level features (edges, textures, colors) and abstract high-level semantic representations (object shapes, contextual relationships).

### Q5: What is an encoder-decoder architecture?
**Answer:** An encoder-decoder architecture consists of two sub-networks:
- The **encoder** progressively downsamples the spatial resolution while increasing channel depth to capture large receptive fields and global context.
- The **decoder** progressively upsamples the compact feature representations back toward the original image resolution, fusing multi-scale feature maps to recover fine spatial boundaries.

### Q6: What is IoU (Intersection over Union)?
**Answer:** Intersection over Union (also known as the Jaccard Index) measures the spatial overlap between the predicted segmentation mask ($P$) and the ground truth mask ($G$):
$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$
where $TP$ is True Positives, $FP$ is False Positives, and $FN$ is False Negatives.

### Q7: Why is mIoU (Mean IoU) useful?
**Answer:** Mean IoU averages the IoU scores across all evaluated classes:
$$\text{mIoU} = \frac{1}{C} \sum_{i=1}^C \text{IoU}_i$$
It prevents class imbalance bias. In road scenes, road and sky cover 70%+ of the pixels, while pedestrians or traffic lights cover less than 1%. Global pixel accuracy can be 90% even if the model detects zero pedestrians. mIoU penalizes failure on small, critical classes equally.

### Q8: What is a confusion matrix in semantic segmentation?
**Answer:** A $C \times C$ matrix where element $C_{ij}$ represents the total count of pixels whose ground truth label is class $i$ and whose predicted label is class $j$. Diagonal elements $C_{ii}$ represent True Positives, row sums minus diagonal represent False Negatives, and column sums minus diagonal represent False Positives.

### Q9: How does fog affect semantic segmentation?
**Answer:** Atmospheric fog acts according to Koschmieder's physical scattering model: $I(x) = I_0(x) t(x) + L(1 - t(x))$. It reduces contrast exponentially with depth ($t(x) = \exp(-\beta d(x))$) and shifts pixel values toward atmospheric airlight ($L$), causing distant road boundaries, vehicles, and traffic signs to blend into the background, leading to false negatives.

### Q10: Why can pixel accuracy be misleading?
**Answer:** Pixel accuracy ($\frac{\sum TP}{\text{Total Pixels}}$) is heavily distorted by extreme class imbalance. In autonomous driving, background classes like road, buildings, and sky dominate the scene. A trivial model that predicts only "road" and "building" could achieve 85% pixel accuracy while completely failing to detect pedestrians, cyclists, or obstacles.

### Q11: How is inference latency measured?
**Answer:** Latency is measured by timing the neural network forward pass strictly:
$$\text{Latency} = t_{\text{end}} - t_{\text{start}}$$
using high-resolution timers (`time.perf_counter()`), in `torch.inference_mode()`, excluding disk I/O and display rendering. End-to-end latency additionally includes image preprocessing, colorization, alpha-blending, and video encoding.

### Q12: What are the limitations of synthetic corruption testing?
**Answer:** Synthetic corruptions use mathematical approximations (Koschmieder fog, 2D motion-blur rain streaks, additive Gaussian noise). They do not perfectly capture real-world complex physical phenomena such as road specular reflections, headlight glare, water droplets adhering to camera lenses, wiper blade occlusion, or multi-scattering atmospheric effects.

### Q13: Did you train, fine-tune, or use a pretrained model?
**Answer:** We utilized the official pretrained `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` checkpoint (and torchvision's `deeplabv3_mobilenet_v3_large`). Pretraining was conducted on the Cityscapes autonomous driving dataset. In our engineering project, we designed and implemented the complete inference engine, video pipeline, deterministic environmental corruption benchmark, quantitative confusion matrix evaluation framework, and interactive demonstration dashboard.

### Q14: What are the system's limitations?
**Answer:**
1. Inference on CPU achieves ~1.7 to 2.0 FPS for high-resolution images; hardware acceleration (CUDA GPU / TensorRT) is required for 30+ FPS real-time driving.
2. Temporal consistency: Individual frames are segmented independently without temporal recurrence (LSTM/GRU/optical flow), which can lead to minor inter-frame flicker in video streams.
3. Severe low-light conditions cause significant contrast degradation on distant road boundaries.

### Q15: What improvements would be needed for real-world autonomous driving?
**Answer:**
1. **Multi-sensor fusion:** Combine camera RGB with LiDAR point clouds and radar to overcome optical degradation in fog, rain, and darkness.
2. **Temporal modeling:** Incorporate video temporal consistency (e.g., optical flow or spatio-temporal transformers) to stabilize mask predictions across consecutive frames.
3. **Hardware acceleration:** Export the model to ONNX / TensorRT with FP16/INT8 quantization for embedded automotive deployment (e.g., NVIDIA Jetson / DRIVE Orin).
4. **Adverse weather domain adaptation:** Fine-tune using domain adaptation on real adverse datasets like Cityscapes-C, ACDC (Adverse Conditions Dataset with Correspondences), or BDD100K.

