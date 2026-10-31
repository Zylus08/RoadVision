# RoadVision: Technical Methodology

## 1. Introduction to Road-Scene Semantic Segmentation

Semantic segmentation is a dense pixel-level classification task where every pixel $p \in \mathbb{R}^{H \times W}$ in an input image is assigned a categorical label $c \in \{0, 1, \dots, C-1\}$. In autonomous-driving systems, semantic segmentation provides critical scene comprehension by separating drivable space (road) from walkable space (sidewalk), vulnerable road users (pedestrians, riders), moving obstacles (cars, buses, trucks, motorcycles, bicycles), and static navigational cues (traffic lights, traffic signs, poles, buildings, vegetation, sky).

Unlike **bounding-box object detection** (which predicts coarse coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ and cannot delineate irregular surfaces such as curved lane boundaries or amorphous terrain) or **instance segmentation** (which distinguishes separate instances of countable objects), semantic segmentation provides complete pixel-level spatial layout of countable things (*objects*) and uncountable stuff (*background/surfaces*).

---

## 2. Model Architecture & Backbone Selection

RoadVision provides a dual-model framework optimized for accuracy, edge efficiency, and viva demonstration:

### 2.1 Primary Architecture: SegFormer (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`)

SegFormer (Xie et al., NeurIPS 2021) is a lightweight, hierarchically structured Vision Transformer designed specifically for efficient semantic segmentation.

- **Encoder (Mix Transformer - MiT-B0):**
  - Uses hierarchical multi-scale feature representations at $\frac{1}{4}$, $\frac{1}{8}$, $\frac{1}{16}$, and $\frac{1}{32}$ resolutions, capturing fine boundaries (poles, pedestrians) and large receptive fields (roads, sky) simultaneously.
  - Replaces traditional positional encodings with Mix-FFN ($3 \times 3$ depthwise convolutions with Zero-padding), making the transformer resolution-invariant.
  - Employs spatial reduction attention ($R=8, 4, 2, 1$) to reduce computational complexity from $\mathcal{O}(N^2)$ to linear in sequence length.
- **Decoder (Lightweight All-MLP):**
  - Unifies multi-scale features through simple linear projection layers.
  - Upsamples and concatenates the four feature maps to $\frac{1}{4}$ resolution.
  - Projects directly to the $C=19$ Cityscapes semantic classes using a $1 \times 1$ convolution.
  - Contains only **3.7 million parameters**, offering rapid inference and low memory footprint.

### 2.2 Secondary Benchmark Architecture: DeepLabV3-MobileNetV3

- Utilizes Atrous Spatial Pyramid Pooling (ASPP) with parallel dilated convolutions at multiple sampling rates ($r \in \{1, 12, 24, 36\}$) and global average pooling.
- MobileNetV3-Large inverted residual bottleneck backbone with squeeze-and-excitation blocks for lightweight compute.

---

## 3. Dataset Taxonomy & Label Mapping

The system follows the official **Cityscapes 19-class benchmark taxonomy**:

| Class ID | Semantic Class | Category | Official Palette (RGB) |
|---|---|---|---|
| 0 | Road | Flat | `(128, 64, 128)` |
| 1 | Sidewalk | Flat | `(244, 35, 232)` |
| 2 | Building | Construction | `(70, 70, 70)` |
| 3 | Wall | Construction | `(102, 102, 156)` |
| 4 | Fence | Construction | `(190, 153, 153)` |
| 5 | Pole | Object | `(153, 153, 153)` |
| 6 | Traffic Light | Object | `(250, 170, 30)` |
| 7 | Traffic Sign | Object | `(220, 220, 0)` |
| 8 | Vegetation | Nature | `(107, 142, 35)` |
| 9 | Terrain | Nature | `(152, 251, 152)` |
| 10 | Sky | Sky | `(70, 130, 180)` |
| 11 | Person | Human | `(220, 20, 60)` |
| 12 | Rider | Human | `(255, 0, 0)` |
| 13 | Car | Vehicle | `(0, 0, 142)` |
| 14 | Truck | Vehicle | `(0, 0, 70)` |
| 15 | Bus | Vehicle | `(0, 60, 100)` |
| 16 | Train | Vehicle | `(0, 80, 100)` |
| 17 | Motorcycle | Vehicle | `(0, 0, 230)` |
| 18 | Bicycle | Vehicle | `(119, 11, 32)` |

- **Ignore Index:** $255$ is assigned to void, unlabeled boundaries, hood of the ego-vehicle, and ego-vehicle reflections.

---

## 4. Inference Pipeline

```
  Input Image / Dashcam Frame (H, W, 3)
                │
                ▼
  Resolution Conditioning (512, 512)
                │
                ▼
  RGB Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                │
                ▼
  SegFormer / DeepLabV3 Forward Pass (torch.inference_mode())
                │
                ▼
  Logits Tensor: (1, 19, H/4, W/4)
                │
                ▼
  Bilinear Upsampling to Native Resolution (H, W)
                │
                ▼
  Argmax Class Prediction Mask: (H, W)
                │
                ├────────────────────────┬────────────────────────┐
                ▼                        ▼                        ▼
       Color-Coded Mask          Alpha-Blended Overlay       Telemetry HUD
```

---

## 5. Environmental Corruption Pipeline

To evaluate safety and robustness under out-of-distribution environmental shifts, RoadVision implements five deterministic corruption functions parameterized by severity $s \in \{1, 2, 3, 4, 5\}$:

### 5.1 Low Illumination (Dusk to Night)
Simulates non-linear camera response under low-light night driving:
$$I'(x) = \alpha(s) \cdot \left(\frac{I(x)}{255.0}\right)^{\gamma(s)} \cdot 255.0$$
where $\alpha \in [0.80, 0.15]$ models overall illumination loss and $\gamma \in [1.25, 2.70]$ compresses shadow details.

### 5.2 Atmospheric Fog (Koschmieder Model)
Based on Koschmieder's physical scattering equation:
$$I_{\text{fog}}(x) = I(x) \cdot t(x) + L \cdot (1 - t(x))$$
where:
$$t(x) = \exp(-\beta(s) \cdot d(x))$$
- $\beta(s) \in [0.8, 3.8]$ is the atmospheric attenuation coefficient.
- $d(x)$ is the normalized camera depth gradient ($1.0$ at horizon, $0.15$ near bumper).
- $L = 220$ is the atmospheric airlight vector.

### 5.3 Synthetic Rain Streaks
Generates directional motion-blurred rain streaks:
$$I_{\text{rain}}(x) = \delta(s) \cdot I(x) + 0.7 \cdot M_{\text{streak}}(x)$$
where $M_{\text{streak}}$ is created by convolving random drop positions with a directional motion blur kernel oriented at $-35^\circ$ to $-40^\circ$.

### 5.4 Gaussian Blur (Defocus / Lens Smear)
Simulates optical defocus or camera vibration using 2D Gaussian kernel filtering:
$$I_{\text{blur}} = I * G_{\sigma(s)}$$
with kernel size $k \in \{5, 9, 15, 21, 31\}$ and standard deviation $\sigma \in [1.2, 8.0]$.

### 5.5 Gaussian Noise (Sensor Electronic Noise)
Models sensor thermal noise and high ISO grain in low ambient lux:
$$I_{\text{noise}} = \text{clip}(I + \mathcal{N}(0, \sigma(s)^2), 0, 255)$$
with standard deviation $\sigma \in [8.0, 75.0]$.

---

## 6. Quantitative Evaluation Metrics

### 6.1 Confusion Matrix Formation
Let $C \in \mathbb{Z}^{19 \times 19}$ be the accumulated confusion matrix across evaluated pixels, where:
- Row $i$: True Ground Truth class.
- Column $j$: Model Predicted class.
- Invalid pixels ($y = 255$) are strictly excluded.

### 6.2 Per-Class Intersection over Union (IoU)
$$\text{IoU}_i = \frac{TP_i}{TP_i + FP_i + FN_i} = \frac{C_{ii}}{\sum_{j=0}^{K-1} C_{ij} + \sum_{j=0}^{K-1} C_{ji} - C_{ii}}$$

### 6.3 Mean Intersection over Union (mIoU)
$$\text{mIoU} = \frac{1}{|S_{\text{present}}|} \sum_{i \in S_{\text{present}}} \text{IoU}_i$$

### 6.4 Pixel Accuracy
$$\text{Pixel Accuracy} = \frac{\sum_{i=0}^{K-1} C_{ii}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} C_{ij}}$$

### 6.5 Robustness Degradation Metrics
$$\Delta \text{mIoU}_{\text{abs}} = \text{mIoU}_{\text{clean}} - \text{mIoU}_{\text{corrupted}}$$
$$\text{Relative Drop (\%)} = \frac{\text{mIoU}_{\text{clean}} - \text{mIoU}_{\text{corrupted}}}{\text{mIoU}_{\text{clean}}} \times 100\%$$

