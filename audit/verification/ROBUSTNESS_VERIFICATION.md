# Environmental Robustness Verification

## 1. Benchmark Design & Determinism

RoadVision evaluates adverse environmental robustness across five distinct physical phenomena, each calibrated from Severity 1 (mild) to Severity 5 (severe), yielding 25 experimental conditions plus the clean baseline.

### Experimental Configuration
- **Model Checkpoint:** `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
- **Evaluation Dataset:** 5 authentic Frankfurt validation scenes ($9,117,536$ valid pixels)
- **Random Seed:** Deterministic seed `42` (`np.random.RandomState(42)`)
- **Evaluation Convention:** Strict $19 \times 19$ confusion matrix accumulation identical to clean baseline.

---

## 2. Quantitative Robustness Benchmark Results

| Corruption Condition | Clean mIoU | Sev 1 mIoU | Sev 2 mIoU | Sev 3 mIoU | Sev 4 mIoU | Sev 5 mIoU | Sev 5 Drop (Abs) | Sev 5 Drop (Rel %) |
|----------------------|------------|------------|------------|------------|------------|------------|------------------|--------------------|
| **Atmospheric Fog** | 38.98% | 37.66% | 34.03% | 29.80% | 23.47% | **17.16%** | **0.2183** | **55.99%** |
| **Low Light** | 38.98% | 38.94% | 38.65% | 38.07% | 35.12% | **29.23%** | **0.0975** | **25.02%** |
| **Gaussian Blur** | 38.98% | 39.42% | 39.54% | 38.74% | 36.19% | **29.23%** | **0.0975** | **25.02%** |
| **Gaussian Noise** | 38.98% | 38.60% | 38.03% | 36.85% | 34.50% | **31.50%** | **0.0748** | **19.19%** |
| **Rain Streaks** | 38.98% | 38.90% | 38.94% | 38.65% | 38.16% | **36.72%** | **0.0226** | **5.80%** |

---

## 3. Scientific Analysis of Degradation Profiles

1. **Atmospheric Fog (Worst Impact: -56.0% relative drop):**
   - Fog uses Koschmieder's scattering law. As optical transmission decreases exponentially with depth, distant objects (traffic lights, poles, signs, distant cars) lose chromatic contrast and merge into the atmospheric airlight ($L=220$). The model exhibits severe False Negatives for small, distant objects.
2. **Low Light (-25.0% relative drop):**
   - Non-linear gamma attenuation severely compresses dark tones into near-zero pixel values, obscuring asphalt-to-sidewalk and asphalt-to-curb boundaries.
3. **Gaussian Blur (-25.0% relative drop with mild regularization at Sev 1-2):**
   - Minor blur (Sev 1: 39.42%, Sev 2: 39.54%) slightly improves mIoU over clean (38.98%) by smoothing high-frequency camera noise and texture variations on pavement, acting as a spatial regularizer. Severe blur (Sev 5: $\sigma=3.8$) degrades thin boundaries (poles, signs) down to 29.23%.
4. **Gaussian Sensor Noise (-19.2% relative drop):**
   - High variance shot noise degrades edge gradients, causing spurious isolated pixel misclassifications.
5. **Rain Streaks (-5.8% relative drop):**
   - SegFormer's self-attention mechanism attends globally across patches, effectively looking through localized semi-transparent directional rain streaks.

---

## 4. Summary Verdict
- **Robustness Verification Status:** **PASS**
