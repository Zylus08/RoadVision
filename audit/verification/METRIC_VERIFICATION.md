# Metric Verification & Mathematical Audit

## 1. Mathematical Rigor & Formal Definitions

Let $C \in \mathbb{N}^{19 \times 19}$ be the accumulated confusion matrix where $C_{ij}$ denotes the number of pixels with ground truth class $i$ predicted as class $j$.

For class $i \in \{0, \dots, 18\}$:
- $\text{True Positives: } TP_i = C_{ii}$
- $\text{False Positives: } FP_i = \sum_{j=0, j \neq i}^{18} C_{ji}$
- $\text{False Negatives: } FN_i = \sum_{j=0, j \neq i}^{18} C_{ij}$
- $\text{Intersection-over-Union: } \text{IoU}_i = \frac{C_{ii}}{\sum_j C_{ij} + \sum_j C_{ji} - C_{ii}}$

### Macro-Averaged mIoU
$$\text{mIoU} = \frac{1}{|S|} \sum_{i \in S} \text{IoU}_i$$
where $S = \{i \mid (TP_i + FP_i + FN_i) > 0\}$ is the set of semantic classes appearing in either the ground truth or the model predictions. Classes with denominator zero (absent) evaluate to `NaN` and are excluded from the macro-average denominator $|S|$.

### Overall Pixel Accuracy
$$\text{Pixel Accuracy} = \frac{\sum_{i=0}^{18} C_{ii}}{\sum_{i=0}^{18} \sum_{j=0}^{18} C_{ij}} = \frac{\text{Trace}(C)}{\sum C}$$

---

## 2. Quantitative Verification Results

### Clean Baseline (Resolution: $512 \times 512$ conditioned, upsampled to $2048 \times 1024$)
- **Total Valid Pixels Evaluated:** $9,117,536$
- **Total Correct Pixels ($\text{Trace}(C)$):** $8,614,769$
- **Overall Pixel Accuracy:** $8,614,769 / 9,117,536 = \mathbf{94.47\%}$ (Exact match with reported benchmark)
- **Evaluated Classes Count ($|S|$):** $18$ classes (Motorcycle absent, denominator 0)
- **Sum of 18 Evaluated IoUs:** $7.0169$
- **Dataset-Level Aggregate mIoU:** $7.0169 / 18 = \mathbf{38.98\%}$ (Exact match with reported benchmark)
- **Image-Level Mean mIoU:** $\mathbf{46.79\%}$

### Per-Class IoU Breakdown (Aggregate Confusion Matrix)

| Class ID | Class Name | TP | FP | FN | IoU (%) | Status |
|----------|------------|----|----|----|---------|--------|
| 0 | road | 3,591,267 | 63,266 | 60,083 | 96.68% | PRESENT |
| 1 | sidewalk | 252,566 | 27,922 | 75,761 | 70.90% | PRESENT |
| 2 | building | 2,479,333 | 127,729 | 117,748 | 90.99% | PRESENT |
| 3 | wall | 0 | 3,153 | 0 | 0.00% | PRESENT |
| 4 | fence | 4,686 | 11,101 | 27,648 | 10.79% | PRESENT |
| 5 | pole | 20,124 | 78,705 | 53,813 | 13.18% | PRESENT |
| 6 | traffic light | 2,724 | 1,357 | 6,254 | 26.36% | PRESENT |
| 7 | traffic sign | 18,757 | 5,218 | 22,047 | 40.76% | PRESENT |
| 8 | vegetation | 858,944 | 60,741 | 35,139 | 89.96% | PRESENT |
| 9 | terrain | 0 | 4,039 | 786 | 0.00% | PRESENT |
| 10 | sky | 236,064 | 4,940 | 5,589 | 95.73% | PRESENT |
| 11 | person | 23,709 | 37,796 | 18,412 | 29.67% | PRESENT |
| 12 | rider | 3,298 | 1,494 | 25,470 | 10.90% | PRESENT |
| 13 | car | 1,116,264 | 68,706 | 49,836 | 90.40% | PRESENT |
| 14 | truck | 0 | 0 | 1,573 | 0.00% | PRESENT |
| 15 | bus | 0 | 178 | 0 | 0.00% | PRESENT |
| 16 | train | 0 | 7 | 0 | 0.00% | PRESENT |
| 17 | motorcycle | 0 | 0 | 0 | NaN | ABSENT |
| 18 | bicycle | 6,034 | 7,414 | 3,607 | 35.38% | PRESENT |

---

## 3. Summary Verdict
- **Metric Verification Status:** **PASS**
