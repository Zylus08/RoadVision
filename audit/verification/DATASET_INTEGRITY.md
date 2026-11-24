# Dataset Integrity Verification

## 1. Provenance and Integrity of Authentic Validation Scenes

The evaluation subset contains five authentic Frankfurt road scenes from the official Cityscapes validation set (provenance: `tackhwa/cityscapes` repository on HuggingFace).

### Scene-by-Scene Integrity Inventory

| Scene Stem | Image File SHA256 (prefix) | Label File SHA256 (prefix) | Resolution | Valid Pixels | Void Pixels (%) | Unique Class IDs |
|------------|----------------------------|----------------------------|------------|--------------|-----------------|------------------|
| `frankfurt_000000_000294` | `3957f8cbb5f5ca48` | `a764591ffe0ab3c2` | $2048 \times 1024$ | 1,854,410 | 11.57% | `[0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 255]` (10 classes) |
| `frankfurt_000000_000576` | `3306db3645b23b57` | `8b4f30a7b4db5fb8` | $2048 \times 1024$ | 1,939,949 | 7.50% | `[0, 1, 2, 4, 5, 7, 8, 10, 12, 13, 18, 255]` (11 classes) |
| `frankfurt_000000_001016` | `4859f13d80b27be8` | `d8422ed8e338a61f` | $2048 \times 1024$ | 1,675,548 | 20.10% | `[0, 1, 2, 5, 7, 8, 11, 12, 13, 18, 255]` (10 classes) |
| `frankfurt_000000_001236` | `1e7b2a096cce9a71` | `f1416e3ab7d765e6` | $2048 \times 1024$ | 1,777,698 | 15.23% | `[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 255]` (14 classes) |
| `frankfurt_000000_001751` | `fb7c6d66e511475c` | `2dc4bc4ea445b9d4` | $2048 \times 1024$ | 1,869,931 | 10.83% | `[0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 18, 255]` (13 classes) |

- **Total Valid Pixels Across 5 Scenes:** $9,117,536$
- **Total Void Pixels Excluded:** $1,368,224$ ($13.05\%$ of total $10,485,760$ pixels)
- **Pairwise Uniqueness Check:** All 5 label mask SHA256 hashes are unique. No dummy masks or duplicates exist.
- **Label Alignment:** Class values map directly to `trainIds` $[0..18]$ plus $255$ ignore index.

---

## 2. Summary Verdict
- **Dataset Integrity Status:** **PASS**
