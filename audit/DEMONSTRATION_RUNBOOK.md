# RoadVision Demonstration Runbook

This step-by-step guide is designed for evaluators, professors, and technical examiners to independently verify and demonstrate RoadVision.

---

## 1. Environment Activation & Test Suite Verification

### Step 1.1: Open Terminal & Activate Environment
```powershell
cd E:\RoadVision
conda activate reswork
```

### Step 1.2: Run the Complete Automated Test Suite
Execute the full automated test suite (23 tests across 5 test files):
```powershell
python -m pytest tests/ -v
```
**Expected Outcome:**
`23 passed in ~50s` with zero errors or failures.

---

## 2. Quantitative Metric Reproduction CLI

### Step 2.1: Reproduce Clean Baseline Metrics
```powershell
python scripts/reproduce_clean_eval.py
```
**Expected Console Output:**
```
Dataset-Level Aggregate mIoU (from ConfMat):    38.98%
Reported Value in Previous Audit:               38.98%
Exact Match with Reported Value:                True
Aggregate Pixel Accuracy:                       94.47%
Reported Value in Previous Audit:               94.47%
Exact Match with Reported Value:                True
```

### Step 2.2: Run Single Image Inference CLI
```powershell
python scripts/run_inference.py --image data/samples/authentic/frankfurt_000000_000294.png --output-dir outputs/predictions
```
**Expected Outcome:**
Outputs saved:
- `outputs/predictions/frankfurt_000000_000294_mask.png`
- `outputs/predictions/frankfurt_000000_000294_overlay.png`
- `outputs/predictions/frankfurt_000000_000294_telemetry.json`

---

## 3. Interactive Web Dashboard Demonstration

### Step 3.1: Launch Streamlit
```powershell
streamlit run app.py
```
Streamlit will open the application at `http://localhost:8501`.

### Step 3.2: Walkthrough Script
1. **Section A (Image Segmentation):**
   - Select any sample scene (e.g. Frankfurt Scene 000294) from the dropdown.
   - Click "Run Segmentation".
   - View the triple view (Input, Colorized Semantic Mask, Overlay) and inspect the scene class proportions (Road ~30%, Building ~39%, Sky ~2%).
2. **Section B (Video Analysis):**
   - Upload or select a sample dashcam clip.
   - Set Frame Skip $k = 3$ for fast inference.
   - Click "Process Video". Observe real-time progress and watch the final annotated H.264 video play directly in the browser with live HUD telemetry.
3. **Section C (Robustness Laboratory):**
   - Select Atmospheric Fog, Severity 4.
   - Click "Simulate Weather & Segment".
   - Inspect the absolute difference map showing misclassified distant road objects.
4. **Section D (Model Evaluation & Benchmarks):**
   - View the verified 25-condition degradation curves and per-class IoU bar charts.
